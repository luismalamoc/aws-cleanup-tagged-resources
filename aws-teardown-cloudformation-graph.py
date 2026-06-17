#!/usr/bin/env python3
"""Teardown CloudFormation stacks by walking export/import dependencies.

This script deletes CloudFormation stacks in safe rounds: importers first,
exporters only after their exports are no longer imported. It cannot break
cycles created by Fn::ImportValue; those are reported as blockers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import boto3
    from botocore.config import Config
    from botocore.credentials import JSONFileCache
    from botocore.exceptions import ClientError
except ImportError as exc:
    print("Missing dependency: boto3. Install with: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    from rich.console import Console
    from rich.table import Table
    RICH_UI = True
except ImportError:
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    RICH_UI = False


STDOUT_CONSOLE = Console(highlight=False) if Console else None
STDERR_CONSOLE = Console(stderr=True, highlight=False) if Console else None

AWS_CLIENT_CONFIG = Config(
    connect_timeout=10,
    read_timeout=30,
    retries={"max_attempts": 3, "mode": "standard"},
)

TERMINAL_STACK_STATUSES = {"DELETE_COMPLETE"}
IN_PROGRESS_STACK_STATUSES = {
    "CREATE_IN_PROGRESS",
    "DELETE_IN_PROGRESS",
    "REVIEW_IN_PROGRESS",
    "ROLLBACK_IN_PROGRESS",
    "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS",
    "UPDATE_IN_PROGRESS",
    "UPDATE_ROLLBACK_COMPLETE_CLEANUP_IN_PROGRESS",
    "UPDATE_ROLLBACK_IN_PROGRESS",
}

ACTIVE_STACK_STATUSES = {
    "CREATE_COMPLETE",
    "CREATE_FAILED",
    "DELETE_FAILED",
    "IMPORT_COMPLETE",
    "IMPORT_ROLLBACK_COMPLETE",
    "ROLLBACK_COMPLETE",
    "UPDATE_COMPLETE",
    "UPDATE_FAILED",
    "UPDATE_ROLLBACK_COMPLETE",
    "UPDATE_ROLLBACK_FAILED",
}

DEFAULT_TAG_KEYS: Tuple[str, ...] = (
    "STAGE",
    "stage",
    "Stage",
    "ENV",
    "env",
    "Environment",
    "environment",
)

INLINE_TEMPLATE_BODY_LIMIT_BYTES = 51200


@dataclass
class StackState:
    identifier: str
    stack_id: str
    name: str
    status: str


@dataclass
class ExportDependency:
    region: str
    exporting_stack_id: str
    exporting_stack_name: str
    export_name: str
    export_value: str
    imports: List[str]
    planned_imports: List[str]
    external_imports: List[str]


@dataclass
class RoundPlan:
    ready: List[StackState]
    blocked: List[StackState]
    dependencies: List[ExportDependency]


@dataclass
class StackTemplatePatch:
    stack: StackState
    replaced_imports: int
    removed_resources: List[str]
    removed_outputs: int
    removed_missing_role_references: int
    removed_missing_role_outputs: int
    removed_statements: int
    template_body: str


def write_out(message: str) -> None:
    if STDOUT_CONSOLE is not None:
        STDOUT_CONSOLE.print(message, markup=False, highlight=False)
        return
    print(message)


def write_err(message: str) -> None:
    if STDERR_CONSOLE is not None:
        STDERR_CONSOLE.print(message, markup=False, highlight=False)
        return
    print(message, file=sys.stderr)


def log(message: str) -> None:
    write_out(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def warn(message: str) -> None:
    write_err(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARN: {message}")


def get_error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def get_error_message(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Message", ""))


def assert_profile_is_safe(profile: str, allow_non_dev_profile: bool) -> None:
    if profile.endswith("-dev"):
        return
    if allow_non_dev_profile:
        warn(f"Using non-dev profile '{profile}' because --allow-non-dev-profile was supplied")
        return
    raise ValueError(
        f"Refusing to use profile '{profile}'. Use a -dev profile or pass --allow-non-dev-profile."
    )


def create_boto3_session(profile: str) -> Any:
    session = boto3.Session(profile_name=profile)
    try:
        provider_chain = session._session.get_component("credential_provider")
        assume_role_provider = provider_chain.get_provider("assume-role")
        if assume_role_provider is not None:
            assume_role_provider.cache = JSONFileCache(str(Path.home() / ".aws" / "cli" / "cache"))
    except Exception as exc:
        warn(f"Could not attach AWS CLI credential cache to boto3 session: {exc}")
    return session


def cloudformation_stack_name(identifier: str) -> str:
    marker = ":stack/"
    if marker in identifier:
        suffix = identifier.split(marker, 1)[1]
        return suffix.split("/", 1)[0]
    return identifier.split("/", 1)[0]


def normalize_stack_identifiers(values: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def parse_csv_values(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_protected_account_ids(values: Sequence[str]) -> Set[str]:
    protected: Set[str] = set()
    for raw in values:
        protected.update(parse_csv_values(raw))
    return protected


def load_tag_keys(raw: str) -> List[str]:
    keys = parse_csv_values(raw)
    if not keys:
        raise ValueError("At least one tag key is required")
    return keys


def load_stack_identifiers(args: argparse.Namespace) -> List[str]:
    values: List[str] = []
    values.extend(args.stack or [])
    if args.stacks:
        for raw in args.stacks.split(","):
            values.append(raw)
    if args.stacks_json:
        if args.stacks_json == "-":
            raw_text = sys.stdin.read()
        else:
            raw_text = Path(args.stacks_json).read_text(encoding="utf-8")
        payload = json.loads(raw_text)
        if isinstance(payload, list):
            values.extend(str(item) for item in payload)
        elif isinstance(payload, dict) and isinstance(payload.get("stacks"), list):
            values.extend(str(item) for item in payload["stacks"])
        else:
            raise ValueError("--stacks-json must contain a JSON list or an object with a stacks list")
    return normalize_stack_identifiers(values)


def stack_has_tag(stack: Dict[str, Any], tag_keys: Sequence[str], tag_value: str) -> bool:
    allowed_keys = {key.lower() for key in tag_keys}
    wanted_value = tag_value.strip().lower()
    for tag in stack.get("Tags", []) or []:
        key = str(tag.get("Key", "")).strip().lower()
        value = str(tag.get("Value", "")).strip().lower()
        if key in allowed_keys and value == wanted_value:
            return True
    return False


def discover_stack_identifiers_by_tag(
    cloudformation: Any,
    tag_keys: Sequence[str],
    tag_value: str,
    stack_name_prefix: str,
) -> List[str]:
    identifiers: List[str] = []
    token = ""
    while True:
        kwargs: Dict[str, Any] = {"StackStatusFilter": sorted(ACTIVE_STACK_STATUSES)}
        if token:
            kwargs["NextToken"] = token
        response = cloudformation.list_stacks(**kwargs)
        for summary in response.get("StackSummaries", []):
            stack_name = str(summary.get("StackName", "")).strip()
            if not stack_name or (stack_name_prefix and not stack_name.startswith(stack_name_prefix)):
                continue
            described = describe_stack_response(cloudformation, stack_name)
            if described is None:
                continue
            if stack_has_tag(described, tag_keys, tag_value):
                identifiers.append(stack_name)
        token = str(response.get("NextToken", "")).strip()
        if not token:
            return normalize_stack_identifiers(identifiers)


def list_cloudformation_imports(cloudformation: Any, export_name: str) -> List[str]:
    imports: List[str] = []
    token = ""
    while True:
        kwargs: Dict[str, str] = {"ExportName": export_name}
        if token:
            kwargs["NextToken"] = token
        try:
            response = cloudformation.list_imports(**kwargs)
        except ClientError as exc:
            code = get_error_code(exc)
            message = get_error_message(exc).lower()
            if code == "ValidationError" and "not imported" in message:
                return imports
            raise
        imports.extend(str(item).strip() for item in response.get("Imports", []) if str(item).strip())
        token = str(response.get("NextToken", "")).strip()
        if not token:
            return imports


def describe_stack_response(cloudformation: Any, identifier: str) -> Optional[Dict[str, Any]]:
    try:
        response = cloudformation.describe_stacks(StackName=identifier)
    except ClientError as exc:
        code = get_error_code(exc)
        message = get_error_message(exc).lower()
        if code == "ValidationError" and "does not exist" in message:
            return None
        raise
    return response["Stacks"][0]


def describe_stack(cloudformation: Any, identifier: str) -> Optional[StackState]:
    stack = describe_stack_response(cloudformation, identifier)
    if stack is None:
        return None
    return StackState(
        identifier=identifier,
        stack_id=str(stack.get("StackId", "")),
        name=str(stack.get("StackName", cloudformation_stack_name(identifier))),
        status=str(stack.get("StackStatus", "")),
    )


def describe_planned_stacks(cloudformation: Any, identifiers: Sequence[str]) -> List[StackState]:
    stacks: List[StackState] = []
    seen_names: Set[str] = set()
    for identifier in identifiers:
        stack = describe_stack(cloudformation, identifier)
        if stack is None:
            log(f"Stack already gone: {identifier}")
            continue
        if stack.name in seen_names:
            continue
        seen_names.add(stack.name)
        stacks.append(stack)
    return stacks


def list_exports_for_planned_stacks(
    cloudformation: Any,
    region: str,
    stacks: Sequence[StackState],
) -> List[ExportDependency]:
    stack_ids = {stack.stack_id for stack in stacks if stack.stack_id}
    stack_names = {stack.name for stack in stacks}
    dependencies: List[ExportDependency] = []
    token = ""
    while True:
        kwargs: Dict[str, str] = {}
        if token:
            kwargs["NextToken"] = token
        response = cloudformation.list_exports(**kwargs)
        for export in response.get("Exports", []):
            exporting_stack_id = str(export.get("ExportingStackId", "")).strip()
            exporting_stack_name = cloudformation_stack_name(exporting_stack_id)
            if exporting_stack_id not in stack_ids and exporting_stack_name not in stack_names:
                continue
            export_name = str(export.get("Name", "")).strip()
            if not export_name:
                continue
            imports = list_cloudformation_imports(cloudformation, export_name)
            if not imports:
                continue
            planned_imports = [item for item in imports if item in stack_names]
            external_imports = [item for item in imports if item not in stack_names]
            dependencies.append(
                ExportDependency(
                    region=region,
                    exporting_stack_id=exporting_stack_id,
                    exporting_stack_name=exporting_stack_name,
                    export_name=export_name,
                    export_value=str(export.get("Value", "")).strip(),
                    imports=imports,
                    planned_imports=planned_imports,
                    external_imports=external_imports,
                )
            )
        token = str(response.get("NextToken", "")).strip()
        if not token:
            return dependencies


def build_round_plan(cloudformation: Any, region: str, identifiers: Sequence[str]) -> RoundPlan:
    stacks = describe_planned_stacks(cloudformation, identifiers)
    active = [stack for stack in stacks if stack.status not in TERMINAL_STACK_STATUSES]
    dependencies = list_exports_for_planned_stacks(cloudformation, region, active)
    blocked_names = {dependency.exporting_stack_name for dependency in dependencies}
    ready = [stack for stack in active if stack.name not in blocked_names and stack.status not in IN_PROGRESS_STACK_STATUSES]
    blocked = [stack for stack in active if stack.name in blocked_names or stack.status in IN_PROGRESS_STACK_STATUSES]
    return RoundPlan(ready=ready, blocked=blocked, dependencies=dependencies)


def dependency_edges(dependencies: Sequence[ExportDependency]) -> Set[Tuple[str, str]]:
    edges: Set[Tuple[str, str]] = set()
    for dependency in dependencies:
        for importer in dependency.planned_imports:
            if importer != dependency.exporting_stack_name:
                edges.add((dependency.exporting_stack_name, importer))
    return edges


def cyclic_stack_names(dependencies: Sequence[ExportDependency]) -> Set[str]:
    edges = dependency_edges(dependencies)
    nodes = {node for edge in edges for node in edge}
    adjacency: Dict[str, Set[str]] = {node: set() for node in nodes}
    for exporter, importer in edges:
        adjacency.setdefault(exporter, set()).add(importer)

    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    cyclic: Set[str] = set()

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency.get(node, set()):
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] != indices[node]:
            return

        component: List[str] = []
        while True:
            item = stack.pop()
            on_stack.remove(item)
            component.append(item)
            if item == node:
                break
        if len(component) > 1:
            cyclic.update(component)

    for node in sorted(nodes):
        if node not in indices:
            strongconnect(node)
    return cyclic


def template_body_to_text(template_body: Any) -> str:
    if isinstance(template_body, str):
        return template_body
    return json.dumps(template_body, indent=2, sort_keys=True)


def remove_importing_policy_statements(template_text: str, import_name_suffixes: Sequence[str]) -> Tuple[str, int]:
    if not import_name_suffixes:
        return template_text, 0

    lines = template_text.splitlines(keepends=True)
    ranges: List[Tuple[int, int]] = []
    for idx, line in enumerate(lines):
        if not any(suffix in line for suffix in import_name_suffixes):
            continue
        start_idx = -1
        start_indent = 0
        for cursor in range(idx, -1, -1):
            match = re.match(r"^(\s*)-\s+Effect\s*:", lines[cursor])
            if match:
                start_idx = cursor
                start_indent = len(match.group(1))
                break
            if re.match(r"^\S", lines[cursor]) and cursor != idx:
                break
        if start_idx < 0:
            continue
        end_idx = len(lines)
        for cursor in range(start_idx + 1, len(lines)):
            if not lines[cursor].strip():
                continue
            indent = len(lines[cursor]) - len(lines[cursor].lstrip(" "))
            if indent <= start_indent:
                end_idx = cursor
                break
        ranges.append((start_idx, end_idx))

    if not ranges:
        return template_text, 0

    merged: List[Tuple[int, int]] = []
    for start_idx, end_idx in sorted(ranges):
        if merged and start_idx <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_idx))
        else:
            merged.append((start_idx, end_idx))

    keep: List[str] = []
    cursor = 0
    for start_idx, end_idx in merged:
        keep.extend(lines[cursor:start_idx])
        cursor = end_idx
    keep.extend(lines[cursor:])
    return "".join(keep), len(merged)


def iter_top_level_blocks(template_text: str, section_name: str) -> Iterable[Tuple[str, int, int, str]]:
    lines = template_text.splitlines(keepends=True)
    section_idx = -1
    for idx, line in enumerate(lines):
        if line.strip() == f"{section_name}:":
            section_idx = idx
            break
    if section_idx < 0:
        return

    section_end = len(lines)
    for idx in range(section_idx + 1, len(lines)):
        if re.match(r"^\S[^:]*:\s*$", lines[idx]):
            section_end = idx
            break

    idx = section_idx + 1
    while idx < section_end:
        line = lines[idx]
        match = re.match(r"^  ([A-Za-z0-9]+):\s*$", line)
        if not match:
            idx += 1
            continue
        logical_id = match.group(1)
        start_idx = idx
        idx += 1
        while idx < section_end:
            if re.match(r"^  [A-Za-z0-9]+:\s*$", lines[idx]):
                break
            idx += 1
        yield logical_id, start_idx, idx, "".join(lines[start_idx:idx])


def iter_top_level_resource_blocks(template_text: str) -> Iterable[Tuple[str, int, int, str]]:
    yield from iter_top_level_blocks(template_text, "Resources")


def circular_managed_policy_logical_ids(template_text: str, import_name_suffixes: Sequence[str]) -> List[str]:
    logical_ids: List[str] = []
    for logical_id, _, _, block in iter_top_level_resource_blocks(template_text):
        if "AWS::IAM::ManagedPolicy" not in block:
            continue
        if any(suffix in block for suffix in import_name_suffixes):
            logical_ids.append(logical_id)
    return logical_ids


def remove_resource_blocks(template_text: str, logical_ids: Sequence[str]) -> Tuple[str, int]:
    wanted = set(logical_ids)
    if not wanted:
        return template_text, 0
    lines = template_text.splitlines(keepends=True)
    ranges = [(start_idx, end_idx) for logical_id, start_idx, end_idx, _ in iter_top_level_resource_blocks(template_text) if logical_id in wanted]
    if not ranges:
        return template_text, 0
    keep: List[str] = []
    cursor = 0
    for start_idx, end_idx in sorted(ranges):
        keep.extend(lines[cursor:start_idx])
        cursor = end_idx
    keep.extend(lines[cursor:])
    return "".join(keep), len(ranges)


def remove_output_blocks_referencing(template_text: str, logical_ids: Sequence[str]) -> Tuple[str, int]:
    wanted = set(logical_ids)
    if not wanted:
        return template_text, 0
    lines = template_text.splitlines(keepends=True)
    ranges = []
    for _, start_idx, end_idx, block in iter_top_level_blocks(template_text, "Outputs"):
        if any(re.search(rf"\b{re.escape(logical_id)}\b", block) for logical_id in wanted):
            ranges.append((start_idx, end_idx))
    if not ranges:
        return template_text, 0
    keep: List[str] = []
    cursor = 0
    for start_idx, end_idx in sorted(ranges):
        keep.extend(lines[cursor:start_idx])
        cursor = end_idx
    keep.extend(lines[cursor:])
    return "".join(keep), len(ranges)


def missing_managed_policy_logical_ids(cloudformation: Any, iam: Any, stack_name: str, logical_ids: Sequence[str]) -> List[str]:
    missing: List[str] = []
    for logical_id in logical_ids:
        try:
            response = cloudformation.describe_stack_resource(StackName=stack_name, LogicalResourceId=logical_id)
        except ClientError:
            continue
        physical_id = str(response.get("StackResourceDetail", {}).get("PhysicalResourceId", "")).strip()
        if not physical_id.startswith("arn:aws:iam::"):
            continue
        try:
            iam.get_policy(PolicyArn=physical_id)
        except ClientError as exc:
            code = get_error_code(exc)
            if code in {"NoSuchEntity", "NoSuchEntityException", "NotFound"}:
                missing.append(logical_id)
                continue
            raise
    return missing


def iam_role_logical_ids(template_body: Any) -> List[str]:
    if not isinstance(template_body, dict):
        return []
    resources = template_body.get("Resources", {})
    if not isinstance(resources, dict):
        return []
    logical_ids: List[str] = []
    for logical_id, resource in resources.items():
        if isinstance(resource, dict) and resource.get("Type") == "AWS::IAM::Role":
            logical_ids.append(str(logical_id))
    return logical_ids


def missing_iam_role_logical_ids(cloudformation: Any, iam: Any, stack_name: str, template_body: Any) -> List[str]:
    missing: List[str] = []
    for logical_id in iam_role_logical_ids(template_body):
        try:
            response = cloudformation.describe_stack_resource(StackName=stack_name, LogicalResourceId=logical_id)
        except ClientError:
            continue
        physical_id = str(response.get("StackResourceDetail", {}).get("PhysicalResourceId", "")).strip()
        if not physical_id:
            continue
        role_name = physical_id.rsplit("/", 1)[-1]
        try:
            iam.get_role(RoleName=role_name)
        except ClientError as exc:
            code = get_error_code(exc)
            if code in {"NoSuchEntity", "NoSuchEntityException", "NotFound"}:
                missing.append(logical_id)
                continue
            raise
    return missing


def is_direct_logical_reference(value: Any, logical_ids: Set[str]) -> bool:
    if not isinstance(value, dict):
        return False
    if set(value.keys()) == {"Ref"}:
        return str(value.get("Ref", "")) in logical_ids
    get_att = value.get("Fn::GetAtt")
    if isinstance(get_att, list) and get_att:
        return str(get_att[0]) in logical_ids
    if isinstance(get_att, str):
        return get_att.split(".", 1)[0] in logical_ids
    return False


def contains_logical_reference(value: Any, logical_ids: Set[str]) -> bool:
    if is_direct_logical_reference(value, logical_ids):
        return True
    if isinstance(value, dict):
        return any(contains_logical_reference(item, logical_ids) for item in value.values())
    if isinstance(value, list):
        return any(contains_logical_reference(item, logical_ids) for item in value)
    return False


def remove_direct_list_references(template_body: Any, logical_ids: Sequence[str]) -> Tuple[Any, int]:
    wanted = set(logical_ids)
    if not wanted:
        return template_body, 0
    if isinstance(template_body, list):
        replaced_items: List[Any] = []
        removed = 0
        for item in template_body:
            if is_direct_logical_reference(item, wanted):
                removed += 1
                continue
            patched_item, child_removed = remove_direct_list_references(item, logical_ids)
            replaced_items.append(patched_item)
            removed += child_removed
        return replaced_items, removed
    if isinstance(template_body, dict):
        replaced: Dict[str, Any] = {}
        removed = 0
        for key, value in template_body.items():
            patched_value, child_removed = remove_direct_list_references(value, logical_ids)
            replaced[key] = patched_value
            removed += child_removed
        return replaced, removed
    return template_body, 0


def remove_unimported_outputs_referencing_logical_ids(
    cloudformation: Any,
    stack_name: str,
    template_body: Any,
    logical_ids: Sequence[str],
    variables: Dict[str, str],
) -> Tuple[Any, int]:
    wanted = set(logical_ids)
    if not wanted or not isinstance(template_body, dict):
        return template_body, 0
    outputs = template_body.get("Outputs")
    if not isinstance(outputs, dict):
        return template_body, 0

    patched_outputs: Dict[str, Any] = {}
    removed = 0
    for logical_id, output in outputs.items():
        if not contains_logical_reference(output, wanted):
            patched_outputs[logical_id] = output
            continue
        export_name = ""
        if isinstance(output, dict):
            export = output.get("Export")
            if isinstance(export, dict):
                export_name = resolve_import_name(export.get("Name", ""), variables)
        imports = list_cloudformation_imports(cloudformation, export_name) if export_name else []
        if imports:
            warn(f"Keeping output {stack_name}.{logical_id}; export {export_name} is still imported by {', '.join(imports)}")
            patched_outputs[logical_id] = output
            continue
        removed += 1

    patched = dict(template_body)
    if patched_outputs:
        patched["Outputs"] = patched_outputs
    else:
        patched.pop("Outputs", None)
    return patched, removed


def patch_missing_iam_role_references(
    cloudformation: Any,
    iam: Any,
    stack_name: str,
    template_body: Any,
    variables: Dict[str, str],
) -> Tuple[Any, int, int]:
    missing_roles = missing_iam_role_logical_ids(cloudformation, iam, stack_name, template_body)
    patched_body, removed_references = remove_direct_list_references(template_body, missing_roles)
    patched_body, removed_outputs = remove_unimported_outputs_referencing_logical_ids(
        cloudformation,
        stack_name,
        patched_body,
        missing_roles,
        variables,
    )
    return patched_body, removed_references, removed_outputs


def get_stack_template_text(cloudformation: Any, stack_name: str) -> str:
    response = cloudformation.get_template(StackName=stack_name)
    return template_body_to_text(response.get("TemplateBody", ""))


def build_template_patches(
    cloudformation: Any,
    iam: Any,
    stacks: Sequence[StackState],
    import_name_suffixes: Sequence[str],
) -> List[StackTemplatePatch]:
    patches: List[StackTemplatePatch] = []
    for stack in stacks:
        template_text = get_stack_template_text(cloudformation, stack.name)
        managed_policy_ids = circular_managed_policy_logical_ids(template_text, import_name_suffixes)
        missing_policy_ids = missing_managed_policy_logical_ids(cloudformation, iam, stack.name, managed_policy_ids)
        patched_text, removed_resources = remove_resource_blocks(template_text, missing_policy_ids)
        patched_text, removed_outputs = remove_output_blocks_referencing(patched_text, missing_policy_ids)
        patched_text, removed_statements = remove_importing_policy_statements(patched_text, import_name_suffixes)
        if removed_resources or removed_statements:
            patches.append(
                StackTemplatePatch(
                    stack=stack,
                    replaced_imports=0,
                    removed_resources=missing_policy_ids,
                    removed_outputs=removed_outputs,
                    removed_missing_role_references=0,
                    removed_missing_role_outputs=0,
                    removed_statements=removed_statements,
                    template_body=patched_text,
                )
            )
    return patches


def existing_stack_parameters(cloudformation: Any, stack_name: str) -> List[Dict[str, Any]]:
    stack = describe_stack_response(cloudformation, stack_name)
    if stack is None:
        return []
    parameters: List[Dict[str, Any]] = []
    for parameter in stack.get("Parameters", []) or []:
        key = str(parameter.get("ParameterKey", "")).strip()
        if key:
            parameters.append({"ParameterKey": key, "UsePreviousValue": True})
    return parameters


def stack_parameter_values(cloudformation: Any, stack_name: str) -> Dict[str, str]:
    stack = describe_stack_response(cloudformation, stack_name)
    if stack is None:
        return {}
    values: Dict[str, str] = {"AWS::StackName": stack_name}
    for parameter in stack.get("Parameters", []) or []:
        key = str(parameter.get("ParameterKey", "")).strip()
        value = str(parameter.get("ParameterValue", "")).strip()
        if key:
            values[key] = value
    return values


def resolve_sub_expression(raw: Any, variables: Dict[str, str]) -> str:
    if isinstance(raw, str):
        template = raw
        replacements = variables
    elif isinstance(raw, list) and raw:
        template = str(raw[0])
        overrides = raw[1] if len(raw) > 1 and isinstance(raw[1], dict) else {}
        replacements = {**variables, **{str(key): str(value) for key, value in overrides.items()}}
    else:
        return ""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return replacements.get(key, match.group(0))

    return re.sub(r"\$\{([^}]+)\}", replace, template)


def resolve_import_name(raw: Any, variables: Dict[str, str]) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        if "Fn::Sub" in raw:
            return resolve_sub_expression(raw["Fn::Sub"], variables)
        if "Ref" in raw:
            return variables.get(str(raw["Ref"]), "")
    return ""


def replace_imports_with_literal_values(
    template_body: Any,
    export_values: Dict[str, str],
    variables: Dict[str, str],
) -> Tuple[Any, int]:
    if isinstance(template_body, dict):
        if set(template_body.keys()) == {"Fn::ImportValue"}:
            import_name = resolve_import_name(template_body["Fn::ImportValue"], variables)
            if import_name in export_values:
                return export_values[import_name], 1
        replaced: Dict[str, Any] = {}
        count = 0
        for key, value in template_body.items():
            new_value, child_count = replace_imports_with_literal_values(value, export_values, variables)
            replaced[key] = new_value
            count += child_count
        return replaced, count

    if isinstance(template_body, list):
        replaced_items: List[Any] = []
        count = 0
        for item in template_body:
            new_item, child_count = replace_imports_with_literal_values(item, export_values, variables)
            replaced_items.append(new_item)
            count += child_count
        return replaced_items, count

    return template_body, 0


def render_template_body(template_body: Any) -> str:
    if isinstance(template_body, str):
        return template_body
    return json.dumps(template_body, separators=(",", ":"), sort_keys=True)


def build_literal_import_patches(
    cloudformation: Any,
    iam: Any,
    dependencies: Sequence[ExportDependency],
    cyclic_names: Set[str],
) -> List[StackTemplatePatch]:
    exports_by_importer: Dict[str, Dict[str, str]] = {}
    for dependency in dependencies:
        if dependency.exporting_stack_name not in cyclic_names:
            continue
        for importer in dependency.planned_imports:
            if importer in cyclic_names:
                exports_by_importer.setdefault(importer, {})[dependency.export_name] = dependency.export_value

    patches: List[StackTemplatePatch] = []
    for importer, export_values in sorted(exports_by_importer.items()):
        stack = describe_stack(cloudformation, importer)
        if stack is None:
            continue
        response = cloudformation.get_template(StackName=importer)
        variables = stack_parameter_values(cloudformation, importer)
        patched_body, replaced = replace_imports_with_literal_values(
            response.get("TemplateBody", ""),
            export_values,
            variables,
        )
        patched_body, removed_role_references, removed_role_outputs = patch_missing_iam_role_references(
            cloudformation,
            iam,
            importer,
            patched_body,
            variables,
        )
        if replaced or removed_role_references or removed_role_outputs:
            patches.append(
                StackTemplatePatch(
                    stack=stack,
                    replaced_imports=replaced,
                    removed_resources=[],
                    removed_outputs=0,
                    removed_missing_role_references=removed_role_references,
                    removed_missing_role_outputs=removed_role_outputs,
                    removed_statements=0,
                    template_body=render_template_body(patched_body),
                )
            )
    return patches


def wait_for_stack_status(
    cloudformation: Any,
    stack_name: str,
    success_statuses: Set[str],
    failure_statuses: Set[str],
    timeout_seconds: int,
    poll_seconds: int,
) -> str:
    deadline = time.monotonic() + max(1, timeout_seconds)
    while time.monotonic() < deadline:
        current = describe_stack(cloudformation, stack_name)
        if current is None:
            return "deleted"
        if current.status in success_statuses:
            return current.status
        if current.status in failure_statuses:
            warn(f"Stack reached failure status: {stack_name} {current.status}")
            for line in recent_stack_events(cloudformation, stack_name):
                warn(f"  {line}")
            return "failed"
        log(f"Waiting for {stack_name}: {current.status}")
        time.sleep(max(1, poll_seconds))
    warn(f"Timed out waiting for stack status: {stack_name}")
    return "timeout"


def wait_for_stack_update(cloudformation: Any, stack_name: str, timeout_seconds: int, poll_seconds: int) -> str:
    return wait_for_stack_status(
        cloudformation,
        stack_name,
        success_statuses={"UPDATE_COMPLETE"},
        failure_statuses={"UPDATE_FAILED", "UPDATE_ROLLBACK_COMPLETE", "UPDATE_ROLLBACK_FAILED"},
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )


def continue_failed_update_rollback(
    cloudformation: Any,
    stack: StackState,
    resources_to_skip: Sequence[str],
    timeout_seconds: int,
    poll_seconds: int,
) -> bool:
    if stack.status != "UPDATE_ROLLBACK_FAILED":
        return True
    if not resources_to_skip:
        warn(f"{stack.name} is UPDATE_ROLLBACK_FAILED but no resources are available to skip")
        return False
    log(f"Continuing rollback for {stack.name}; skipping: {', '.join(resources_to_skip)}")
    try:
        cloudformation.continue_update_rollback(
            StackName=stack.stack_id or stack.name,
            ResourcesToSkip=list(resources_to_skip),
        )
    except ClientError as exc:
        warn(f"continue_update_rollback failed for {stack.name}: {get_error_code(exc)} {get_error_message(exc)}")
        return False
    result = wait_for_stack_status(
        cloudformation,
        stack.name,
        success_statuses={"UPDATE_ROLLBACK_COMPLETE"},
        failure_statuses={"UPDATE_ROLLBACK_FAILED"},
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
    return result == "UPDATE_ROLLBACK_COMPLETE"


def ensure_template_bucket(s3: Any, bucket: str, region: str) -> None:
    try:
        s3.head_bucket(Bucket=bucket)
        return
    except ClientError as exc:
        code = get_error_code(exc)
        status_code = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        if code not in {"404", "NoSuchBucket", "NotFound"} and status_code != 404:
            raise

    log(f"Creating temporary CloudFormation template bucket: {bucket}")
    kwargs: Dict[str, Any] = {"Bucket": bucket}
    if region != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket(**kwargs)
    try:
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
    except ClientError as exc:
        warn(f"Could not set public access block on {bucket}: {get_error_code(exc)} {get_error_message(exc)}")


def update_stack_template_argument(
    s3: Any,
    stack_name: str,
    template_body: str,
    account_id: str,
    region: str,
    template_bucket: str,
) -> Dict[str, str]:
    body_size = len(template_body.encode("utf-8"))
    if body_size <= INLINE_TEMPLATE_BODY_LIMIT_BYTES:
        log(f"Using inline template body for {stack_name} ({body_size} bytes)")
        return {"TemplateBody": template_body}

    bucket = template_bucket or f"cfn-teardown-{account_id}-{region}"
    key = f"cloudformation-teardown/{stack_name}-{int(time.time())}.template.json"
    ensure_template_bucket(s3, bucket, region)
    log(f"Uploading patched template for {stack_name} to s3://{bucket}/{key} ({body_size} bytes)")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=template_body.encode("utf-8"),
        ServerSideEncryption="AES256",
        ContentType="application/json",
    )
    template_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=3600,
    )
    return {"TemplateURL": template_url}


def apply_template_patches(
    cloudformation: Any,
    s3: Any,
    patches: Sequence[StackTemplatePatch],
    account_id: str,
    region: str,
    template_bucket: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> bool:
    for patch in patches:
        current = describe_stack(cloudformation, patch.stack.name)
        if current is None:
            continue
        if not continue_failed_update_rollback(
            cloudformation,
            current,
            patch.removed_resources,
            timeout_seconds,
            poll_seconds,
        ):
            return False
        log(
            f"Updating {patch.stack.name} to replace {patch.replaced_imports} circular import(s), "
            f"remove {len(patch.removed_resources)} missing managed policy resource(s), "
            f"{patch.removed_outputs} output(s), and "
            f"{patch.removed_statements} circular IAM import statement(s); "
            f"remove {patch.removed_missing_role_references} missing IAM role reference(s) "
            f"and {patch.removed_missing_role_outputs} missing IAM role output(s)"
        )
        try:
            template_argument = update_stack_template_argument(
                s3,
                patch.stack.name,
                patch.template_body,
                account_id,
                region,
                template_bucket,
            )
            cloudformation.update_stack(
                StackName=patch.stack.stack_id or patch.stack.name,
                **template_argument,
                Parameters=existing_stack_parameters(cloudformation, patch.stack.name),
                Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM", "CAPABILITY_AUTO_EXPAND"],
            )
        except ClientError as exc:
            code = get_error_code(exc)
            message = get_error_message(exc)
            if code == "ValidationError" and "no updates are to be performed" in message.lower():
                log(f"No template update needed for {patch.stack.name}")
                continue
            warn(f"update_stack failed for {patch.stack.name}: {code} {message}")
            return False
        result = wait_for_stack_update(cloudformation, patch.stack.name, timeout_seconds, poll_seconds)
        if result != "UPDATE_COMPLETE":
            warn(f"Stopping after stack update result for {patch.stack.name}: {result}")
            return False
    return True


def maybe_break_circular_iam_imports(
    cloudformation: Any,
    iam: Any,
    s3: Any,
    region: str,
    account_id: str,
    template_bucket: str,
    identifiers: Sequence[str],
    import_name_suffixes: Sequence[str],
    apply: bool,
    timeout_seconds: int,
    poll_seconds: int,
) -> Sequence[str]:
    if not import_name_suffixes:
        return identifiers

    plan = build_round_plan(cloudformation, region, identifiers)
    cyclic_names = cyclic_stack_names(plan.dependencies)
    if not cyclic_names:
        log("No circular CloudFormation export/import dependencies detected")
        return identifiers

    cyclic_stacks = [stack for stack in plan.blocked if stack.name in cyclic_names]
    literal_patches = build_literal_import_patches(cloudformation, iam, plan.dependencies, cyclic_names)
    if literal_patches:
        for patch in literal_patches:
            log(
                f"Template patch candidate: {patch.stack.name} replaces "
                f"{patch.replaced_imports} circular import(s) with literal value(s), "
                f"removes {patch.removed_missing_role_references} missing IAM role reference(s), "
                f"and removes {patch.removed_missing_role_outputs} missing IAM role output(s)"
            )
        if not apply:
            log("Dry-run only. Re-run with --apply to update templates and then delete stacks.")
            return identifiers
        if not apply_template_patches(
            cloudformation,
            s3,
            literal_patches,
            account_id,
            region,
            template_bucket,
            timeout_seconds,
            poll_seconds,
        ):
            raise SystemExit("Could not replace circular imports; aborting before delete")
        return identifiers

    patches = build_template_patches(cloudformation, iam, cyclic_stacks, import_name_suffixes)
    if not patches:
        warn("Circular dependencies found, but no matching IAM policy statements were removable")
        return identifiers

    for patch in patches:
        log(
            f"Template patch candidate: {patch.stack.name} removes "
            f"{len(patch.removed_resources)} missing managed policy resource(s) "
            f"{patch.removed_outputs} output(s), and {patch.removed_statements} statement(s) "
            f"matching {', '.join(import_name_suffixes)}"
        )

    if not apply:
        log("Dry-run only. Re-run with --apply to update templates and then delete stacks.")
        return identifiers

    if not apply_template_patches(
        cloudformation,
        s3,
        patches,
        account_id,
        region,
        template_bucket,
        timeout_seconds,
        poll_seconds,
    ):
        raise SystemExit("Could not break circular IAM imports; aborting before delete")
    return identifiers


def print_dependency_graph(dependencies: Sequence[ExportDependency]) -> None:
    if not dependencies:
        log("No CloudFormation export/import blockers found for remaining planned stacks")
        return
    warn("CloudFormation export/import blockers remain")
    if STDOUT_CONSOLE is not None and RICH_UI and Table is not None:
        table = Table(title="CloudFormation Export Dependency Graph", header_style="bold magenta")
        table.add_column("Region", style="white")
        table.add_column("Exporter", style="cyan", overflow="fold")
        table.add_column("Export", style="white", overflow="fold")
        table.add_column("Imported By", style="yellow", overflow="fold")
        table.add_column("External Importers", style="red", overflow="fold")
        for dependency in dependencies:
            table.add_row(
                dependency.region,
                dependency.exporting_stack_name,
                dependency.export_name,
                ", ".join(dependency.imports),
                ", ".join(dependency.external_imports) or "-",
            )
        STDOUT_CONSOLE.print(table)
        return
    print("CloudFormation export dependency graph:")
    for dependency in dependencies:
        external = ", ".join(dependency.external_imports) or "-"
        print(
            f"  {dependency.region}: {dependency.exporting_stack_name} exports {dependency.export_name} "
            f"-> imported by {', '.join(dependency.imports)} (external: {external})"
        )


def print_round_plan(round_idx: int, plan: RoundPlan) -> None:
    log(f"CloudFormation teardown round {round_idx}: ready={len(plan.ready)} blocked={len(plan.blocked)}")
    if plan.ready:
        print("Ready to delete:")
        for stack in plan.ready:
            print(f"  - {stack.name} ({stack.status})")
    if plan.blocked:
        print("Blocked or in-progress:")
        for stack in plan.blocked:
            print(f"  - {stack.name} ({stack.status})")


def recent_stack_events(cloudformation: Any, stack_name: str, limit: int = 5) -> List[str]:
    try:
        response = cloudformation.describe_stack_events(StackName=stack_name)
    except ClientError:
        return []
    lines: List[str] = []
    for event in response.get("StackEvents", [])[:limit]:
        logical = str(event.get("LogicalResourceId", ""))
        resource_type = str(event.get("ResourceType", ""))
        status = str(event.get("ResourceStatus", ""))
        reason = str(event.get("ResourceStatusReason", "") or "")
        lines.append(f"{logical} {resource_type} {status} {reason}".strip())
    return lines


def wait_for_stack_delete(
    cloudformation: Any,
    stack: StackState,
    timeout_seconds: int,
    poll_seconds: int,
) -> str:
    deadline = time.monotonic() + max(1, timeout_seconds)
    while time.monotonic() < deadline:
        current = describe_stack(cloudformation, stack.stack_id or stack.name)
        if current is None:
            return "deleted"
        if current.status == "DELETE_COMPLETE":
            return "deleted"
        if current.status == "DELETE_FAILED":
            warn(f"Stack delete failed: {stack.name}")
            for line in recent_stack_events(cloudformation, stack.name):
                warn(f"  {line}")
            return "failed"
        log(f"Waiting for {stack.name}: {current.status}")
        time.sleep(max(1, poll_seconds))
    warn(f"Timed out waiting for stack delete: {stack.name}")
    return "timeout"


def delete_ready_stacks(
    cloudformation: Any,
    stacks: Sequence[StackState],
    timeout_seconds: int,
    poll_seconds: int,
) -> Tuple[int, int, int]:
    deleted = 0
    failed = 0
    timed_out = 0
    for stack in stacks:
        log(f"Deleting CloudFormation stack {stack.name}")
        try:
            cloudformation.delete_stack(StackName=stack.stack_id or stack.name)
        except ClientError as exc:
            code = get_error_code(exc)
            message = get_error_message(exc)
            warn(f"delete_stack failed for {stack.name}: {code} {message}")
            failed += 1
            continue
        result = wait_for_stack_delete(cloudformation, stack, timeout_seconds, poll_seconds)
        if result == "deleted":
            deleted += 1
        elif result == "timeout":
            timed_out += 1
        else:
            failed += 1
    return deleted, failed, timed_out


def require_confirmation(stacks: Sequence[str], profile: str, region: str) -> None:
    print("You are about to DELETE CloudFormation stacks:")
    print(f"  profile: {profile}")
    print(f"  region : {region}")
    print(f"  count  : {len(stacks)}")
    for stack in stacks:
        print(f"  - {stack}")
    answer = input("Type DELETE-CFN to continue: ").strip()
    if answer != "DELETE-CFN":
        raise SystemExit("Confirmation failed; aborting")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teardown CloudFormation stacks by export/import dependency graph.")
    parser.add_argument("--profile", default="catalytic-pc-dev", help="AWS profile to use.")
    parser.add_argument("--region", default="us-east-1", help="AWS region.")
    parser.add_argument("--stack", action="append", help="Stack name or ARN. Can be repeated.")
    parser.add_argument("--stacks", default="", help="Comma-separated stack names or ARNs.")
    parser.add_argument("--stacks-json", help="Path to JSON stack list, or '-' to read JSON from stdin.")
    parser.add_argument("--tag-value", default="", help="Discover CloudFormation stacks whose tags match this value.")
    parser.add_argument("--tag-keys", default=",".join(DEFAULT_TAG_KEYS), help="Comma-separated stack tag keys for --tag-value discovery.")
    parser.add_argument("--stack-name-prefix", default="", help="Optional stack name prefix filter for --tag-value discovery.")
    parser.add_argument(
        "--break-circular-iam-import-suffix",
        action="append",
        default=[],
        help="Before delete, remove IAM policy statements importing circular exports whose names contain this suffix. Can be repeated.",
    )
    parser.add_argument("--apply", action="store_true", help="Delete stacks. Dry-run if omitted.")
    parser.add_argument("--force", action="store_true", help="Skip interactive confirmation in apply mode.")
    parser.add_argument("--allow-non-dev-profile", action="store_true", help="Allow profile names that do not end with -dev.")
    parser.add_argument(
        "--protected-account-id",
        action="append",
        default=[],
        help=(
            "AWS account ID to protect from execution. Can be repeated or passed as a comma-separated list. "
            "If omitted, no account is protected."
        ),
    )
    parser.add_argument("--max-rounds", type=int, default=6, help="Maximum dependency delete rounds.")
    parser.add_argument("--round-wait-seconds", type=int, default=20, help="Wait between rounds after deletes.")
    parser.add_argument("--delete-timeout-seconds", type=int, default=1800, help="Timeout per stack delete.")
    parser.add_argument("--poll-seconds", type=int, default=15, help="Poll interval while waiting for stack deletes.")
    parser.add_argument("--template-bucket", default="", help="Optional S3 bucket for patched templates larger than CloudFormation's inline limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_profile_is_safe(args.profile, args.allow_non_dev_profile)
    session = create_boto3_session(args.profile)
    cloudformation = session.client("cloudformation", region_name=args.region, config=AWS_CLIENT_CONFIG)
    iam = session.client("iam", region_name=args.region, config=AWS_CLIENT_CONFIG)
    s3 = session.client("s3", region_name=args.region, config=AWS_CLIENT_CONFIG)
    sts = session.client("sts", region_name=args.region, config=AWS_CLIENT_CONFIG)
    identity = sts.get_caller_identity()
    account_id = str(identity.get("Account", "unknown"))
    protected_account_ids = load_protected_account_ids(args.protected_account_id)
    if account_id in protected_account_ids:
        raise SystemExit(f"Refusing to operate on protected AWS account {account_id}")
    log(f"AWS account (caller identity): {account_id}")
    log(f"Profile: {args.profile}")
    log(f"Region: {args.region}")
    log(f"Mode: {'apply' if args.apply else 'dry-run'}")

    stack_identifiers = load_stack_identifiers(args)
    if args.tag_value:
        discovered = discover_stack_identifiers_by_tag(
            cloudformation,
            load_tag_keys(args.tag_keys),
            args.tag_value,
            args.stack_name_prefix,
        )
        log(f"Discovered {len(discovered)} stack(s) matching tag value '{args.tag_value}'")
        stack_identifiers = normalize_stack_identifiers([*stack_identifiers, *discovered])

    if not stack_identifiers:
        log("No CloudFormation stacks supplied or discovered. Nothing to do.")
        return

    if args.apply and not args.force:
        require_confirmation(stack_identifiers, args.profile, args.region)

    stack_identifiers = list(
        maybe_break_circular_iam_imports(
            cloudformation,
            iam,
            s3,
            args.region,
            account_id,
            args.template_bucket,
            stack_identifiers,
            args.break_circular_iam_import_suffix,
            args.apply,
            timeout_seconds=max(1, args.delete_timeout_seconds),
            poll_seconds=max(1, args.poll_seconds),
        )
    )

    total_deleted = 0
    total_failed = 0
    total_timed_out = 0
    current_identifiers = stack_identifiers

    for round_idx in range(1, max(1, args.max_rounds) + 1):
        plan = build_round_plan(cloudformation, args.region, current_identifiers)
        print_round_plan(round_idx, plan)
        print_dependency_graph(plan.dependencies)

        if not plan.ready:
            if plan.blocked:
                warn("No CloudFormation stacks are currently safe to delete. Break imports/cycles or remove external importers first.")
            else:
                log("All supplied CloudFormation stacks are deleted or absent.")
            break

        if not args.apply:
            log("Dry-run complete. Re-run with --apply to delete ready stacks.")
            break

        deleted, failed, timed_out = delete_ready_stacks(
            cloudformation,
            plan.ready,
            timeout_seconds=max(1, args.delete_timeout_seconds),
            poll_seconds=max(1, args.poll_seconds),
        )
        total_deleted += deleted
        total_failed += failed
        total_timed_out += timed_out

        current_identifiers = [stack.name for stack in plan.blocked]
        if not current_identifiers:
            log("CloudFormation teardown complete.")
            break
        if failed or timed_out:
            warn("Some CloudFormation stack deletes failed or timed out; stopping before the next dependency round.")
            break
        if round_idx < max(1, args.max_rounds):
            time.sleep(max(0, args.round_wait_seconds))

    log("CloudFormation teardown summary:")
    print(f"  deleted: {total_deleted}")
    print(f"  failed: {total_failed}")
    print(f"  timed_out: {total_timed_out}")


if __name__ == "__main__":
    main()
