#!/usr/bin/env python3
"""Teardown CloudFormation stacks by walking export/import dependencies.

This script deletes CloudFormation stacks in safe rounds: importers first,
exporters only after their exports are no longer imported. It cannot break
cycles created by Fn::ImportValue; those are reported as blockers.
"""

from __future__ import annotations

import argparse
import json
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


def describe_stack(cloudformation: Any, identifier: str) -> Optional[StackState]:
    try:
        response = cloudformation.describe_stacks(StackName=identifier)
    except ClientError as exc:
        code = get_error_code(exc)
        message = get_error_message(exc).lower()
        if code == "ValidationError" and "does not exist" in message:
            return None
        raise
    stack = response["Stacks"][0]
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
    parser.add_argument("--apply", action="store_true", help="Delete stacks. Dry-run if omitted.")
    parser.add_argument("--force", action="store_true", help="Skip interactive confirmation in apply mode.")
    parser.add_argument("--allow-non-dev-profile", action="store_true", help="Allow profile names that do not end with -dev.")
    parser.add_argument("--max-rounds", type=int, default=6, help="Maximum dependency delete rounds.")
    parser.add_argument("--round-wait-seconds", type=int, default=20, help="Wait between rounds after deletes.")
    parser.add_argument("--delete-timeout-seconds", type=int, default=1800, help="Timeout per stack delete.")
    parser.add_argument("--poll-seconds", type=int, default=15, help="Poll interval while waiting for stack deletes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_profile_is_safe(args.profile, args.allow_non_dev_profile)
    stack_identifiers = load_stack_identifiers(args)
    if not stack_identifiers:
        log("No CloudFormation stacks supplied. Nothing to do.")
        return

    if args.apply and not args.force:
        require_confirmation(stack_identifiers, args.profile, args.region)

    session = create_boto3_session(args.profile)
    cloudformation = session.client("cloudformation", region_name=args.region, config=AWS_CLIENT_CONFIG)
    sts = session.client("sts", region_name=args.region, config=AWS_CLIENT_CONFIG)
    identity = sts.get_caller_identity()
    log(f"AWS account (caller identity): {identity.get('Account', 'unknown')}")
    log(f"Profile: {args.profile}")
    log(f"Region: {args.region}")
    log(f"Mode: {'apply' if args.apply else 'dry-run'}")

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
