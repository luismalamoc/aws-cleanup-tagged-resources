#!/usr/bin/env python3
"""Delete AWS resources for a stage/tag using a pre-generated inventory JSON.

The script reads resources from an inventory file (for example output from aws-list-resources),
filters by stage/tag, builds a dependency-aware deletion plan, and executes it in rounds.

Safety defaults:
- Dry-run by default (no deletes unless --apply)
- Strong confirmation prompt before delete
- Profile guard (expects profile ending in -dev unless explicitly overridden)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError as exc:
    print("Missing dependency: boto3. Install with: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from exc

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress as RichProgress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    RICH_UI = True
except ImportError:
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]
    RichProgress = None  # type: ignore[assignment]
    SpinnerColumn = None  # type: ignore[assignment]
    BarColumn = None  # type: ignore[assignment]
    TextColumn = None  # type: ignore[assignment]
    TimeElapsedColumn = None  # type: ignore[assignment]
    MofNCompleteColumn = None  # type: ignore[assignment]
    RICH_UI = False


STDOUT_CONSOLE = Console(highlight=False) if Console else None
STDERR_CONSOLE = Console(stderr=True, highlight=False) if Console else None

STATUS_STYLES: Dict[str, str] = {
    "planned": "cyan",
    "deleted": "green",
    "skipped": "yellow",
    "deferred": "magenta",
    "failed": "bold red",
    "pending": "bold red",
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

RETRYABLE_ERROR_CODES: Set[str] = {
    "BucketNotEmpty",
    "ConflictException",
    "DeleteConflict",
    "DependencyViolation",
    "InUse",
    "InvalidCacheClusterState",
    "InvalidDBInstanceState",
    "InvalidParameterValue",
    "InvalidState",
    "OperationNotPermitted",
    "RequestLimitExceeded",
    "ResourceInUse",
    "ResourceInUseException",
    "Throttling",
    "ThrottlingException",
}

GENERIC_IDENTIFIER_KEYS: Tuple[str, ...] = (
    "Arn",
    "ARN",
    "Id",
    "ID",
    "Name",
    "ResourceArn",
    "TaskDefinitionArn",
    "ClusterName",
    "ServiceName",
    "NodegroupName",
    "FargateProfileName",
    "FunctionName",
    "LogGroupName",
    "QueueUrl",
    "TopicArn",
    "DBInstanceIdentifier",
    "RepositoryName",
    "BucketName",
)

# Identity-like fields used for name matching in --match-mode tag-or-name.
# This intentionally avoids scanning every scalar property to reduce false positives
# (for example, matching an RDS instance by MonitoringRoleArn).
RESOURCE_NAME_MATCH_FIELDS: Dict[str, Tuple[str, ...]] = {
    "AWS::ECS::Service": ("ServiceName", "Name", "ServiceArn", "Cluster"),
    "AWS::ECS::TaskDefinition": ("TaskDefinitionArn", "Arn", "Family", "Revision"),
    "AWS::AutoScaling::AutoScalingGroup": ("AutoScalingGroupName",),
    "AWS::EC2::Instance": ("InstanceId", "PrivateDnsName"),
    "AWS::Lambda::Function": ("FunctionName", "FunctionArn", "Name"),
    "AWS::Logs::LogGroup": ("LogGroupName", "Arn"),
    "AWS::SQS::Queue": ("QueueUrl", "QueueArn", "QueueName"),
    "AWS::SNS::Topic": ("TopicArn", "DisplayName"),
    "AWS::ElasticLoadBalancingV2::LoadBalancer": ("LoadBalancerArn", "LoadBalancerName"),
    "AWS::ElasticLoadBalancingV2::TargetGroup": ("TargetGroupArn", "TargetGroupName"),
    "AWS::ECS::Cluster": ("ClusterName", "Arn"),
    "AWS::EKS::Cluster": ("Name", "Arn"),
    "AWS::EKS::Nodegroup": ("NodegroupName", "NodegroupArn", "ClusterName", "Arn"),
    "AWS::EKS::FargateProfile": ("FargateProfileName", "FargateProfileArn", "ClusterName", "Arn"),
    "AWS::ElastiCache::ReplicationGroup": ("ReplicationGroupId", "ARN"),
    "AWS::ElastiCache::CacheCluster": ("CacheClusterId", "ClusterName", "ARN"),
    "AWS::RDS::DBInstance": ("DBInstanceIdentifier", "DBName", "DBInstanceArn"),
    "AWS::ECR::Repository": ("RepositoryName", "RepositoryArn"),
    "AWS::OpenSearchService::Domain": ("DomainName", "ARN"),
    "AWS::Elasticsearch::Domain": ("DomainName", "ARN"),
    "AWS::S3::Bucket": ("BucketName",),
}


@dataclass
class InventoryRecord:
    account_id: str
    region: str
    resource_type: str
    resource_key: str
    properties: Dict[str, Any]


@dataclass
class DeletionTask:
    region: str
    resource_type: str
    resource_key: str
    handler_name: str
    priority: int
    payload: Dict[str, str]
    match_reason: str


@dataclass
class DeletionResult:
    status: str
    message: str


@dataclass
class DeleteContext:
    session: Any
    profile: str
    default_region: str
    dry_run: bool
    cloudcontrol_progress_seconds: int
    clients: Dict[str, Any]

    def client(self, service_name: str, region: str) -> Any:
        target_region = region or self.default_region
        cache_key = f"{service_name}@{target_region}"
        if cache_key not in self.clients:
            self.clients[cache_key] = self.session.client(service_name, region_name=target_region)
        return self.clients[cache_key]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
    write_out(f"[{now_str()}] {message}")


def warn(message: str) -> None:
    write_err(f"[{now_str()}] WARN: {message}")


def print_status_line(task: DeletionTask, result: DeletionResult) -> None:
    line = f"[{result.status.upper():8}] {task.resource_type} [{task.region}] {task.resource_key} | {result.message}"
    if STDOUT_CONSOLE is not None and RICH_UI:
        STDOUT_CONSOLE.print(line, style=STATUS_STYLES.get(result.status, "white"), markup=False, highlight=False)
        return
    print(line)


def print_pending_line(task: DeletionTask) -> None:
    line = f"  [pending] {task.resource_type} [{task.region}] {task.resource_key}"
    if STDOUT_CONSOLE is not None and RICH_UI:
        STDOUT_CONSOLE.print(line, style=STATUS_STYLES["pending"], markup=False, highlight=False)
        return
    print(line)


def print_type_breakdown(rows: Sequence[Tuple[str, int]]) -> None:
    if STDOUT_CONSOLE is not None and RICH_UI and Table is not None:
        table = Table(title="Matched Resource Types", header_style="bold cyan")
        table.add_column("Resource Type", style="white")
        table.add_column("Count", justify="right", style="bold")
        for resource_type, count in rows:
            table.add_row(resource_type, str(count))
        STDOUT_CONSOLE.print(table)
        return

    for resource_type, count in rows:
        print(f"  {resource_type}: {count}")


def print_stats_summary(stats: Counter[str]) -> None:
    if STDOUT_CONSOLE is not None and RICH_UI and Table is not None:
        table = Table(title="Deletion Summary", header_style="bold cyan")
        table.add_column("Status", style="white")
        table.add_column("Count", justify="right", style="bold")
        for key in sorted(stats):
            style = STATUS_STYLES.get(key, "white")
            table.add_row(key, str(stats[key]), style=style)
        STDOUT_CONSOLE.print(table)
        return

    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete AWS resources by stage/tag from an inventory JSON file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Inventory JSON path or filename inside --results-dir.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory used for fallback lookup when --input is a filename.",
    )
    parser.add_argument("--profile", default="catalytic-pc-dev", help="AWS profile to use.")
    parser.add_argument("--region", default="us-east-1", help="Default AWS region.")
    parser.add_argument(
        "--tag-value",
        required=True,
        help="Stage/tag value to match (example: staging).",
    )
    parser.add_argument(
        "--tag-keys",
        default=",".join(DEFAULT_TAG_KEYS),
        help="Comma-separated tag keys to match against.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["tag", "tag-or-name"],
        default="tag-or-name",
        help=(
            "Matching strategy: 'tag' only tagged resources, or 'tag-or-name' to also include "
            "resources whose identifiers contain the stage value."
        ),
    )
    parser.add_argument(
        "--include-types",
        default="",
        help="Optional comma-separated AWS::Type allow list.",
    )
    parser.add_argument(
        "--exclude-types",
        default="",
        help="Optional comma-separated AWS::Type block list.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply deletions. Dry-run if omitted.")
    parser.add_argument("--force", action="store_true", help="Skip interactive SURE prompt in apply mode.")
    parser.add_argument(
        "--allow-non-dev-profile",
        action="store_true",
        help="Allow profile names that do not end with -dev.",
    )
    parser.add_argument("--max-rounds", type=int, default=4, help="Max deletion rounds for deferred tasks.")
    parser.add_argument(
        "--round-wait-seconds",
        type=int,
        default=20,
        help="Wait between rounds when deferred tasks remain.",
    )
    parser.add_argument(
        "--disable-generic-fallback",
        action="store_true",
        help="Disable generic delete attempts for matched resource types without specific handlers.",
    )
    parser.add_argument(
        "--cloudcontrol-progress-seconds",
        type=int,
        default=15,
        help=(
            "Print progress while waiting for Cloud Control deletes every N seconds "
            "(0 disables progress logs)."
        ),
    )
    return parser.parse_args()


def parse_csv_set(raw: str) -> Set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def parse_tag_keys(raw: str) -> List[str]:
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    if not keys:
        raise ValueError("At least one tag key is required")
    return keys


def resolve_path(candidate: str, results_dir: Path, arg_name: str) -> Path:
    path = Path(candidate).expanduser()
    if path.is_file():
        return path.resolve()

    fallback = (results_dir / candidate).expanduser()
    if fallback.is_file():
        return fallback.resolve()

    raise FileNotFoundError(
        f"Could not find {arg_name}: '{candidate}'. Checked direct path and '{results_dir / candidate}'."
    )


def assert_profile_is_safe(profile: str, allow_non_dev_profile: bool) -> None:
    if allow_non_dev_profile:
        return
    if not profile.endswith("-dev"):
        raise ValueError(
            f"Refusing to run with profile '{profile}'. Use --allow-non-dev-profile to override."
        )


def get_error_code(exc: ClientError) -> str:
    payload = exc.response.get("Error", {}) if isinstance(exc.response, dict) else {}
    code = payload.get("Code")
    return str(code) if code else "UnknownError"


def is_retryable(exc: ClientError) -> bool:
    return get_error_code(exc) in RETRYABLE_ERROR_CODES


def load_inventory(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_inventory_records(doc: Dict[str, Any], fallback_region: str) -> Iterable[InventoryRecord]:
    meta = doc.get("_metadata", {})
    account_id = str(meta.get("account_id", "unknown"))

    regions = doc.get("regions", {})
    if not isinstance(regions, dict):
        return

    for region, region_payload in regions.items():
        region_name = str(region or fallback_region)
        if not isinstance(region_payload, dict):
            continue

        for resource_type, resources in region_payload.items():
            if not isinstance(resources, dict):
                continue

            for resource_key, resource_props in resources.items():
                props = resource_props if isinstance(resource_props, dict) else {}
                yield InventoryRecord(
                    account_id=account_id,
                    region=region_name,
                    resource_type=str(resource_type),
                    resource_key=str(resource_key),
                    properties=props,
                )


def get_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def normalize_tags(props: Dict[str, Any]) -> Dict[str, str]:
    tags: Dict[str, str] = {}

    def add_tag(raw_key: Any, raw_value: Any) -> None:
        key = get_string(raw_key).strip().lower()
        if key:
            tags[key] = get_string(raw_value).strip()

    def parse_tag_item(item: Any) -> None:
        if isinstance(item, list):
            for entry in item:
                parse_tag_item(entry)
            return

        if not isinstance(item, dict):
            return

        lowered = {str(k).lower(): k for k in item.keys()}
        if "key" in lowered and "value" in lowered:
            add_tag(item[lowered["key"]], item[lowered["value"]])
            return
        if "name" in lowered and "value" in lowered:
            add_tag(item[lowered["name"]], item[lowered["value"]])
            return

        for sub_key, sub_value in item.items():
            if isinstance(sub_value, (str, int, float, bool)):
                add_tag(sub_key, sub_value)

    for key in ("Tags", "TagList", "TagSet", "tags", "tagList", "tagSet"):
        value = props.get(key)
        if value is not None:
            parse_tag_item(value)

    return tags


def iter_scalar_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from iter_scalar_strings(child)
        return

    if isinstance(value, list):
        for child in value:
            yield from iter_scalar_strings(child)
        return

    if isinstance(value, (str, int, float, bool)):
        text = get_string(value).strip()
        if text:
            yield text


def name_match_candidates(record: InventoryRecord) -> List[str]:
    candidates: List[str] = [record.resource_key]

    fields = RESOURCE_NAME_MATCH_FIELDS.get(record.resource_type)
    if fields:
        for field in fields:
            raw_value = record.properties.get(field)
            if raw_value is None:
                continue
            if isinstance(raw_value, (str, int, float, bool)):
                text = get_string(raw_value).strip()
                if text:
                    candidates.append(text)
            elif isinstance(raw_value, list):
                for item in raw_value:
                    text = get_string(item).strip()
                    if text:
                        candidates.append(text)
    else:
        # Fallback for unknown types: check only the resource key.
        pass

    # Deduplicate while preserving order.
    seen: Set[str] = set()
    deduped: List[str] = []
    for value in candidates:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def generic_identifier_candidates(record: InventoryRecord) -> List[str]:
    candidates: List[str] = []

    def add(value: Any) -> None:
        text = get_string(value).strip()
        if not text:
            return
        if len(text) > 1024:
            return
        candidates.append(text)

    add(record.resource_key)
    if "|" in record.resource_key:
        add(record.resource_key.split("|", 1)[0])

    for field in RESOURCE_NAME_MATCH_FIELDS.get(record.resource_type, ()):
        raw = record.properties.get(field)
        if raw is None:
            continue
        if isinstance(raw, list):
            for item in raw:
                add(item)
        else:
            add(raw)

    for field in GENERIC_IDENTIFIER_KEYS:
        if field not in record.properties:
            continue
        raw = record.properties.get(field)
        if isinstance(raw, list):
            for item in raw:
                add(item)
        else:
            add(raw)

    for value in iter_scalar_strings(record.properties):
        if value.startswith("arn:"):
            add(value)

    # Deduplicate while preserving order.
    deduped: List[str] = []
    seen: Set[str] = set()
    for value in candidates:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def stage_regex(stage_value: str) -> re.Pattern[str]:
    escaped = re.escape(stage_value.lower())
    return re.compile(rf"(^|[^a-z0-9]){escaped}([^a-z0-9]|$)")


def match_reason(
    record: InventoryRecord,
    tag_value: str,
    tag_keys: Sequence[str],
    mode: str,
    stage_re: re.Pattern[str],
) -> str:
    normalized_value = tag_value.strip().lower()
    tags = normalize_tags(record.properties)

    for raw_key in tag_keys:
        key = raw_key.strip().lower()
        if not key:
            continue
        tag = tags.get(key, "").strip().lower()
        if tag == normalized_value:
            return f"tag:{raw_key}={tag_value}"

    if mode == "tag":
        return ""

    candidates = name_match_candidates(record)

    for candidate in candidates:
        lowered = candidate.lower()
        if stage_re.search(lowered):
            short = candidate
            if len(short) > 96:
                short = short[:93] + "..."
            return f"name:{short}"

    return ""


def summarize_by_type(records: Sequence[InventoryRecord]) -> List[Tuple[str, int]]:
    counter = Counter(record.resource_type for record in records)
    return sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))


def pick_first(props: Dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = get_string(props.get(key)).strip()
        if value:
            return value
    return default


def parse_cluster_service_from_arn(service_arn: str) -> Tuple[str, str]:
    # arn:aws:ecs:region:account:service/cluster/service
    marker = ":service/"
    if marker not in service_arn:
        return "", ""
    suffix = service_arn.split(marker, 1)[1]
    parts = suffix.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def parse_eks_nodegroup_from_arn(nodegroup_arn: str) -> Tuple[str, str]:
    # arn:aws:eks:region:account:nodegroup/<cluster>/<nodegroup>/<id>
    marker = ":nodegroup/"
    if marker not in nodegroup_arn:
        return "", ""
    suffix = nodegroup_arn.split(marker, 1)[1]
    parts = suffix.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def parse_eks_fargate_profile_from_arn(profile_arn: str) -> Tuple[str, str]:
    # arn:aws:eks:region:account:fargateprofile/<cluster>/<profile>/<id>
    marker = ":fargateprofile/"
    if marker not in profile_arn:
        return "", ""
    suffix = profile_arn.split(marker, 1)[1]
    parts = suffix.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "", ""


def task_from_record(
    record: InventoryRecord,
    reason: str,
    allow_generic_fallback: bool,
) -> Optional[DeletionTask]:
    resource_type = record.resource_type
    props = record.properties
    key = record.resource_key

    if resource_type == "AWS::ECS::Service":
        service_arn = pick_first(props, "ServiceArn", "Arn", default=key.split("|", 1)[0])
        cluster = pick_first(props, "Cluster")
        service_name = pick_first(props, "ServiceName", "Name")
        if not cluster or not service_name:
            parsed_cluster, parsed_service = parse_cluster_service_from_arn(service_arn)
            cluster = cluster or parsed_cluster
            service_name = service_name or parsed_service
        if not cluster or not service_name:
            return None
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_ecs_service",
            priority=10,
            payload={"cluster": cluster, "service": service_name},
            match_reason=reason,
        )

    if resource_type == "AWS::ECS::TaskDefinition":
        task_definition_arn = pick_first(props, "TaskDefinitionArn", "Arn", default=key.split("|", 1)[0])
        if not task_definition_arn:
            return None
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_ecs_task_definition",
            priority=72,
            payload={"task_definition": task_definition_arn},
            match_reason=reason,
        )

    if resource_type == "AWS::AutoScaling::AutoScalingGroup":
        name = pick_first(props, "AutoScalingGroupName", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_autoscaling_group",
            priority=20,
            payload={"auto_scaling_group": name},
            match_reason=reason,
        )

    if resource_type == "AWS::EC2::Instance":
        instance_id = pick_first(props, "InstanceId", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_ec2_instance",
            priority=25,
            payload={"instance_id": instance_id},
            match_reason=reason,
        )

    if resource_type == "AWS::Lambda::Function":
        function_name = pick_first(props, "FunctionName", "Function", "Name", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_lambda_function",
            priority=30,
            payload={"function_name": function_name},
            match_reason=reason,
        )

    if resource_type == "AWS::Logs::LogGroup":
        log_group = pick_first(props, "LogGroupName", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_log_group",
            priority=35,
            payload={"log_group_name": log_group},
            match_reason=reason,
        )

    if resource_type == "AWS::SQS::Queue":
        queue_url = pick_first(props, "QueueUrl", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_sqs_queue",
            priority=40,
            payload={"queue_url": queue_url},
            match_reason=reason,
        )

    if resource_type == "AWS::SNS::Topic":
        topic_arn = pick_first(props, "TopicArn", "Arn", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_sns_topic",
            priority=40,
            payload={"topic_arn": topic_arn},
            match_reason=reason,
        )

    if resource_type == "AWS::ElasticLoadBalancingV2::LoadBalancer":
        lb_arn = pick_first(props, "LoadBalancerArn", "Arn", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_elbv2_load_balancer",
            priority=50,
            payload={"load_balancer_arn": lb_arn},
            match_reason=reason,
        )

    if resource_type == "AWS::ElasticLoadBalancingV2::TargetGroup":
        tg_arn = pick_first(props, "TargetGroupArn", "Arn", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_elbv2_target_group",
            priority=60,
            payload={"target_group_arn": tg_arn},
            match_reason=reason,
        )

    if resource_type == "AWS::ECS::Cluster":
        cluster_name = pick_first(props, "ClusterName", "Name", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_ecs_cluster",
            priority=70,
            payload={"cluster_name": cluster_name},
            match_reason=reason,
        )

    if resource_type == "AWS::EKS::Cluster":
        cluster_name = pick_first(props, "Name", "ClusterName", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_eks_cluster",
            priority=75,
            payload={"cluster_name": cluster_name},
            match_reason=reason,
        )

    if resource_type == "AWS::EKS::Nodegroup":
        cluster_name = pick_first(props, "ClusterName", "clusterName")
        nodegroup_name = pick_first(props, "NodegroupName", "nodegroupName", "Name")
        nodegroup_arn = pick_first(props, "NodegroupArn", "Arn", default=key.split("|", 1)[0])
        if not cluster_name or not nodegroup_name:
            parsed_cluster, parsed_nodegroup = parse_eks_nodegroup_from_arn(nodegroup_arn)
            cluster_name = cluster_name or parsed_cluster
            nodegroup_name = nodegroup_name or parsed_nodegroup
        if (not cluster_name or not nodegroup_name) and "|" in key:
            left, right = key.split("|", 1)
            cluster_name = cluster_name or left
            nodegroup_name = nodegroup_name or right
        if not cluster_name or not nodegroup_name:
            return None
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_eks_nodegroup",
            priority=73,
            payload={"cluster_name": cluster_name, "nodegroup_name": nodegroup_name},
            match_reason=reason,
        )

    if resource_type == "AWS::EKS::FargateProfile":
        cluster_name = pick_first(props, "ClusterName", "clusterName")
        profile_name = pick_first(props, "FargateProfileName", "fargateProfileName", "Name")
        profile_arn = pick_first(props, "FargateProfileArn", "Arn", default=key.split("|", 1)[0])
        if not cluster_name or not profile_name:
            parsed_cluster, parsed_profile = parse_eks_fargate_profile_from_arn(profile_arn)
            cluster_name = cluster_name or parsed_cluster
            profile_name = profile_name or parsed_profile
        if (not cluster_name or not profile_name) and "|" in key:
            left, right = key.split("|", 1)
            cluster_name = cluster_name or left
            profile_name = profile_name or right
        if not cluster_name or not profile_name:
            return None
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_eks_fargate_profile",
            priority=74,
            payload={"cluster_name": cluster_name, "fargate_profile_name": profile_name},
            match_reason=reason,
        )

    if resource_type == "AWS::ElastiCache::ReplicationGroup":
        repl_id = pick_first(props, "ReplicationGroupId", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_elasticache_replication_group",
            priority=78,
            payload={"replication_group_id": repl_id},
            match_reason=reason,
        )

    if resource_type == "AWS::ElastiCache::CacheCluster":
        cluster_id = pick_first(props, "CacheClusterId", "ClusterName", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_elasticache_cluster",
            priority=80,
            payload={"cache_cluster_id": cluster_id},
            match_reason=reason,
        )

    if resource_type == "AWS::RDS::DBInstance":
        db_id = pick_first(props, "DBInstanceIdentifier", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_rds_instance",
            priority=85,
            payload={"db_instance_id": db_id},
            match_reason=reason,
        )

    if resource_type == "AWS::ECR::Repository":
        repo_name = pick_first(props, "RepositoryName", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_ecr_repository",
            priority=90,
            payload={"repository_name": repo_name},
            match_reason=reason,
        )

    if resource_type == "AWS::OpenSearchService::Domain":
        domain = pick_first(props, "DomainName", "Name", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_opensearch_domain",
            priority=92,
            payload={"domain_name": domain},
            match_reason=reason,
        )

    if resource_type == "AWS::Elasticsearch::Domain":
        domain = pick_first(props, "DomainName", "Name", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_elasticsearch_domain",
            priority=93,
            payload={"domain_name": domain},
            match_reason=reason,
        )

    if resource_type == "AWS::S3::Bucket":
        bucket = pick_first(props, "BucketName", default=key)
        return DeletionTask(
            region=record.region,
            resource_type=resource_type,
            resource_key=key,
            handler_name="delete_s3_bucket",
            priority=100,
            payload={"bucket_name": bucket},
            match_reason=reason,
        )

    if allow_generic_fallback:
        identifiers = generic_identifier_candidates(record)
        if identifiers:
            return DeletionTask(
                region=record.region,
                resource_type=resource_type,
                resource_key=key,
                handler_name="delete_generic_resource",
                priority=900,
                payload={
                    "type_name": resource_type,
                    "identifiers_json": json.dumps(identifiers),
                },
                match_reason=reason,
            )

    return None


def make_result(status: str, message: str) -> DeletionResult:
    return DeletionResult(status=status, message=message)


def delete_autoscaling_group(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    name = task.payload["auto_scaling_group"]
    if ctx.dry_run:
        return make_result("planned", f"Delete AutoScaling Group {name} (force)")

    asg = ctx.client("autoscaling", task.region)
    try:
        asg.delete_auto_scaling_group(AutoScalingGroupName=name, ForceDelete=True)
        return make_result("deleted", f"Deletion requested for AutoScaling Group {name}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "ValidationError":
            return make_result("skipped", f"AutoScaling Group {name} not found")
        if is_retryable(exc):
            return make_result("deferred", f"AutoScaling Group {name}: {code}")
        return make_result("failed", f"AutoScaling Group {name}: {code}")


def delete_ec2_instance(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    instance_id = task.payload["instance_id"]
    if ctx.dry_run:
        return make_result("planned", f"Terminate EC2 instance {instance_id}")

    ec2 = ctx.client("ec2", task.region)
    try:
        ec2.terminate_instances(InstanceIds=[instance_id])
        return make_result("deleted", f"Termination requested for EC2 {instance_id}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "InvalidInstanceID.NotFound":
            return make_result("skipped", f"EC2 {instance_id} already deleted")
        if is_retryable(exc):
            return make_result("deferred", f"EC2 {instance_id}: {code}")
        return make_result("failed", f"EC2 {instance_id}: {code}")


def delete_lambda_function(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    function_name = task.payload["function_name"]
    if ctx.dry_run:
        return make_result("planned", f"Delete Lambda function {function_name}")

    lamb = ctx.client("lambda", task.region)
    try:
        lamb.delete_function(FunctionName=function_name)
        return make_result("deleted", f"Deleted Lambda {function_name}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "ResourceNotFoundException":
            return make_result("skipped", f"Lambda {function_name} already deleted")
        if is_retryable(exc):
            return make_result("deferred", f"Lambda {function_name}: {code}")
        return make_result("failed", f"Lambda {function_name}: {code}")


def delete_log_group(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    log_group_name = task.payload["log_group_name"]
    if ctx.dry_run:
        return make_result("planned", f"Delete CloudWatch log group {log_group_name}")

    logs = ctx.client("logs", task.region)
    try:
        logs.delete_log_group(logGroupName=log_group_name)
        return make_result("deleted", f"Deleted log group {log_group_name}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "ResourceNotFoundException":
            return make_result("skipped", f"Log group {log_group_name} already deleted")
        if is_retryable(exc):
            return make_result("deferred", f"Log group {log_group_name}: {code}")
        return make_result("failed", f"Log group {log_group_name}: {code}")


def delete_sqs_queue(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    queue_url = task.payload["queue_url"]
    if ctx.dry_run:
        return make_result("planned", f"Delete SQS queue {queue_url}")

    sqs = ctx.client("sqs", task.region)
    try:
        sqs.delete_queue(QueueUrl=queue_url)
        return make_result("deleted", f"Deleted SQS queue {queue_url}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code in {"AWS.SimpleQueueService.NonExistentQueue", "QueueDoesNotExist"}:
            return make_result("skipped", f"SQS queue already deleted: {queue_url}")
        if is_retryable(exc):
            return make_result("deferred", f"SQS {queue_url}: {code}")
        return make_result("failed", f"SQS {queue_url}: {code}")


def delete_sns_topic(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    topic_arn = task.payload["topic_arn"]
    if ctx.dry_run:
        return make_result("planned", f"Delete SNS topic {topic_arn}")

    sns = ctx.client("sns", task.region)
    try:
        sns.delete_topic(TopicArn=topic_arn)
        return make_result("deleted", f"Deleted SNS topic {topic_arn}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "NotFound":
            return make_result("skipped", f"SNS topic already deleted: {topic_arn}")
        if is_retryable(exc):
            return make_result("deferred", f"SNS {topic_arn}: {code}")
        return make_result("failed", f"SNS {topic_arn}: {code}")


def iter_elbv2_listeners(elbv2: Any, lb_arn: str) -> Iterable[str]:
    marker: Optional[str] = None
    while True:
        params: Dict[str, Any] = {"LoadBalancerArn": lb_arn}
        if marker:
            params["Marker"] = marker
        response = elbv2.describe_listeners(**params)
        for listener in response.get("Listeners", []):
            listener_arn = get_string(listener.get("ListenerArn")).strip()
            if listener_arn:
                yield listener_arn
        marker = response.get("NextMarker")
        if not marker:
            return


def delete_elbv2_load_balancer(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    lb_arn = task.payload["load_balancer_arn"]
    if ctx.dry_run:
        return make_result("planned", f"Delete ELBv2 load balancer {lb_arn} (listeners first)")

    elbv2 = ctx.client("elbv2", task.region)
    try:
        for listener_arn in iter_elbv2_listeners(elbv2, lb_arn):
            try:
                elbv2.delete_listener(ListenerArn=listener_arn)
                log(f"Deleted ELBv2 listener {listener_arn}")
            except ClientError as exc:
                code = get_error_code(exc)
                if code not in {"ListenerNotFound"}:
                    warn(f"Failed to delete ELBv2 listener {listener_arn}: {code}")

        elbv2.delete_load_balancer(LoadBalancerArn=lb_arn)
        return make_result("deleted", f"Deleted ELBv2 load balancer {lb_arn}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "LoadBalancerNotFound":
            return make_result("skipped", f"ELBv2 load balancer not found: {lb_arn}")
        if is_retryable(exc):
            return make_result("deferred", f"ELBv2 load balancer {lb_arn}: {code}")
        return make_result("failed", f"ELBv2 load balancer {lb_arn}: {code}")


def delete_elbv2_target_group(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    tg_arn = task.payload["target_group_arn"]
    if ctx.dry_run:
        return make_result("planned", f"Delete ELBv2 target group {tg_arn}")

    elbv2 = ctx.client("elbv2", task.region)
    try:
        elbv2.delete_target_group(TargetGroupArn=tg_arn)
        return make_result("deleted", f"Deleted ELBv2 target group {tg_arn}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "TargetGroupNotFound":
            return make_result("skipped", f"Target group not found: {tg_arn}")
        if is_retryable(exc):
            return make_result("deferred", f"Target group {tg_arn}: {code}")
        return make_result("failed", f"Target group {tg_arn}: {code}")


def delete_ecs_service(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    cluster = task.payload["cluster"]
    service = task.payload["service"]
    if ctx.dry_run:
        return make_result("planned", f"Delete ECS service {cluster}/{service}")

    ecs = ctx.client("ecs", task.region)
    try:
        try:
            ecs.update_service(cluster=cluster, service=service, desiredCount=0)
        except ClientError:
            pass
        ecs.delete_service(cluster=cluster, service=service, force=True)
        return make_result("deleted", f"Deletion requested for ECS service {cluster}/{service}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code in {"ServiceNotFoundException", "ClusterNotFoundException"}:
            return make_result("skipped", f"ECS service already deleted: {cluster}/{service}")
        if is_retryable(exc):
            return make_result("deferred", f"ECS service {cluster}/{service}: {code}")
        return make_result("failed", f"ECS service {cluster}/{service}: {code}")


def delete_ecs_task_definition(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    task_definition = task.payload["task_definition"]
    if ctx.dry_run:
        return make_result("planned", f"Deregister ECS task definition {task_definition}")

    ecs = ctx.client("ecs", task.region)
    try:
        ecs.deregister_task_definition(taskDefinition=task_definition)
        return make_result("deleted", f"Deregistered ECS task definition {task_definition}")
    except ClientError as exc:
        code = get_error_code(exc)
        message = get_string(exc.response.get("Error", {}).get("Message", "")).lower()
        if code in {"ClientException", "InvalidParameterException"} and (
            "unable to describe" in message or "not found" in message or "inactive" in message
        ):
            return make_result("skipped", f"ECS task definition already inactive/missing: {task_definition}")
        if is_retryable(exc):
            return make_result("deferred", f"ECS task definition {task_definition}: {code}")
        return make_result("failed", f"ECS task definition {task_definition}: {code}")


def list_ecs_pages(ecs: Any, method: str, key: str, cluster: str) -> List[str]:
    items: List[str] = []
    token: Optional[str] = None
    while True:
        params: Dict[str, Any] = {"cluster": cluster, "maxResults": 100}
        if token:
            params["nextToken"] = token
        response = getattr(ecs, method)(**params)
        items.extend(response.get(key, []))
        token = response.get("nextToken")
        if not token:
            break
    return items


def delete_ecs_cluster(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    cluster = task.payload["cluster_name"]
    if ctx.dry_run:
        return make_result("planned", f"Delete ECS cluster {cluster} (services/tasks/instances first)")

    ecs = ctx.client("ecs", task.region)
    try:
        service_arns = list_ecs_pages(ecs, "list_services", "serviceArns", cluster)
        for service_arn in service_arns:
            service_name = service_arn.rsplit("/", 1)[-1]
            try:
                try:
                    ecs.update_service(cluster=cluster, service=service_name, desiredCount=0)
                except ClientError:
                    pass
                ecs.delete_service(cluster=cluster, service=service_name, force=True)
                log(f"Deletion requested for ECS service {cluster}/{service_name}")
            except ClientError as exc:
                code = get_error_code(exc)
                if code not in {"ServiceNotFoundException", "ClusterNotFoundException"}:
                    warn(f"Could not delete ECS service {cluster}/{service_name}: {code}")

        task_arns = list_ecs_pages(ecs, "list_tasks", "taskArns", cluster)
        for task_arn in task_arns:
            try:
                ecs.stop_task(cluster=cluster, task=task_arn, reason="Stage cleanup")
            except ClientError as exc:
                code = get_error_code(exc)
                if code not in {"TaskNotFoundException", "ClusterNotFoundException"}:
                    warn(f"Could not stop ECS task {task_arn}: {code}")

        container_instances = list_ecs_pages(ecs, "list_container_instances", "containerInstanceArns", cluster)
        for ci_arn in container_instances:
            try:
                ecs.deregister_container_instance(cluster=cluster, containerInstance=ci_arn, force=True)
            except ClientError as exc:
                code = get_error_code(exc)
                if code not in {"ClusterNotFoundException"}:
                    warn(f"Could not deregister ECS container instance {ci_arn}: {code}")

        ecs.delete_cluster(cluster=cluster)
        return make_result("deleted", f"Deletion requested for ECS cluster {cluster}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "ClusterNotFoundException":
            return make_result("skipped", f"ECS cluster already deleted: {cluster}")
        if is_retryable(exc):
            return make_result("deferred", f"ECS cluster {cluster}: {code}")
        return make_result("failed", f"ECS cluster {cluster}: {code}")


def delete_eks_cluster(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    cluster = task.payload["cluster_name"]
    if ctx.dry_run:
        return make_result("planned", f"Delete EKS cluster {cluster} (nodegroups/fargate first)")

    eks = ctx.client("eks", task.region)
    deleted_children = 0
    try:
        nodegroups = eks.list_nodegroups(clusterName=cluster).get("nodegroups", [])
        for nodegroup in nodegroups:
            try:
                eks.delete_nodegroup(clusterName=cluster, nodegroupName=nodegroup)
                deleted_children += 1
            except ClientError as exc:
                code = get_error_code(exc)
                if code not in {"ResourceNotFoundException"}:
                    warn(f"Could not delete EKS nodegroup {cluster}/{nodegroup}: {code}")

        fargate_profiles = eks.list_fargate_profiles(clusterName=cluster).get("fargateProfileNames", [])
        for profile in fargate_profiles:
            try:
                eks.delete_fargate_profile(clusterName=cluster, fargateProfileName=profile)
                deleted_children += 1
            except ClientError as exc:
                code = get_error_code(exc)
                if code not in {"ResourceNotFoundException"}:
                    warn(f"Could not delete EKS fargate profile {cluster}/{profile}: {code}")

        eks.delete_cluster(name=cluster)
        return make_result("deleted", f"Deletion requested for EKS cluster {cluster}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "ResourceNotFoundException":
            return make_result("skipped", f"EKS cluster already deleted: {cluster}")
        if is_retryable(exc):
            note = f"EKS cluster {cluster}: {code}"
            if deleted_children:
                note += f" (child deletes requested: {deleted_children})"
            return make_result("deferred", note)
        return make_result("failed", f"EKS cluster {cluster}: {code}")


def delete_eks_nodegroup(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    cluster_name = task.payload["cluster_name"]
    nodegroup_name = task.payload["nodegroup_name"]
    if ctx.dry_run:
        return make_result("planned", f"Delete EKS nodegroup {cluster_name}/{nodegroup_name}")

    eks = ctx.client("eks", task.region)
    try:
        eks.delete_nodegroup(clusterName=cluster_name, nodegroupName=nodegroup_name)
        return make_result("deleted", f"Deletion requested for EKS nodegroup {cluster_name}/{nodegroup_name}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "ResourceNotFoundException":
            return make_result("skipped", f"EKS nodegroup already deleted: {cluster_name}/{nodegroup_name}")
        if is_retryable(exc):
            return make_result("deferred", f"EKS nodegroup {cluster_name}/{nodegroup_name}: {code}")
        return make_result("failed", f"EKS nodegroup {cluster_name}/{nodegroup_name}: {code}")


def delete_eks_fargate_profile(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    cluster_name = task.payload["cluster_name"]
    profile_name = task.payload["fargate_profile_name"]
    if ctx.dry_run:
        return make_result("planned", f"Delete EKS fargate profile {cluster_name}/{profile_name}")

    eks = ctx.client("eks", task.region)
    try:
        eks.delete_fargate_profile(clusterName=cluster_name, fargateProfileName=profile_name)
        return make_result("deleted", f"Deletion requested for EKS fargate profile {cluster_name}/{profile_name}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "ResourceNotFoundException":
            return make_result("skipped", f"EKS fargate profile already deleted: {cluster_name}/{profile_name}")
        if is_retryable(exc):
            return make_result("deferred", f"EKS fargate profile {cluster_name}/{profile_name}: {code}")
        return make_result("failed", f"EKS fargate profile {cluster_name}/{profile_name}: {code}")


def delete_elasticache_replication_group(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    repl_id = task.payload["replication_group_id"]
    if ctx.dry_run:
        return make_result("planned", f"Delete ElastiCache replication group {repl_id}")

    ec = ctx.client("elasticache", task.region)
    try:
        ec.delete_replication_group(
            ReplicationGroupId=repl_id,
            RetainPrimaryCluster=False,
        )
        return make_result("deleted", f"Deletion requested for replication group {repl_id}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code in {"ReplicationGroupNotFoundFault"}:
            return make_result("skipped", f"Replication group already deleted: {repl_id}")
        if is_retryable(exc):
            return make_result("deferred", f"Replication group {repl_id}: {code}")
        return make_result("failed", f"Replication group {repl_id}: {code}")


def delete_elasticache_cluster(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    cluster_id = task.payload["cache_cluster_id"]
    if ctx.dry_run:
        return make_result("planned", f"Delete ElastiCache cache cluster {cluster_id}")

    ec = ctx.client("elasticache", task.region)
    try:
        ec.delete_cache_cluster(CacheClusterId=cluster_id)
        return make_result("deleted", f"Deletion requested for cache cluster {cluster_id}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code in {"CacheClusterNotFoundFault"}:
            return make_result("skipped", f"Cache cluster already deleted: {cluster_id}")
        if is_retryable(exc):
            return make_result("deferred", f"Cache cluster {cluster_id}: {code}")
        return make_result("failed", f"Cache cluster {cluster_id}: {code}")


def delete_rds_instance(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    db_id = task.payload["db_instance_id"]
    if ctx.dry_run:
        return make_result("planned", f"Delete RDS instance {db_id} (skip final snapshot)")

    rds = ctx.client("rds", task.region)
    try:
        rds.delete_db_instance(
            DBInstanceIdentifier=db_id,
            SkipFinalSnapshot=True,
            DeleteAutomatedBackups=True,
        )
        return make_result("deleted", f"Deletion requested for RDS {db_id}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "DBInstanceNotFound":
            return make_result("skipped", f"RDS instance already deleted: {db_id}")
        if is_retryable(exc):
            return make_result("deferred", f"RDS {db_id}: {code}")
        return make_result("failed", f"RDS {db_id}: {code}")


def delete_ecr_repository(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    repo = task.payload["repository_name"]
    if ctx.dry_run:
        return make_result("planned", f"Delete ECR repository {repo} (force)")

    ecr = ctx.client("ecr", task.region)
    try:
        ecr.delete_repository(repositoryName=repo, force=True)
        return make_result("deleted", f"Deleted ECR repository {repo}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code == "RepositoryNotFoundException":
            return make_result("skipped", f"ECR repository already deleted: {repo}")
        if is_retryable(exc):
            return make_result("deferred", f"ECR repository {repo}: {code}")
        return make_result("failed", f"ECR repository {repo}: {code}")


def delete_opensearch_domain(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    domain = task.payload["domain_name"]
    if ctx.dry_run:
        return make_result("planned", f"Delete OpenSearch domain {domain}")

    opensearch = ctx.client("opensearch", task.region)
    try:
        opensearch.delete_domain(DomainName=domain)
        return make_result("deleted", f"Deletion requested for OpenSearch domain {domain}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code in {"ResourceNotFoundException", "ValidationException"}:
            return make_result("skipped", f"OpenSearch domain not found: {domain}")
        if is_retryable(exc):
            return make_result("deferred", f"OpenSearch domain {domain}: {code}")
        return make_result("failed", f"OpenSearch domain {domain}: {code}")


def delete_elasticsearch_domain(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    domain = task.payload["domain_name"]
    if ctx.dry_run:
        return make_result("planned", f"Delete Elasticsearch domain {domain}")

    es = ctx.client("es", task.region)
    try:
        es.delete_elasticsearch_domain(DomainName=domain)
        return make_result("deleted", f"Deletion requested for Elasticsearch domain {domain}")
    except ClientError as exc:
        code = get_error_code(exc)
        if code in {"ResourceNotFoundException", "ValidationException"}:
            return make_result("skipped", f"Elasticsearch domain not found: {domain}")
        if is_retryable(exc):
            return make_result("deferred", f"Elasticsearch domain {domain}: {code}")
        return make_result("failed", f"Elasticsearch domain {domain}: {code}")


def chunked(items: Sequence[Dict[str, str]], size: int) -> Iterable[Sequence[Dict[str, str]]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def empty_bucket_objects(s3: Any, bucket: str) -> Tuple[int, int]:
    removed = 0
    aborted_uploads = 0

    versions_paginator = s3.get_paginator("list_object_versions")
    for page in versions_paginator.paginate(Bucket=bucket):
        objects: List[Dict[str, str]] = []

        for item in page.get("Versions", []):
            key = item.get("Key")
            version_id = item.get("VersionId")
            if key is not None and version_id is not None:
                objects.append({"Key": str(key), "VersionId": str(version_id)})

        for item in page.get("DeleteMarkers", []):
            key = item.get("Key")
            version_id = item.get("VersionId")
            if key is not None and version_id is not None:
                objects.append({"Key": str(key), "VersionId": str(version_id)})

        for payload in chunked(objects, 1000):
            if payload:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": list(payload), "Quiet": True})
                removed += len(payload)

    uploads_paginator = s3.get_paginator("list_multipart_uploads")
    for page in uploads_paginator.paginate(Bucket=bucket):
        for upload in page.get("Uploads", []):
            key = upload.get("Key")
            upload_id = upload.get("UploadId")
            if key is None or upload_id is None:
                continue
            s3.abort_multipart_upload(Bucket=bucket, Key=str(key), UploadId=str(upload_id))
            aborted_uploads += 1

    return removed, aborted_uploads


def delete_s3_bucket(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    bucket = task.payload["bucket_name"]
    if ctx.dry_run:
        return make_result(
            "planned",
            f"Delete S3 bucket {bucket} (versions + delete markers + multipart uploads + bucket)",
        )

    s3 = ctx.client("s3", task.region)
    try:
        try:
            location = s3.get_bucket_location(Bucket=bucket)
            bucket_region = get_string(location.get("LocationConstraint")).strip() or "us-east-1"
            if bucket_region != task.region:
                s3 = ctx.client("s3", bucket_region)
        except ClientError:
            pass

        try:
            s3.delete_bucket_policy(Bucket=bucket)
        except ClientError:
            pass

        removed, aborted_uploads = empty_bucket_objects(s3, bucket)
        s3.delete_bucket(Bucket=bucket)
        return make_result(
            "deleted",
            (
                f"Deleted S3 bucket {bucket} "
                f"(objects/versions removed: {removed}, multipart uploads aborted: {aborted_uploads})"
            ),
        )
    except ClientError as exc:
        code = get_error_code(exc)
        if code in {"NoSuchBucket", "404"}:
            return make_result("skipped", f"S3 bucket already deleted: {bucket}")
        if is_retryable(exc):
            return make_result("deferred", f"S3 bucket {bucket}: {code}")
        return make_result("failed", f"S3 bucket {bucket}: {code}")


def wait_cloudcontrol_delete(
    cloudcontrol: Any,
    request_token: str,
    timeout_seconds: int = 180,
    poll_seconds: int = 5,
    progress_interval_seconds: int = 0,
    progress_label: str = "",
) -> Tuple[str, str, str]:
    max_attempts = max(1, timeout_seconds // max(1, poll_seconds))
    start_time = time.monotonic()
    next_progress = max(1, progress_interval_seconds) if progress_interval_seconds > 0 else 0
    for _ in range(max_attempts):
        response = cloudcontrol.get_resource_request_status(RequestToken=request_token)
        event = response.get("ProgressEvent", {})
        status = get_string(event.get("OperationStatus")).strip()
        message = get_string(event.get("StatusMessage")).strip()
        error_code = get_string(event.get("ErrorCode")).strip()

        if status in {"SUCCESS", "FAILED", "CANCEL_COMPLETE"}:
            return status, error_code, message

        if next_progress:
            elapsed_seconds = int(time.monotonic() - start_time)
            if elapsed_seconds >= next_progress:
                label = f" ({progress_label})" if progress_label else ""
                short_token = request_token[:12] if request_token else "unknown"
                log(
                    "Cloud Control delete in progress"
                    f"{label}: status={status or 'IN_PROGRESS'}, elapsed={elapsed_seconds}s, token={short_token}..."
                )
                next_progress += max(1, progress_interval_seconds)

        time.sleep(max(1, poll_seconds))

    return "TIMEOUT", "Timeout", "Timed out waiting for Cloud Control delete status"


def delete_generic_resource(ctx: DeleteContext, task: DeletionTask) -> DeletionResult:
    type_name = task.payload["type_name"]
    identifiers_raw = task.payload.get("identifiers_json", "[]")
    try:
        identifiers = json.loads(identifiers_raw)
    except json.JSONDecodeError:
        identifiers = []
    if not isinstance(identifiers, list):
        identifiers = []
    identifiers = [get_string(item).strip() for item in identifiers if get_string(item).strip()]

    if not identifiers:
        return make_result("failed", f"No candidate identifiers found for {type_name}")

    if ctx.dry_run:
        preview = identifiers[0]
        return make_result("planned", f"Generic delete {type_name} using identifier {preview}")

    cloudcontrol = ctx.client("cloudcontrol", task.region)
    had_not_found = False
    last_error = ""

    for identifier in identifiers:
        try:
            response = cloudcontrol.delete_resource(TypeName=type_name, Identifier=identifier)
            event = response.get("ProgressEvent", {})
            request_token = get_string(event.get("RequestToken", "")).strip()
            if not request_token:
                return make_result("failed", f"Cloud Control returned no request token for {type_name}:{identifier}")

            progress_label = f"{type_name}:{identifier}"
            log(f"Cloud Control delete started for {progress_label} (request token: {request_token})")

            status, error_code, message = wait_cloudcontrol_delete(
                cloudcontrol,
                request_token,
                progress_interval_seconds=max(0, ctx.cloudcontrol_progress_seconds),
                progress_label=progress_label,
            )
            message_lower = message.lower()

            if status == "SUCCESS":
                log(f"Cloud Control delete finished for {progress_label} with status=SUCCESS")
                return make_result("deleted", f"Deleted {type_name} via Cloud Control ({identifier})")

            if status in {"FAILED", "CANCEL_COMPLETE"}:
                log(
                    f"Cloud Control delete finished for {progress_label} "
                    f"with status={status}, error_code={error_code or 'none'}"
                )
                if error_code in {"NotFound", "ResourceNotFound", "NotFoundException", "ResourceNotFoundException"}:
                    had_not_found = True
                    continue
                if "not found" in message_lower:
                    had_not_found = True
                    continue
                if error_code in {"AlreadyExists", "AlreadyDeleted"}:
                    return make_result("skipped", f"Resource already deleted for {type_name}:{identifier}")
                last_error = f"{error_code or status}: {message or 'Cloud Control delete failed'}"
                continue

            if status == "TIMEOUT":
                log(f"Cloud Control delete timed out for {progress_label}")
                return make_result("deferred", f"Timed out deleting {type_name}:{identifier}")

            last_error = f"Unexpected status {status} for {type_name}:{identifier}"
            continue

        except ClientError as exc:
            code = get_error_code(exc)
            message = get_string(exc.response.get("Error", {}).get("Message", ""))
            if code in {"ResourceNotFoundException", "NotFoundException", "NotFound"}:
                had_not_found = True
                continue
            if is_retryable(exc):
                return make_result("deferred", f"Generic delete {type_name}:{identifier}: {code}")
            if code in {"ValidationException", "InvalidRequestException", "GeneralServiceException"}:
                last_error = f"{code}: {message}"
                continue
            return make_result("failed", f"Generic delete {type_name}:{identifier}: {code}")

    if had_not_found and not last_error:
        return make_result("skipped", f"{type_name} already deleted or not found")
    if had_not_found and last_error:
        return make_result("failed", f"{type_name}: {last_error}")
    if last_error:
        return make_result("failed", f"{type_name}: {last_error}")
    return make_result("failed", f"Generic delete failed for {type_name}")


DELETE_HANDLERS: Dict[str, Callable[[DeleteContext, DeletionTask], DeletionResult]] = {
    "delete_autoscaling_group": delete_autoscaling_group,
    "delete_ec2_instance": delete_ec2_instance,
    "delete_lambda_function": delete_lambda_function,
    "delete_log_group": delete_log_group,
    "delete_sqs_queue": delete_sqs_queue,
    "delete_sns_topic": delete_sns_topic,
    "delete_elbv2_load_balancer": delete_elbv2_load_balancer,
    "delete_elbv2_target_group": delete_elbv2_target_group,
    "delete_ecs_service": delete_ecs_service,
    "delete_ecs_task_definition": delete_ecs_task_definition,
    "delete_ecs_cluster": delete_ecs_cluster,
    "delete_eks_nodegroup": delete_eks_nodegroup,
    "delete_eks_fargate_profile": delete_eks_fargate_profile,
    "delete_eks_cluster": delete_eks_cluster,
    "delete_elasticache_replication_group": delete_elasticache_replication_group,
    "delete_elasticache_cluster": delete_elasticache_cluster,
    "delete_rds_instance": delete_rds_instance,
    "delete_ecr_repository": delete_ecr_repository,
    "delete_opensearch_domain": delete_opensearch_domain,
    "delete_elasticsearch_domain": delete_elasticsearch_domain,
    "delete_s3_bucket": delete_s3_bucket,
    "delete_generic_resource": delete_generic_resource,
}


def require_confirmation(total: int, profile: str, default_region: str, tag_value: str, source: Path) -> None:
    print("")
    print("You are about to DELETE resources with this scope:")
    print(f"  profile    : {profile}")
    print(f"  region     : {default_region}")
    print(f"  stage/tag  : {tag_value}")
    print(f"  inventory  : {source}")
    print(f"  resources  : {total}")
    typed = input("Type SURE in uppercase to continue: ").strip()
    if typed != "SURE":
        raise SystemExit("Aborted by user.")


def print_plan(tasks: Sequence[DeletionTask], unsupported: Sequence[InventoryRecord]) -> None:
    log(f"Supported resources planned: {len(tasks)}")
    if STDOUT_CONSOLE is not None and RICH_UI and Table is not None:
        table = Table(title="Deletion Plan", header_style="bold cyan")
        table.add_column("Priority", justify="right", style="bold")
        table.add_column("Type", style="white")
        table.add_column("Region", style="white")
        table.add_column("Target", style="white")
        table.add_column("Reason", style="white")
        for task in tasks:
            detail = ", ".join(f"{k}={v}" for k, v in task.payload.items())
            table.add_row(f"{task.priority:03d}", task.resource_type, task.region, detail, task.match_reason)
        STDOUT_CONSOLE.print(table)
    else:
        for task in tasks:
            detail = ", ".join(f"{k}={v}" for k, v in task.payload.items())
            print(
                f"  [priority={task.priority:03d}] {task.resource_type} "
                f"[{task.region}] {detail} | reason={task.match_reason}"
            )

    if unsupported:
        log(f"Matched but unsupported resource types: {len(unsupported)}")
        if STDOUT_CONSOLE is not None and RICH_UI and Table is not None:
            table = Table(title="Unsupported Matched Resources", header_style="bold yellow")
            table.add_column("Type", style="white")
            table.add_column("Region", style="white")
            table.add_column("Resource Key", style="white")
            for item in unsupported:
                table.add_row(item.resource_type, item.region, item.resource_key)
            STDOUT_CONSOLE.print(table)
        else:
            for item in unsupported:
                print(f"  [skip] {item.resource_type} [{item.region}] {item.resource_key}")


def run_delete_rounds(
    ctx: DeleteContext,
    tasks: Sequence[DeletionTask],
    max_rounds: int,
    round_wait_seconds: int,
) -> None:
    pending: List[DeletionTask] = list(tasks)
    stats: Counter[str] = Counter()

    for round_idx in range(1, max_rounds + 1):
        if not pending:
            break

        log(f"Starting round {round_idx}/{max_rounds} with {len(pending)} pending resources")
        next_pending: List[DeletionTask] = []
        progressed = 0

        total_in_round = len(pending)
        progress = None
        progress_task_id = None
        if (
            STDOUT_CONSOLE is not None
            and RICH_UI
            and RichProgress is not None
            and SpinnerColumn is not None
            and TextColumn is not None
            and BarColumn is not None
            and MofNCompleteColumn is not None
            and TimeElapsedColumn is not None
        ):
            progress = RichProgress(
                SpinnerColumn(),
                TextColumn("[bold blue]Round {task.fields[round_label]}", justify="left"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TextColumn("{task.fields[last_event]}", justify="left"),
                TimeElapsedColumn(),
                console=STDOUT_CONSOLE,
                transient=False,
            )
            progress.start()
            progress_task_id = progress.add_task(
                "delete-round",
                total=total_in_round,
                round_label=f"{round_idx}/{max_rounds}",
                last_event="starting",
            )

        try:
            for task_idx, task in enumerate(pending, start=1):
                label = f"{task.resource_type} [{task.region}] {task.resource_key}"
                log(
                    f"Task start {task_idx}/{total_in_round} in round {round_idx}/{max_rounds}: "
                    f"{label} (handler={task.handler_name})"
                )
                started_at = time.monotonic()
                handler = DELETE_HANDLERS[task.handler_name]
                result = handler(ctx, task)
                elapsed = time.monotonic() - started_at
                stats[result.status] += 1

                print_status_line(task, result)

                if progress is not None and progress_task_id is not None:
                    progress.update(
                        progress_task_id,
                        advance=1,
                        last_event=f"{result.status.upper()} {task_idx}/{total_in_round}",
                    )

                log(
                    f"Task end {task_idx}/{total_in_round} in round {round_idx}/{max_rounds}: "
                    f"{label} status={result.status} duration={elapsed:.1f}s"
                )

                if result.status in {"deleted", "skipped", "planned"}:
                    progressed += 1
                    continue
                if result.status == "deferred":
                    next_pending.append(task)
        finally:
            if progress is not None:
                progress.stop()

        pending = next_pending

        if pending and round_idx < max_rounds:
            if progressed == 0:
                warn("No progress this round while deferred resources remain. Stopping early.")
                break
            log(f"Waiting {round_wait_seconds}s before next round")
            time.sleep(max(0, round_wait_seconds))

    if pending:
        warn("Some resources are still pending (dependency or eventual consistency).")
        for task in pending:
            print_pending_line(task)

    log("Deletion summary:")
    print_stats_summary(stats)


def main() -> None:
    args = parse_args()

    if args.force and not args.apply:
        raise ValueError("--force is only valid with --apply")

    assert_profile_is_safe(args.profile, args.allow_non_dev_profile)
    tag_keys = parse_tag_keys(args.tag_keys)
    include_types = parse_csv_set(args.include_types)
    exclude_types = parse_csv_set(args.exclude_types)
    allow_generic_fallback = not args.disable_generic_fallback

    input_path = resolve_path(args.input, Path(args.results_dir), "--input")
    inventory = load_inventory(input_path)
    stage_re = stage_regex(args.tag_value)

    matched_records: List[Tuple[InventoryRecord, str]] = []
    all_records = list(iter_inventory_records(inventory, args.region))

    for record in all_records:
        if include_types and record.resource_type not in include_types:
            continue
        if record.resource_type in exclude_types:
            continue

        reason = match_reason(
            record=record,
            tag_value=args.tag_value,
            tag_keys=tag_keys,
            mode=args.match_mode,
            stage_re=stage_re,
        )
        if reason:
            matched_records.append((record, reason))

    if not matched_records:
        log("No resources matched the provided stage/tag and filters.")
        return

    records_only = [record for record, _ in matched_records]
    log(f"Inventory matched resources: {len(records_only)}")
    print_type_breakdown(summarize_by_type(records_only))

    tasks: List[DeletionTask] = []
    unsupported: List[InventoryRecord] = []
    for record, reason in matched_records:
        task = task_from_record(record, reason, allow_generic_fallback=allow_generic_fallback)
        if task is None:
            unsupported.append(record)
        else:
            tasks.append(task)

    tasks.sort(key=lambda item: (item.priority, item.resource_type, item.region, item.resource_key))
    print_plan(tasks, unsupported)

    if not tasks:
        log("No supported resources to delete after filtering. Nothing to do.")
        return

    session = boto3.Session(profile_name=args.profile)
    ctx = DeleteContext(
        session=session,
        profile=args.profile,
        default_region=args.region,
        dry_run=not args.apply,
        cloudcontrol_progress_seconds=max(0, args.cloudcontrol_progress_seconds),
        clients={},
    )

    sts = ctx.client("sts", args.region)
    identity = sts.get_caller_identity()
    log(f"AWS account (caller identity): {identity.get('Account', 'unknown')}")
    log(f"Profile: {args.profile}")
    log(f"Default region: {args.region}")
    log(f"Tag value: {args.tag_value}")
    log(f"Match mode: {args.match_mode}")
    log(f"Mode: {'apply' if args.apply else 'dry-run'}")

    if not args.apply:
        run_delete_rounds(
            ctx=ctx,
            tasks=tasks,
            max_rounds=1,
            round_wait_seconds=0,
        )
        log("Dry-run complete. Re-run with --apply to execute deletions.")
        return

    if not args.force:
        require_confirmation(len(tasks), args.profile, args.region, args.tag_value, input_path)

    run_delete_rounds(
        ctx=ctx,
        tasks=tasks,
        max_rounds=max(1, args.max_rounds),
        round_wait_seconds=max(0, args.round_wait_seconds),
    )


if __name__ == "__main__":
    main()
