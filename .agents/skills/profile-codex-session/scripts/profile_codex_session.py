#!/usr/bin/env python3
"""Produce a privacy-preserving, top-down latency profile of a Codex JSONL session."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CATEGORY_LABELS = {
    "remote": "远程/加速器命令",
    "build_test": "测试/构建",
    "package": "依赖/包管理",
    "vcs": "Git/版本控制",
    "github": "GitHub/外部协作",
    "read_search": "读取/搜索",
    "edit": "修改文件",
    "edit_validate": "修改并验证",
    "command_wait": "命令等待/轮询",
    "subagent_wait": "子代理等待",
    "subagent_control": "子代理调度",
    "plan": "计划更新",
    "web_mcp": "Web/MCP",
    "shell_other": "其他命令",
    "tool_other": "其他工具",
}


@dataclass(frozen=True)
class Event:
    at: float
    raw: dict[str, Any]


@dataclass(frozen=True)
class CallSpan:
    call_id: str
    name: str
    category: str
    start: float
    end: float
    complete: bool
    source: str
    output: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class TaskWindow:
    turn_id: str
    start: float
    end: float
    complete: bool
    duration_ms: float | None = None
    ttft_ms: float | None = None


def parse_iso(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def iso_time(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    valid = sorted((start, end) for start, end in intervals if end > start)
    merged: list[list[float]] = []
    for start, end in valid:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_length(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def intersect_interval(
    interval: tuple[float, float], windows: Iterable[tuple[float, float]]
) -> list[tuple[float, float]]:
    start, end = interval
    return [
        (max(start, window_start), min(end, window_end))
        for window_start, window_end in windows
        if min(end, window_end) > max(start, window_start)
    ]


def output_text(payload: dict[str, Any]) -> str:
    output = payload.get("output", "")
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in output
        )
    return str(output)


def resolve_session(target: str, codex_home: Path) -> tuple[Path, list[Path]]:
    direct = Path(target).expanduser()
    if direct.is_file():
        return direct.resolve(), []

    matches: list[Path] = []
    for directory in (codex_home / "sessions", codex_home / "archived_sessions"):
        if directory.is_dir():
            matches.extend(directory.rglob(f"*{target}*.jsonl"))
    if not matches:
        raise FileNotFoundError(
            f"找不到会话 {target!r}；已搜索 {codex_home / 'sessions'} 和 "
            f"{codex_home / 'archived_sessions'}"
        )
    ordered = sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)
    return ordered[0].resolve(), [path.resolve() for path in ordered[1:]]


def load_events(path: Path) -> tuple[list[Event], int]:
    events: list[Event] = []
    malformed = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                timestamp = raw.get("timestamp")
                if not timestamp:
                    continue
                events.append(Event(parse_iso(timestamp), raw))
            except (json.JSONDecodeError, TypeError, ValueError):
                malformed += 1
    events.sort(key=lambda event: event.at)
    if not events:
        raise ValueError(f"{path} 中没有可解析且带时间戳的事件")
    return events, malformed


def session_id(events: list[Event], fallback: str) -> str:
    for event in events:
        if event.raw.get("type") == "session_meta":
            payload = event.raw.get("payload", {})
            return str(payload.get("id") or payload.get("session_id") or fallback)
    match = re.search(r"([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})", fallback)
    return match.group(1) if match else fallback


def build_task_windows(events: list[Event]) -> list[TaskWindow]:
    starts: dict[str, float] = {}
    windows: list[TaskWindow] = []
    last_event = events[-1].at

    for event in events:
        if event.raw.get("type") != "event_msg":
            continue
        payload = event.raw.get("payload", {})
        kind = payload.get("type")
        turn_id = str(payload.get("turn_id") or "unknown")
        if kind == "task_started":
            starts[turn_id] = event.at
        elif kind == "task_complete":
            start = starts.pop(turn_id, None)
            if start is None:
                started_at = payload.get("started_at")
                start = float(started_at) if isinstance(started_at, (int, float)) else event.at
            duration_ms = payload.get("duration_ms")
            ttft_ms = payload.get("time_to_first_token_ms")
            windows.append(
                TaskWindow(
                    turn_id=turn_id,
                    start=start,
                    end=event.at,
                    complete=True,
                    duration_ms=float(duration_ms) if isinstance(duration_ms, (int, float)) else None,
                    ttft_ms=float(ttft_ms) if isinstance(ttft_ms, (int, float)) else None,
                )
            )

    for turn_id, start in starts.items():
        windows.append(TaskWindow(turn_id, start, max(start, last_event), False))

    if not windows:
        windows.append(TaskWindow("inferred", events[0].at, last_event, False))
    return sorted(windows, key=lambda window: window.start)


def nested_tools(source: str) -> list[str]:
    return re.findall(r"tools\.([A-Za-z0-9_]+)\s*\(", source)


def classify_call(name: str, source: str) -> str:
    lowered = source.lower()
    tools = set(nested_tools(source))

    if name in {"wait_agent"}:
        return "subagent_wait"
    if name in {"spawn_agent", "send_message", "followup_task", "list_agents", "interrupt_agent"}:
        return "subagent_control"
    if name in {"wait", "write_stdin"}:
        return "command_wait"
    if name in {"update_plan"} or "update_plan" in tools:
        return "plan"
    if any(token in name for token in ("web", "mcp")):
        return "web_mcp"

    edits = bool({"apply_patch"} & tools) or name == "apply_patch"
    command = bool({"exec_command"} & tools) or name in {"exec", "exec_command"}
    validates = bool(
        re.search(
            r"\b(pytest|unittest|npm\s+(?:test|run)|pnpm\s+(?:test|run)|yarn\s+(?:test|run)|"
            r"cargo\s+(?:test|build|check)|go\s+test|make(?:\s|$)|cmake|ninja|ruff|mypy|"
            r"tsc|quick_validate|--self-test)\b",
            lowered,
        )
    )
    if edits and validates:
        return "edit_validate"
    if edits:
        return "edit"
    if not command:
        return "tool_other"

    if re.search(r"\b(ssh|scp|rsync)\b|npu-smi|ascend|cuda|remote", lowered):
        return "remote"
    if validates:
        return "build_test"
    if re.search(r"\b(pip|uv|poetry|conda|npm|pnpm|yarn|brew)\b.*\b(install|add|sync|update)", lowered):
        return "package"
    if re.search(r"\bgh\s+(?:pr|issue|api|run)|github", lowered):
        return "github"
    if re.search(r"\bgit\s+", lowered):
        return "vcs"
    if re.search(
        r"\b(rg|grep|find|sed|head|tail|less|wc|stat|ls|pwd|jq)\b|"
        r"read_mcp_resource|list_mcp_resource",
        lowered,
    ):
        return "read_search"
    return "shell_other"


def pair_calls(events: list[Event]) -> tuple[list[CallSpan], int]:
    starts: dict[str, tuple[float, str, str, str]] = {}
    calls: list[CallSpan] = []
    unmatched_outputs = 0
    last_event = events[-1].at

    for event in events:
        if event.raw.get("type") != "response_item":
            continue
        payload = event.raw.get("payload", {})
        kind = payload.get("type")
        call_id = str(payload.get("call_id") or payload.get("id") or "")
        if not call_id:
            continue
        if kind in {"custom_tool_call", "function_call"}:
            name = str(payload.get("name") or "unknown")
            source = str(payload.get("input") or payload.get("arguments") or "")
            starts[call_id] = (event.at, name, source, kind)
        elif kind in {"custom_tool_call_output", "function_call_output"}:
            start = starts.pop(call_id, None)
            if start is None:
                unmatched_outputs += 1
                continue
            started_at, name, source, _ = start
            calls.append(
                CallSpan(
                    call_id=call_id,
                    name=name,
                    category=classify_call(name, source),
                    start=started_at,
                    end=max(started_at, event.at),
                    complete=True,
                    source=source,
                    output=output_text(payload),
                )
            )

    for call_id, (started_at, name, source, _) in starts.items():
        calls.append(
            CallSpan(
                call_id=call_id,
                name=name,
                category=classify_call(name, source),
                start=started_at,
                end=max(started_at, last_event),
                complete=False,
                source=source,
                output="",
            )
        )
    return sorted(calls, key=lambda call: call.start), unmatched_outputs


def background_intervals(calls: list[CallSpan]) -> list[tuple[float, float]]:
    """Infer command lifetime after a yielded exec returns and before its final wait returns."""
    opened: dict[str, float] = {}
    intervals: list[tuple[float, float]] = []
    cell_pattern = re.compile(r"Script running with cell ID\s+([^\s]+)")

    for call in calls:
        match = cell_pattern.search(call.output)
        if match and call.category != "command_wait":
            opened.setdefault(match.group(1), call.end)

        if call.name not in {"wait", "write_stdin"}:
            continue
        try:
            arguments = json.loads(call.source)
        except json.JSONDecodeError:
            arguments = {}
        cell_id = str(arguments.get("cell_id") or arguments.get("session_id") or "")
        if not cell_id:
            continue
        if "Script running with cell ID" in call.output:
            opened.setdefault(cell_id, call.start)
        elif cell_id in opened:
            intervals.append((opened.pop(cell_id), call.end))
    return intervals


def token_summary(events: list[Event]) -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    context_windows: list[int] = []
    for event in events:
        if event.raw.get("type") != "event_msg":
            continue
        payload = event.raw.get("payload", {})
        if payload.get("type") != "token_count":
            continue
        info = payload.get("info") or {}
        usage = info.get("last_token_usage")
        if isinstance(usage, dict):
            requests.append(usage)
        window = info.get("model_context_window")
        if isinstance(window, int):
            context_windows.append(window)

    input_values = [float(request.get("input_tokens", 0)) for request in requests]
    cached_values = [float(request.get("cached_input_tokens", 0)) for request in requests]
    output_values = [float(request.get("output_tokens", 0)) for request in requests]
    total_input = sum(input_values)
    return {
        "requests": len(requests),
        "input_tokens": int(total_input),
        "cached_input_tokens": int(sum(cached_values)),
        "output_tokens": int(sum(output_values)),
        "cache_ratio": (sum(cached_values) / total_input) if total_input else None,
        "input_p50": percentile(input_values, 0.5),
        "input_p90": percentile(input_values, 0.9),
        "input_max": max(input_values) if input_values else None,
        "context_windows": sorted(set(context_windows)),
    }


def model_configs(events: list[Event]) -> list[str]:
    configs: set[str] = set()
    for event in events:
        if event.raw.get("type") != "turn_context":
            continue
        payload = event.raw.get("payload", {})
        model = payload.get("model")
        collaboration = payload.get("collaboration_mode") or {}
        settings = collaboration.get("settings") if isinstance(collaboration, dict) else {}
        effort = settings.get("reasoning_effort") if isinstance(settings, dict) else None
        if model:
            configs.add(f"{model}/{effort or 'unknown'}")
    return sorted(configs)


def build_report(
    path: Path,
    events: list[Event],
    malformed: int,
    alternatives: list[Path],
    top_n: int,
) -> dict[str, Any]:
    tasks = build_task_windows(events)
    task_intervals = merge_intervals((task.start, task.end) for task in tasks)
    active_seconds = interval_length(task_intervals)
    session_span = max(0.0, events[-1].at - events[0].at)
    human_idle = max(0.0, session_span - active_seconds)

    calls, unmatched_outputs = pair_calls(events)
    clipped_by_call: dict[str, list[tuple[float, float]]] = {}
    foreground: list[tuple[float, float]] = []
    category_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    category_counts: Counter[str] = Counter()
    incomplete_calls = 0
    for call in calls:
        clipped = intersect_interval((call.start, call.end), task_intervals)
        if not clipped:
            continue
        clipped_by_call[call.call_id] = clipped
        foreground.extend(clipped)
        category_intervals[call.category].extend(clipped)
        category_counts[call.category] += 1
        if not call.complete:
            incomplete_calls += 1

    foreground_seconds = interval_length(foreground)
    background = []
    for interval in background_intervals(calls):
        background.extend(intersect_interval(interval, task_intervals))
    observed_intervals = foreground + background
    observed_seconds = interval_length(observed_intervals)
    background_only_seconds = max(0.0, observed_seconds - foreground_seconds)
    residual_seconds = max(0.0, active_seconds - observed_seconds)

    category_rows = []
    summed_category_seconds = 0.0
    for category, intervals in category_intervals.items():
        seconds = interval_length(intervals)
        summed_category_seconds += seconds
        category_rows.append(
            {
                "category": category,
                "label": CATEGORY_LABELS.get(category, category),
                "calls": category_counts[category],
                "seconds": seconds,
                "active_ratio": seconds / active_seconds if active_seconds else None,
            }
        )
    category_rows.sort(key=lambda row: row["seconds"], reverse=True)

    top_calls = []
    for call in calls:
        clipped = clipped_by_call.get(call.call_id, [])
        seconds = interval_length(clipped)
        if seconds <= 0:
            continue
        top_calls.append(
            {
                "category": call.category,
                "label": CATEGORY_LABELS.get(call.category, call.category),
                "tool": call.name,
                "seconds": seconds,
                "complete": call.complete,
            }
        )
    top_calls.sort(key=lambda row: row["seconds"], reverse=True)

    exact_durations = [task.duration_ms for task in tasks if task.duration_ms is not None]
    exact_ttft = [task.ttft_ms for task in tasks if task.ttft_ms is not None]
    compacted = sum(
        1
        for event in events
        if (
            event.raw.get("type") == "compacted"
            or (
                event.raw.get("type") == "event_msg"
                and event.raw.get("payload", {}).get("type") == "context_compacted"
            )
        )
    )

    return {
        "session": {
            "id": session_id(events, path.name),
            "path": str(path),
            "snapshot_utc": iso_time(events[-1].at),
            "first_event_utc": iso_time(events[0].at),
            "active": any(not task.complete for task in tasks),
            "alternative_matches": [str(candidate) for candidate in alternatives],
            "malformed_lines_skipped": malformed,
        },
        "time": {
            "session_span_seconds": session_span,
            "active_task_seconds": active_seconds,
            "human_idle_between_tasks_seconds": human_idle,
            "foreground_tool_union_seconds": foreground_seconds,
            "background_command_only_seconds": background_only_seconds,
            "observed_tool_or_background_seconds": observed_seconds,
            "model_api_orchestration_unattributed_seconds": residual_seconds,
            "observed_ratio": observed_seconds / active_seconds if active_seconds else None,
            "residual_ratio": residual_seconds / active_seconds if active_seconds else None,
            "category_overlap_seconds": max(0.0, summed_category_seconds - foreground_seconds),
        },
        "tasks": {
            "count": len(tasks),
            "complete": sum(task.complete for task in tasks),
            "incomplete": sum(not task.complete for task in tasks),
            "exact_duration_ms_p50": percentile(exact_durations, 0.5),
            "exact_duration_ms_p90": percentile(exact_durations, 0.9),
            "exact_ttft_ms_p50": percentile(exact_ttft, 0.5),
            "exact_ttft_ms_p90": percentile(exact_ttft, 0.9),
        },
        "tools": {
            "paired_or_open_calls": len(calls),
            "incomplete_calls": incomplete_calls,
            "unmatched_outputs": unmatched_outputs,
            "categories": category_rows,
            "top_slowest": top_calls[:top_n],
        },
        "model": {
            "configurations": model_configs(events),
            "compactions": compacted,
            **token_summary(events),
        },
        "method": {
            "denominator": "task_started 到 task_complete；活跃任务截止 JSONL 最后事件",
            "overlap": "工具与后台命令按时间区间取并集，避免并行重复计时",
            "privacy": "报告不输出提示词、命令正文、工具输出或文件内容",
            "residual_limit": (
                "残差包含模型服务排队、推理、流式传输、网络、本地编排、思考及未记录等待；"
                "仅凭 JSONL 不能继续可靠拆分"
            ),
        },
    }


def fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 3600:
        return f"{value / 3600:.2f} h"
    if value >= 60:
        return f"{value / 60:.2f} min"
    return f"{value:.2f} s"


def fmt_integer(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def render_markdown(report: dict[str, Any]) -> str:
    session = report["session"]
    timing = report["time"]
    tasks = report["tasks"]
    tools = report["tools"]
    model = report["model"]
    residual_pct = (timing["residual_ratio"] or 0.0) * 100
    observed_pct = (timing["observed_ratio"] or 0.0) * 100
    cache_pct = model["cache_ratio"] * 100 if model["cache_ratio"] is not None else None
    configs = ", ".join(model["configurations"]) or "未记录"

    lines = [
        f"# Codex 会话耗时分析：{session['id']}",
        "",
        f"- 快照：`{session['snapshot_utc']}`；状态：{'仍在执行' if session['active'] else '已完成'}",
        f"- 活跃任务时间：**{fmt_seconds(timing['active_task_seconds'])}**；任务外人为间隔：{fmt_seconds(timing['human_idle_between_tasks_seconds'])}",
        f"- 已观测工具/后台命令：**{fmt_seconds(timing['observed_tool_or_background_seconds'])}（{observed_pct:.2f}%）**",
        f"- 模型/API/编排/未归因残差：**{fmt_seconds(timing['model_api_orchestration_unattributed_seconds'])}（{residual_pct:.2f}%）**",
        "",
        "## Top-down 构成",
        "",
        "| 类别 | 调用数 | 去重前类别耗时 | 占活跃时间 |",
        "|---|---:|---:|---:|",
    ]
    for row in tools["categories"]:
        ratio = row["active_ratio"] * 100 if row["active_ratio"] is not None else 0.0
        lines.append(f"| {row['label']} | {row['calls']} | {fmt_seconds(row['seconds'])} | {ratio:.2f}% |")
    lines.extend(
        [
            f"| 后台命令（仅未被前台覆盖部分） | — | {fmt_seconds(timing['background_command_only_seconds'])} | — |",
            f"| 模型/API/编排/未归因残差 | — | {fmt_seconds(timing['model_api_orchestration_unattributed_seconds'])} | {residual_pct:.2f}% |",
            "",
        ]
    )
    if timing["category_overlap_seconds"] > 0.001:
        lines.append(
            f"> 类别行存在 {fmt_seconds(timing['category_overlap_seconds'])} 并行重叠；总占比使用区间并集，不重复累计。"
        )
        lines.append("")

    lines.extend(
        [
            "## 最慢工具调用",
            "",
            "| 排名 | 类别 | 工具 | 耗时 | 完整 |",
            "|---:|---|---|---:|---|",
        ]
    )
    for index, row in enumerate(tools["top_slowest"], 1):
        lines.append(
            f"| {index} | {row['label']} | `{row['tool']}` | {fmt_seconds(row['seconds'])} | {'是' if row['complete'] else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 模型请求与上下文",
            "",
            f"- 配置：{configs}",
            f"- 模型请求：{model['requests']}；输入 token：{fmt_integer(model['input_tokens'])}；输出 token：{fmt_integer(model['output_tokens'])}",
            f"- 输入缓存命中：{f'{cache_pct:.2f}%' if cache_pct is not None else 'n/a'}；单次输入 p50/p90/max：{fmt_integer(model['input_p50'])}/{fmt_integer(model['input_p90'])}/{fmt_integer(model['input_max'])}",
            f"- 上下文压缩：{model['compactions']} 次；上下文窗口：{', '.join(map(str, model['context_windows'])) or '未记录'}",
            f"- 任务：{tasks['count']} 个（完成 {tasks['complete']}，未完成 {tasks['incomplete']}）",
        ]
    )
    if tasks["exact_duration_ms_p50"] is not None:
        lines.append(
            f"- 已完成任务精确 E2E p50/p90：{fmt_seconds(tasks['exact_duration_ms_p50'] / 1000)}/{fmt_seconds(tasks['exact_duration_ms_p90'] / 1000)}"
        )
    if tasks["exact_ttft_ms_p50"] is not None:
        lines.append(
            f"- 已完成任务 TTFT p50/p90：{fmt_seconds(tasks['exact_ttft_ms_p50'] / 1000)}/{fmt_seconds(tasks['exact_ttft_ms_p90'] / 1000)}"
        )

    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- {report['method']['residual_limit']}。要把残差拆成服务端排队/推理/流式传输，需要 OpenTelemetry 或 API 网关追踪。",
            f"- 计时口径：{report['method']['denominator']}；{report['method']['overlap']}。",
            f"- 数据质量：跳过损坏/未写完 JSONL 行 {session['malformed_lines_skipped']} 条；未闭合工具调用 {tools['incomplete_calls']} 个；无起点输出 {tools['unmatched_outputs']} 个。",
            f"- 隐私：{report['method']['privacy']}。",
        ]
    )
    if session["alternative_matches"]:
        lines.append(f"- 同一标识另有 {len(session['alternative_matches'])} 个匹配，已选择修改时间最新者。")
    return "\n".join(lines) + "\n"


def run_self_test() -> None:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()

    def item(offset: float, record_type: str, payload: dict[str, Any]) -> Event:
        return Event(base + offset, {"type": record_type, "payload": payload})

    events = [
        item(0, "event_msg", {"type": "task_started", "turn_id": "turn-1"}),
        item(
            1,
            "response_item",
            {
                "type": "custom_tool_call",
                "call_id": "call-1",
                "name": "exec",
                "input": "const r = await tools.exec_command({cmd: 'rg needle .'});",
            },
        ),
        item(
            3,
            "response_item",
            {"type": "custom_tool_call_output", "call_id": "call-1", "output": "Script completed"},
        ),
        item(
            10,
            "event_msg",
            {
                "type": "task_complete",
                "turn_id": "turn-1",
                "duration_ms": 10000,
                "time_to_first_token_ms": 1500,
            },
        ),
    ]
    report = build_report(Path("synthetic.jsonl"), events, 0, [], 5)
    assert abs(report["time"]["active_task_seconds"] - 10.0) < 1e-9
    assert abs(report["time"]["observed_tool_or_background_seconds"] - 2.0) < 1e-9
    assert abs(report["time"]["residual_ratio"] - 0.8) < 1e-9
    assert report["tasks"]["exact_ttft_ms_p50"] == 1500
    assert report["tools"]["categories"][0]["category"] == "read_search"
    print("self-test: PASS")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="分析 Codex 会话 JSONL 的 top-down 耗时构成，不输出敏感正文。"
    )
    parser.add_argument("session", nargs="?", help="会话 ID 或 JSONL 文件路径")
    parser.add_argument(
        "--codex-home",
        type=Path,
        default=Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))),
        help="Codex 数据目录（默认 $CODEX_HOME 或 ~/.codex）",
    )
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--top", type=int, default=10, help="显示最慢工具调用数量")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.self_test:
        run_self_test()
        return 0
    if not args.session:
        print("error: 请提供会话 ID 或 JSONL 文件路径", file=sys.stderr)
        return 2
    if args.top < 1:
        print("error: --top 必须大于 0", file=sys.stderr)
        return 2
    try:
        path, alternatives = resolve_session(args.session, args.codex_home.expanduser())
        events, malformed = load_events(path)
        report = build_report(path, events, malformed, alternatives, args.top)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
