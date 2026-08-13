---
name: profile-codex-session
description: Analyze a Codex task, thread, conversation, or rollout JSONL by session ID or file path and produce a reproducible top-down latency report. Use when a Codex AI development task feels slow and the user wants to distinguish command execution, file reading/searching, file edits, tests/builds, remote work, sub-agent waits, model request volume, context growth, compaction, human idle time, and the remaining model/API/orchestration latency. Supports active and archived local Codex sessions under ~/.codex.
---

# Profile Codex Session

Use the bundled analyzer as the source of truth for arithmetic. Do not estimate durations by reading timestamps manually unless the script cannot parse the session.

## Run the analyzer

Resolve a session ID automatically:

```bash
python3 scripts/profile_codex_session.py <session-id>
```

Analyze a specific rollout or emit machine-readable data:

```bash
python3 scripts/profile_codex_session.py /absolute/path/to/rollout.jsonl
python3 scripts/profile_codex_session.py <session-id> --format json --top 20
```

When invoking the script from outside this skill directory, use its absolute path. Pass `--codex-home PATH` only when Codex data is not under `$CODEX_HOME` or `~/.codex`.

## Workflow

1. Run the script immediately when the user supplies a session ID or JSONL path.
2. Treat the output as a snapshot if the task is still active. State the snapshot timestamp.
3. Lead with the active-task denominator, observed tool/background union, and residual share.
4. Identify the largest measured categories and slow calls. Use counts plus time; high call count with low time is workflow churn, not the primary latency source.
5. Inspect model request count, per-request input size, cache ratio, model/reasoning configuration, and compactions. Large cached prompts still create request-processing work.
6. Keep human idle between separate task turns outside the active-work denominator.
7. Give recommendations only for dominant evidence-backed causes.
8. If precise model-service attribution is required, recommend OpenTelemetry or gateway traces rather than claiming the JSONL can provide it.

## Interpret the report correctly

- Tool timing is the interval from a recorded call to its output. Parallel intervals are unioned for the headline percentage.
- Yielded commands continue in the background. Count only their uncovered lifetime beyond foreground tool spans.
- Category rows can overlap when parallel calls use different categories; do not sum them as a wall-clock total.
- `模型/API/编排/未归因残差` is subtraction, not direct inference timing. It can contain service queueing, inference, streaming, network delay, local orchestration, reasoning, and unrecorded waits.
- `task_complete.duration_ms` and `time_to_first_token_ms` are exact when Codex recorded them. Active tasks lack final exact values.
- A malformed final JSONL line is normal while a session is being written. Report skipped rows and incomplete calls instead of failing the whole analysis.
- Never print raw prompts, commands, tool output, or file content in the report.

## Recommend by dominant cause

- Remote/test/build dominated: narrow test scope, remove redundant reruns, parallelize independent checks, and keep long processes in one monitored session.
- Read/search dominated: use targeted `rg`, reuse earlier findings, avoid repeatedly loading the same large files, and delegate distinct searches only when parallel work is warranted.
- Edit churn dominated: batch coherent edits, define acceptance criteria earlier, and validate at stable seams.
- Residual dominated with large contexts or many requests: reduce repeated context, lower reasoning effort when risk allows, split long phases, and avoid unnecessary model round trips.
- TTFT dominated: inspect service queue, network, proxy, and gateway telemetry.
- Human idle dominated: report it separately; do not blame the agent runtime.

## Validate the analyzer

Run the built-in arithmetic check after changing the script:

```bash
python3 scripts/profile_codex_session.py --self-test
```

Then analyze one real active or archived session and one nonexistent ID. The latter must fail with exit code 2 and a concise diagnostic.
