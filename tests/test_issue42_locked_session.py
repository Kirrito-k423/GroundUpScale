from __future__ import annotations

import importlib.util
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
SESSION_SCRIPT = (
    REPOSITORY_ROOT
    / "goal_process/issue-42-transformer-matmul-frontier/run_locked_session.py"
)
SHELL_ENTRY = (
    REPOSITORY_ROOT
    / "goal_process/issue-42-transformer-matmul-frontier/run_locked_session.sh"
)


def test_issue42_locked_session_is_bounded_and_pre_registered() -> None:
    spec = importlib.util.spec_from_file_location("issue42_locked_session", SESSION_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.DOMAINS == (
        "attention-context",
        "attention-qk",
        "mlp-contract",
        "mlp-expand",
        "projection",
    )
    assert module.SEARCH_SESSIONS == 3
    assert module.HOLDOUT_SESSIONS == 3
    assert module.WARMUP_ITERATIONS == 100
    assert module.REPETITIONS == 100
    assert module.INNER_ITERATIONS == 100
    script = SHELL_ENTRY.read_text(encoding="utf-8")
    assert "ASCEND_RT_VISIBLE_DEVICES" in script
    assert "GROUNDUPSCALE_ISSUE" in script
    assert "/home/t00906153/GroundUpScale-issue-42" in script
    assert "ascend-910b2-host.owner" in script
    assert "/home/miniconda3/envs/lmz_pt27py311/bin/python3.11" in script
    assert 'PYTHONPATH="$repository/src"' in script
    assert "uv run" not in script
    assert "issue-42-transformer-matmul-frontier/tmp/$session_id" in script
