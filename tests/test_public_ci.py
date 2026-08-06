from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_public_ci_is_deterministic_and_never_targets_personal_hardware() -> None:
    workflow = (
        REPOSITORY_ROOT / ".github/workflows/compiler-ci.yml"
    ).read_text(encoding="utf-8")

    assert "runs-on: ubuntu-latest" in workflow
    assert "self-hosted" not in workflow
    assert "groundupscale run" not in workflow
    assert "pytest -q" in workflow
    assert "diff -ru" in workflow
    assert "permissions:\n  contents: read" in workflow
