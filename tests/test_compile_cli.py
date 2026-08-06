from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_compile_cli_writes_inspectable_structural_and_semantic_artifacts(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "compile",
            str(REPOSITORY_ROOT / "specs/plans/mac-cpu-prefill.yaml"),
            "--repository-root",
            str(REPOSITORY_ROOT),
            "--output",
            str(tmp_path),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["semantic_operation_count"] == 54
    assert summary["semantic_compilation_fingerprint"]
    for artifact in (
        "model-ir.json",
        "workload-ir.json",
        "semantic-ir.json",
        "provenance.json",
        "compilation.json",
    ):
        path = tmp_path / artifact
        assert path.is_file()
        assert json.loads(path.read_text(encoding="utf-8"))
