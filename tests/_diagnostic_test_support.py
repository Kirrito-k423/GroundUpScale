from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


DIAGNOSTIC_INPUT_KEYS = (
    "resolved_configuration",
    "resolved_ir",
    "hardware",
    "cohort_id",
    "execution_domain",
)


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def refresh_diagnostic_bundle(
    run: Path,
    document: dict[str, object],
    *,
    run_id: str,
    hardware_cohort: str,
) -> Path:
    inputs = {key: document[key] for key in DIAGNOSTIC_INPUT_KEYS}
    evidence = {
        key: value
        for key, value in document.items()
        if key not in {*DIAGNOSTIC_INPUT_KEYS, "schema", "digests"}
    }
    document["digests"] = {
        "input_sha256": canonical_digest(inputs),
        "evidence_sha256": canonical_digest(evidence),
    }
    evidence_path = run / "diagnostic/evidence.json"
    artifact_digest = write_json(evidence_path, document)
    manifest_path = run / "run.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = run_id
    manifest["hardware_cohort"] = hardware_cohort
    manifest["artifacts"][0]["sha256"] = artifact_digest
    write_json(manifest_path, manifest)
    return run
