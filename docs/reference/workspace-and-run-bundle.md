# Workspace and Run Bundle layout

> **In one sentence:** GroundUpScale separates authored YAML Specs, generated
> immutable Run Bundles, promoted evidence, and test fixtures while using one
> Run Manifest to make every local or remote artifact self-describing.

## Source workspace

The recommended repository layout is:

```text
specs/
├── models/
├── workloads/
├── analysis-cases/
├── deployments/
├── hardware/
├── fabrics/
├── benchmarks/
└── plans/

plugins/

evidence/
├── datasets/
└── calibrations/

tests/
└── fixtures/
    ├── specs/
    ├── ir/
    └── observations/
```

The directories communicate intent but do not define identity. Every Spec still
uses its Spec Envelope and explicit Spec References.

### Configuration placement

- `models/` contains Model Specs and reusable Composite Module definitions.
- `workloads/` contains training, inference, RL, and service workflows.
- `analysis-cases/` contains Shape, Driver, and Observation Window conditions.
- `deployments/` contains scoped FSDP, offload, TP, PP, EP, CP, placement, and
  service policy configuration.
- `hardware/` contains reusable Hardware Specs.
- `fabrics/` contains concrete device, memory, storage, and interconnect topology.
- `benchmarks/` contains Benchmark Cases and Instrumentation Profiles.
- `plans/` contains Analysis Plans that assemble one reproducible analysis.

A strategy configuration belongs in Deployment Intent because its meaning
depends on scope. Optional reusable presets may be referenced, but each Run
Bundle locks the completely resolved effective Deployment Intent.

## Generated workspace

The default local Artifact Store uses `.groundupscale/`, which is ignored by Git:

```text
.groundupscale/
├── cache/
└── runs/
    └── <run-id>/
        ├── run.manifest.json
        ├── resolved/
        ├── ir/
        ├── prediction/
        ├── observation/
        ├── comparison/
        ├── calibration/
        ├── reports/
        └── logs/
```

Cache entries are disposable. Run Bundles are immutable evidence-bearing
artifacts and are never modified in place after completion.

## Run Bundle contents

```text
<run-id>/
├── run.manifest.json
├── resolved/
│   ├── inputs.lock.json
│   └── environment.json
├── ir/
│   ├── model.ir.json
│   ├── workload.ir.json
│   ├── semantic.ir.json
│   ├── cost.ir.json
│   └── execution.ir.json
├── prediction/
│   ├── metrics.json
│   ├── schedule.json
│   └── explanation.graph.json
├── observation/
│   ├── raw/
│   ├── observation.trace.jsonl
│   └── alignment.map.json
├── comparison/
│   ├── discrepancy.json
│   └── error-attribution.json
├── calibration/
│   └── candidate-profile.yaml
├── reports/
│   └── report.html
└── logs/
    └── events.jsonl
```

Artifacts are optional according to run purpose. A prediction-only run need not
contain observation or comparison outputs, but the manifest must state which
expected stages were skipped, failed, or completed.

## Run identity

Two identities are required:

- the deterministic Compilation Fingerprint identifies equivalent effective
  compilation inputs;
- the Run ID identifies one invocation or measurement attempt, because repeated
  observations of the same compilation are distinct evidence.

Neither an absolute path nor a `latest` alias is an authoritative identity.

## Run Manifest

The Run Manifest is the only required entry point for reading a bundle. It
contains:

```json
{
  "schema": "groundupscale.dev/run-manifest/v1alpha1",
  "run_id": "20260806T120000Z-01",
  "compilation_fingerprint": "sha256:...",
  "status": "completed",
  "hardware_cohort": "apple-m4-macos15-pytorch-mps",
  "artifacts": [
    {
      "role": "semantic-ir",
      "path": "ir/semantic.ir.json",
      "media_type": "application/json",
      "schema": "groundupscale.dev/semantic-ir/v1alpha1",
      "sha256": "...",
      "produced_by": "semantic-compiler@...",
      "inputs": ["artifact:workload-ir", "artifact:model-ir"]
    }
  ]
}
```

Every artifact entry declares its semantic role, Schema, content digest,
producer, and lineage. Readers resolve roles through the manifest rather than
guessing filenames.

## Format policy

| Artifact | Default format | Reason |
|---|---|---|
| human-authored Spec | YAML | readable and editable source contract |
| small structured IR and result | canonical JSON | deterministic hashing and broad tooling |
| normalized event stream | JSONL | append-friendly streaming and inspection |
| vendor trace | native raw format | evidence preservation and tool compatibility |
| large metric/sample table | Parquet when justified | columnar analysis without forcing it into the MVP |
| human report | standalone HTML | local and CI viewing without a server |

YAML remains the only human-authored Spec format. A generated candidate profile
may also use YAML because promotion turns it into a reviewed Calibration Profile;
its generated status and provenance remain explicit.

## Evidence promotion

Large raw traces normally stay in a CI Artifact or content-addressed object
store. The repository keeps:

```text
evidence/datasets/<dataset>.yaml
    immutable member hashes and artifact URIs

evidence/calibrations/<profile>.yaml
    reviewed and promoted Calibration Profiles
```

Candidate profiles remain inside Run Bundles until independent validation and
explicit promotion. Small deterministic observations needed by tests may be
copied into `tests/fixtures/observations/` with their provenance and expected
use declared.

## Storage adapters

The Run Manifest contract is independent of storage. Initial implementations may
use:

- a local filesystem Artifact Store;
- GitHub Actions artifacts for CI;
- a future object store for large traces and datasets.

Moving an artifact changes its location and possibly its URI, not its content
identity. The digest, Schema, role, and lineage remain stable.

## Lifecycle and safety rules

- Completed Run Bundles are immutable; reruns create new Run IDs.
- Partial runs preserve successfully written artifacts and mark status clearly.
- Writers use a temporary directory and atomically publish the final manifest.
- Secrets and unrestricted environment dumps are never captured.
- Cache cleanup and Run Bundle retention are separate operations.
- CI uploads the complete manifest plus all locally available referenced files.
- `latest` may be a convenience index but cannot be cited as evidence.
