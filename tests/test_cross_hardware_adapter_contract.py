from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from groundupscale.diagnostics import diagnose_run_bundle, render_diagnostic_report
from groundupscale.measurement_contract import MeasurementAdapter
from _diagnostic_test_support import (
    canonical_digest as _canonical_digest,
    refresh_diagnostic_bundle,
)
from test_capability_surface import _surface_version
from test_diagnostic_bundle import _write_frozen_m4_bundle


class _RecordedFixtureAdapter:
    def __init__(self, document: dict[str, object]) -> None:
        self._document = deepcopy(document)
        self.operations: list[dict[str, str]] = []

    def _record(self, operation: str) -> None:
        self.operations.append(
            {
                "operation": operation,
                "evidence_ref": f"artifact://adapter-operations/{operation}.json",
            }
        )

    def discover_capabilities(self) -> dict[str, object]:
        self._record("discover_capabilities")
        return deepcopy(self._document["measurement_capability_manifest"])

    def fingerprint_cohort(self) -> dict[str, object]:
        self._record("fingerprint_cohort")
        hardware = self._document["hardware"]
        execution_domain = self._document["execution_domain"]
        baseline = self._document["baseline_timing_lane"]
        adapter = self._document["measurement_adapter"]
        return {
            "device": hardware.get("device"),
            "partition": hardware.get("partition"),
            "topology": hardware.get("topology"),
            "software": hardware.get("software"),
            "power_clock": deepcopy(hardware["power_clock"]),
            "numeric_execution": {
                key: execution_domain[key]
                for key in (
                    "dtype",
                    "layout",
                    "alignment_bytes",
                    "threads",
                    "execution_mode",
                )
            },
            "timer_protocol": {
                "source": baseline["timer"]["source"],
                "resolution_ns": baseline["timer"]["resolution_ns"],
                "monotonic": baseline["timer"]["monotonic"],
                "completion_kind": baseline["completion_boundary"]["kind"],
                "duration_reducer": baseline["completion_boundary"].get(
                    "duration_reducer"
                ),
                "adapter_id": adapter["adapter_id"],
                "adapter_version": adapter["adapter_version"],
                "protocol_id": adapter["protocol_id"],
                "protocol_version": adapter["protocol_version"],
            },
            "execution_context": {
                key: execution_domain[key]
                for key in (
                    "affinity",
                    "numa",
                    "context",
                    "stream",
                    "concurrency",
                )
            },
            "communication": deepcopy(
                self._document["communication_identity"]
            ),
        }

    def preflight(self) -> dict[str, object]:
        self._record("preflight")
        return deepcopy(self._document["environment"])

    def build_timing_plan(self, case: dict[str, object]) -> dict[str, object]:
        self._record("build_timing_plan")
        return {
            "case": deepcopy(case),
            "pair_id": self._document["baseline_timing_lane"]["pair_id"],
            "baseline_lane_id": self._document["baseline_timing_lane"][
                "lane_id"
            ],
            "diagnostic_lane_id": self._document[
                "diagnostic_profiling_lane"
            ]["lane_id"],
            "completion_boundary": deepcopy(
                self._document["baseline_timing_lane"]["completion_boundary"]
            ),
            "evidence_ref": "artifact://adapter-operations/timing-plan.json",
        }

    def collect(
        self,
        case: dict[str, object],
        timing_plan: dict[str, object],
    ) -> dict[str, object]:
        self._record("collect")
        return {
            "case": deepcopy(case),
            "timing_plan_ref": timing_plan["evidence_ref"],
            "baseline_timing_lane": deepcopy(
                self._document["baseline_timing_lane"]
            ),
            "diagnostic_profiling_lane": deepcopy(
                self._document["diagnostic_profiling_lane"]
            ),
        }


def _rewrite_bundle(
    root: Path,
    *,
    adapter_id: str,
    device: str,
    cohort_id: str,
    candidate_id: str,
    candidate_family: str,
    invalid_required_evidence: str | None = None,
    cohort_change: str | None = None,
    reference_cohort_id: str | None = None,
    transient_failure: str | None = None,
    omit_capability_manifest: bool = False,
    omit_cohort_evidence: bool = False,
    omit_derived_provenance: bool = False,
    surface_cohort_id: str | None = None,
    self_reported_profile_qualification: bool = False,
    invalid_completion_protocol: str | None = None,
    qualified_profile_ablation: bool = False,
) -> Path:
    run = _write_frozen_m4_bundle(root)
    evidence_path = run / "diagnostic/evidence.json"
    document = json.loads(evidence_path.read_text(encoding="utf-8"))

    document["hardware"]["device"] = device
    document["hardware"]["software"] = f"{adapter_id}-runtime-v1"
    document["hardware"]["power_clock"] = {
        "power_policy": "balanced",
        "clock_policy": "automatic",
    }
    document["cohort_id"] = cohort_id
    document["execution_domain"].update(
        {
            "affinity": "all-performance-cores",
            "numa": "single-domain",
            "context": "default",
            "stream": "not_applicable",
            "concurrency": 1,
        }
    )
    document["communication_identity"] = {"status": "not_applicable"}
    document["candidate"]["candidate_id"] = candidate_id
    document["candidate"]["family"] = candidate_family
    document["candidate"]["exact_shape_best_of_correct"][
        "winner_candidate_id"
    ] = candidate_id
    document["candidate"]["exact_shape_best_of_correct"][
        "eligible_candidate_ids"
    ] = [candidate_id]
    document["single_node_schedule"]["candidate_id"] = candidate_id

    anchor = document["frontier_anchors"][0]
    anchor_id = f"anchor-{adapter_id}-q-proj-001"
    anchor["anchor_id"] = anchor_id
    anchor["candidate_id"] = candidate_id
    anchor["cohort_id"] = cohort_id
    anchor["execution_domain"] = document["execution_domain"]
    anchor["baseline_lane_id"] = f"baseline-{adapter_id}-q-proj-holdout-001"
    anchor["evidence_ref"] = f"artifact://frontier/{anchor_id}.json"
    document["resource_physical_floor"]["resource_terms"][0][
        "cohort_id"
    ] = cohort_id
    document["resource_physical_floor"]["resource_terms"][0][
        "execution_domain"
    ] = document["execution_domain"]

    surface = _surface_version("v1", previous_version=None)
    surface_id = f"surface://{adapter_id}/matmul/1d"
    surface["surface_id"] = surface_id
    surface["cohort_id"] = surface_cohort_id or cohort_id
    surface["candidate_family"] = candidate_family
    for surface_anchor in surface["anchors"]:
        surface_anchor["anchor_id"] = (
            f"{adapter_id}-{surface_anchor['anchor_id']}"
        )
        surface_anchor["candidate_id"] = candidate_id
        surface_anchor["candidate_family"] = candidate_family
        surface_anchor["cohort_id"] = surface_cohort_id or cohort_id
        surface_anchor["evidence_ref"] = (
            f"artifact://frontier/{surface_anchor['anchor_id']}.json"
        )
    renamed_anchor_ids = [
        surface_anchor["anchor_id"] for surface_anchor in surface["anchors"]
    ]
    surface["cells"][0]["anchor_ids"] = renamed_anchor_ids
    surface.pop("input_digest")
    surface["input_digest"] = _canonical_digest(surface)
    document["capability_surfaces"] = [surface]
    document["surface_queries"] = [
        {
            "query_id": f"query-{adapter_id}-exact-shape",
            "surface_id": surface_id,
            "surface_version": "v1",
            "shape": {"s": 128},
            "domain": surface["domain"],
        }
    ]
    document["measurement_adapter"] = {
        "adapter_id": adapter_id,
        "adapter_version": "v1",
        "protocol_id": "exact-shape-diagnostic",
        "protocol_version": "v1",
        "evidence_ref": f"artifact://adapter/{adapter_id}.json",
    }
    limited_statuses = {
        "timer.primary": "measured",
        "counter.cpu_cycles": "derived",
        "counter.vector_width": "declared",
        "counter.l2_cache_misses": "unsupported",
        "counter.energy_joules": "permission_denied",
        "trace.task_timeline": "not_requested",
        "timer.native_device": "not_applicable",
        "counter.instructions": "collection_failed",
        "counter.cache_occupancy": "unknown",
    }
    m4_statuses = {
        **limited_statuses,
        "counter.cpu_cycles": "measured",
        "counter.l2_cache_misses": "measured",
        "trace.task_timeline": "measured",
        "counter.instructions": "measured",
    }
    field_statuses = (
        limited_statuses if adapter_id == "capability-limited-test" else m4_statuses
    )
    available_values = {
        "timer.primary": 1,
        "counter.cpu_cycles": 8_100_000,
        "counter.vector_width": 128,
        "counter.l2_cache_misses": 42_000,
        "trace.task_timeline": "artifact://trace/tasks.json",
        "counter.instructions": 16_200_000,
    }
    fields = []
    for field_name, status in field_statuses.items():
        field = {
            "field": field_name,
            "status": status,
            "required_for_anchor": field_name == "timer.primary",
            "source": adapter_id,
            "scope": "exact-shape-operator",
            "attribution": "direct" if status == "measured" else "declared",
            "intrusion": "baseline" if field_name == "timer.primary" else "diagnostic",
        }
        if status in {"measured", "derived", "declared"}:
            field["value"] = available_values[field_name]
        if status == "derived":
            field["derivation"] = {
                "method": "cycles-from-reference-clock-and-duration",
                "input_evidence_refs": [
                    f"artifact://adapter/{adapter_id}-timer-primary.json",
                    f"artifact://adapter/{adapter_id}-clock-rate.json",
                ],
            }
        fields.append(field)
    document["measurement_capability_manifest"] = {
        "manifest_id": f"manifest-{adapter_id}-v1",
        "adapter_id": adapter_id,
        "cohort_id": cohort_id,
        "fields": fields,
        "evidence_ref": f"artifact://adapter/{adapter_id}-capabilities.json",
    }
    pair_id = f"lane-pair-{adapter_id}-q-proj-001"
    baseline = document["baseline_timing_lane"]
    baseline["lane_id"] = f"baseline-{adapter_id}-q-proj-001"
    baseline["pair_id"] = pair_id
    baseline["cohort_id"] = cohort_id
    baseline["candidate_id"] = candidate_id
    baseline["execution_domain"] = document["execution_domain"]
    document["diagnostic_profiling_lane"] = {
        "lane_id": f"diagnostic-{adapter_id}-q-proj-001",
        "pair_id": pair_id,
        "paired_baseline_lane_id": baseline["lane_id"],
        "cohort_id": cohort_id,
        "candidate_id": candidate_id,
        "execution_domain": document["execution_domain"],
        "instrumentation_profile": "diagnostic-counters/v1",
        "observation_validity": "COLLECTED",
        "frontier_role": "NONE",
        "completion_boundary": baseline["completion_boundary"],
        "timer": baseline["timer"],
        "raw_samples_ns": [100_000, 110_000, 120_000],
        "overhead_ablation": {"status": "not_provided"},
        "evidence_ref": f"artifact://observation/{adapter_id}-diagnostic.json",
    }
    if invalid_completion_protocol == "device_without_stream_sync":
        invalid_boundary = {
            "kind": "device-event-stream-completion",
            "closed": True,
            "threadpool_joined": True,
            "device_event_id": "event-001",
            "stream_id": "stream-001",
            "stream_synchronized": False,
        }
        baseline["completion_boundary"] = invalid_boundary
        anchor["completion_boundary"] = invalid_boundary
        document["diagnostic_profiling_lane"][
            "completion_boundary"
        ] = invalid_boundary
    elif invalid_completion_protocol == "device_host_timer":
        invalid_boundary = {
            "kind": "device-event-stream-completion",
            "closed": True,
            "threadpool_joined": True,
            "device_event_id": "event-001",
            "stream_id": "stream-001",
            "stream_synchronized": True,
        }
        baseline["completion_boundary"] = invalid_boundary
        anchor["completion_boundary"] = invalid_boundary
        document["diagnostic_profiling_lane"][
            "completion_boundary"
        ] = invalid_boundary
    elif invalid_completion_protocol == "cross_rank_absolute_clock":
        invalid_boundary = {
            "kind": "distributed-rank-local-duration",
            "closed": True,
            "threadpool_joined": True,
            "rank_local_durations": True,
            "duration_reducer": "max",
            "rank_duration_refs": ["rank://0/duration", "rank://1/duration"],
            "absolute_timestamps_subtracted": True,
        }
        baseline["completion_boundary"] = invalid_boundary
        anchor["completion_boundary"] = invalid_boundary
        document["diagnostic_profiling_lane"][
            "completion_boundary"
        ] = invalid_boundary
    elif invalid_completion_protocol == "cpu_wall_clock":
        for lane in (baseline, document["diagnostic_profiling_lane"], anchor):
            lane["timer"]["source"] = "wall-clock"
            lane["timer"]["monotonic"] = False
    if self_reported_profile_qualification:
        document["diagnostic_profiling_lane"]["overhead_ablation"] = {
            "status": "qualified",
            "within_error_budget": True,
            "instrumentation_profile": "diagnostic-counters/v1",
            "policy_ref": "policy://unverified-overhead/v1",
            "evidence_ref": "artifact://ablation/self-reported.json",
        }
        document["policies"]["profiling_overhead"] = {
            "policy_id": "profiling-overhead",
            "version": "v1",
            "scope": "exact-shape adapter evidence",
            "change_reason": "qualify one diagnostic timing mode",
            "revalidation": "on instrumentation or cohort change",
            "instrumentation_profiles": ["different-profile/v1"],
            "validity_domain_ref": "domain://profiling-overhead/v1",
            "maximum_overhead_ratio": 0.05,
            "minimum_independent_sessions": 2,
        }
    elif qualified_profile_ablation:
        document["diagnostic_profiling_lane"]["overhead_ablation"] = {
            "status": "qualified",
            "instrumentation_profile": "diagnostic-counters/v1",
            "selection": {
                "session_ids": ["selection-001", "selection-002"],
                "evidence_ref": "artifact://ablation/selection.json",
            },
            "holdout": {
                "pair_id": pair_id,
                "baseline_lane_id": baseline["lane_id"],
                "diagnostic_lane_id": document["diagnostic_profiling_lane"][
                    "lane_id"
                ],
                "baseline_session_ids": ["baseline-001", "baseline-002"],
                "diagnostic_session_ids": [
                    "diagnostic-001",
                    "diagnostic-002",
                ],
                "baseline_raw_samples_ns": [1_000_000, 1_010_000],
                "diagnostic_raw_samples_ns": [1_020_000, 1_030_000],
                "evidence_ref": "artifact://ablation/holdout.json",
            },
            "evidence_ref": "artifact://ablation/qualification.json",
        }
        document["policies"]["profiling_overhead"] = {
            "policy_id": "profiling-overhead",
            "version": "v1",
            "scope": "exact-shape adapter evidence",
            "change_reason": "qualify one diagnostic timing mode",
            "revalidation": "on instrumentation or cohort change",
            "instrumentation_profiles": ["diagnostic-counters/v1"],
            "validity_domain_ref": "domain://profiling-overhead/v1",
            "maximum_overhead_ratio": 0.05,
            "minimum_independent_sessions": 2,
        }
    if invalid_required_evidence == "hardware_identity":
        document["hardware"].pop("topology")
    elif invalid_required_evidence == "correctness":
        document["correctness"]["passed"] = False
    elif invalid_required_evidence == "completion_boundary":
        document["frontier_anchors"][0]["completion_boundary"]["closed"] = False
    elif invalid_required_evidence == "primary_timer":
        primary_timer = next(
            field
            for field in document["measurement_capability_manifest"]["fields"]
            if field["field"] == "timer.primary"
        )
        primary_timer["status"] = "unsupported"
        primary_timer.pop("value")

    reference_identity = {
        "device": document["hardware"].get("device"),
        "partition": document["hardware"].get("partition"),
        "topology": document["hardware"].get("topology"),
        "software": document["hardware"].get("software"),
        "power_clock": dict(document["hardware"]["power_clock"]),
        "numeric_execution": {
            key: document["execution_domain"][key]
            for key in (
                "dtype",
                "layout",
                "alignment_bytes",
                "threads",
                "execution_mode",
            )
        },
        "timer_protocol": {
            "source": baseline["timer"]["source"],
            "resolution_ns": baseline["timer"]["resolution_ns"],
            "monotonic": baseline["timer"]["monotonic"],
            "completion_kind": baseline["completion_boundary"]["kind"],
            "duration_reducer": baseline["completion_boundary"].get(
                "duration_reducer"
            ),
            "adapter_id": adapter_id,
            "adapter_version": "v1",
            "protocol_id": "exact-shape-diagnostic",
            "protocol_version": "v1",
        },
        "execution_context": {
            key: document["execution_domain"][key]
            for key in (
                "affinity",
                "numa",
                "context",
                "stream",
                "concurrency",
            )
        },
        "communication": document["communication_identity"],
    }
    if cohort_change in {"device", "partition", "topology", "software"}:
        document["hardware"][cohort_change] += "-changed"
    elif cohort_change == "numeric_execution":
        document["execution_domain"]["execution_mode"] = "compiled"
        anchor["execution_domain"] = document["execution_domain"]
        document["resource_physical_floor"]["resource_terms"][0][
            "execution_domain"
        ] = document["execution_domain"]
        baseline["execution_domain"] = document["execution_domain"]
        document["diagnostic_profiling_lane"]["execution_domain"] = document[
            "execution_domain"
        ]
        surface["domain"]["execution_mode"] = "compiled"
        for surface_anchor in surface["anchors"]:
            surface_anchor["domain"] = surface["domain"]
        document["surface_queries"][0]["domain"] = surface["domain"]
    elif cohort_change == "timer_protocol":
        for lane in (baseline, document["diagnostic_profiling_lane"], anchor):
            lane["timer"]["source"] = "test-clock-v2"
    elif cohort_change == "power_clock":
        document["hardware"]["power_clock"]["power_policy"] = "performance"
    elif cohort_change == "execution_context":
        document["execution_domain"]["affinity"] = "performance-cores-only"
        anchor["execution_domain"] = document["execution_domain"]
        document["resource_physical_floor"]["resource_terms"][0][
            "execution_domain"
        ] = document["execution_domain"]
        baseline["execution_domain"] = document["execution_domain"]
        document["diagnostic_profiling_lane"]["execution_domain"] = document[
            "execution_domain"
        ]
    elif cohort_change == "communication":
        document["communication_identity"] = {
            "status": "applicable",
            "rank_count": 2,
            "topology": "two-rank-local",
            "backend": "fixture-collective",
            "algorithm": "ring",
            "routing": "direct",
        }
    elif cohort_change == "duration_reducer":
        distributed_boundary = {
            "kind": "distributed-rank-local-duration",
            "closed": True,
            "rank_local_durations": True,
            "duration_reducer": "sum",
            "rank_duration_refs": ["rank://0/duration", "rank://1/duration"],
            "absolute_timestamps_subtracted": False,
        }
        rank_local_timer = {
            "source": "rank-local-monotonic",
            "resolution_ns": 1,
            "monotonic": True,
            "clock_domain": "rank-local",
        }
        for lane in (baseline, document["diagnostic_profiling_lane"], anchor):
            lane["completion_boundary"] = deepcopy(distributed_boundary)
            lane["timer"] = deepcopy(rank_local_timer)
        reference_identity["timer_protocol"].update(
            {
                "source": "rank-local-monotonic",
                "completion_kind": "distributed-rank-local-duration",
                "duration_reducer": "max",
            }
        )

    document["policies"]["cohort"] = {
        "policy_id": "hardware-validity-cohort",
        "version": "v1",
        "scope": "exact-shape adapter evidence",
        "change_reason": "make cohort split and retry explicit",
        "revalidation": "on stable identity or transient health change",
        "maximum_retry_attempts": 2,
    }
    document["cohort_evidence"] = {
        "reference_cohort_id": reference_cohort_id or cohort_id,
        "reference_identity": reference_identity,
        "transient_failures": (
            []
            if transient_failure is None
            else [
                {
                    "kind": transient_failure,
                    "retryable": True,
                    "evidence_ref": (
                        f"artifact://environment/{transient_failure}.json"
                    ),
                }
            ]
        ),
        "retry_attempt": 1,
        "evidence_ref": f"artifact://cohort/{adapter_id}.json",
    }
    recorded_adapter = _RecordedFixtureAdapter(document)
    assert isinstance(recorded_adapter, MeasurementAdapter)
    case = {
        "benchmark_case": document["resolved_configuration"]["benchmark_case"],
        "semantic_node": document["resolved_ir"]["semantic_node"],
        "execution_domain": document["execution_domain"],
    }
    document["measurement_capability_manifest"] = (
        recorded_adapter.discover_capabilities()
    )
    document["cohort_evidence"]["observed_identity"] = (
        recorded_adapter.fingerprint_cohort()
    )
    document["environment"] = recorded_adapter.preflight()
    timing_plan = recorded_adapter.build_timing_plan(case)
    collection = recorded_adapter.collect(case, timing_plan)
    document["timing_plan"] = timing_plan
    document["baseline_timing_lane"] = collection["baseline_timing_lane"]
    document["diagnostic_profiling_lane"] = collection[
        "diagnostic_profiling_lane"
    ]
    document["measurement_adapter"]["operation_evidence"] = (
        recorded_adapter.operations
    )
    if omit_capability_manifest:
        document.pop("measurement_capability_manifest")
    if omit_derived_provenance:
        derived = next(
            field
            for field in document["measurement_capability_manifest"]["fields"]
            if field["status"] == "derived"
        )
        derived.pop("derivation", None)
    if omit_cohort_evidence:
        document.pop("cohort_evidence")
    surface.pop("input_digest")
    surface["input_digest"] = _canonical_digest(surface)

    return refresh_diagnostic_bundle(
        run,
        document,
        run_id=f"run-{adapter_id}-exact-shape",
        hardware_cohort=cohort_id,
    )


def test_two_adapters_share_protocol_without_reusing_cohort_anchor_or_surface(
    tmp_path: Path,
) -> None:
    m4 = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path / "m4",
            adapter_id="apple-m4-cpu",
            device="Apple M4 CPU",
            cohort_id="cohort-apple-m4-cpu-v1",
            candidate_id="torch.matmul.cpu",
            candidate_family="pytorch-cpu-matmul",
        )
    )
    limited = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path / "limited",
            adapter_id="capability-limited-test",
            device="Capability-limited test device",
            cohort_id="cohort-capability-limited-v1",
            candidate_id="test.matmul.limited",
            candidate_family="test-limited-matmul",
        )
    )

    assert m4["evidence"]["execution_domain"] == limited["evidence"][
        "execution_domain"
    ]
    assert m4["adapter_contract"]["protocol"] == limited[
        "adapter_contract"
    ]["protocol"] == {
        "protocol_id": "exact-shape-diagnostic",
        "protocol_version": "v1",
    }
    assert [
        operation["operation"]
        for operation in limited["adapter_contract"]["operation_evidence"]
    ] == [
        "discover_capabilities",
        "fingerprint_cohort",
        "preflight",
        "build_timing_plan",
        "collect",
    ]
    assert {
        m4["adapter_contract"]["adapter_id"],
        limited["adapter_contract"]["adapter_id"],
    } == {"apple-m4-cpu", "capability-limited-test"}
    assert m4["evidence"]["cohort_id"] != limited["evidence"]["cohort_id"]
    assert m4["axes"]["operator_achievable_frontier"]["anchor_id"] != limited[
        "axes"
    ]["operator_achievable_frontier"]["anchor_id"]
    assert m4["capability_surfaces"][0]["surface_id"] != limited[
        "capability_surfaces"
    ][0]["surface_id"]
    assert m4["capability_surface_queries"][0]["cohort_id"] == m4[
        "evidence"
    ]["cohort_id"]
    assert limited["capability_surface_queries"][0]["cohort_id"] == limited[
        "evidence"
    ]["cohort_id"]


def test_capability_manifest_preserves_standard_field_statuses_without_zero_fill(
    tmp_path: Path,
) -> None:
    m4 = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path / "m4",
            adapter_id="apple-m4-cpu",
            device="Apple M4 CPU",
            cohort_id="cohort-apple-m4-cpu-v1",
            candidate_id="torch.matmul.cpu",
            candidate_family="pytorch-cpu-matmul",
        )
    )
    limited = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path / "limited",
            adapter_id="capability-limited-test",
            device="Capability-limited test device",
            cohort_id="cohort-capability-limited-v1",
            candidate_id="test.matmul.limited",
            candidate_family="test-limited-matmul",
        )
    )

    m4_fields = {
        field["field"]: field
        for field in m4["adapter_contract"]["observation_fields"]
    }
    limited_fields = {
        field["field"]: field
        for field in limited["adapter_contract"]["observation_fields"]
    }
    assert set(m4_fields) == set(limited_fields)
    assert {field["status"] for field in limited_fields.values()} == {
        "measured",
        "derived",
        "declared",
        "unsupported",
        "permission_denied",
        "not_requested",
        "not_applicable",
        "collection_failed",
        "unknown",
    }
    unavailable = {
        "unsupported",
        "permission_denied",
        "not_requested",
        "not_applicable",
        "collection_failed",
        "unknown",
    }
    assert all(
        "value" not in field
        for field in limited_fields.values()
        if field["status"] in unavailable
    )


def test_missing_required_evidence_blocks_anchor_with_insufficient_evidence(
    tmp_path: Path,
) -> None:
    expected_reasons = {
        "hardware_identity": "incomplete-required-identity",
        "correctness": "correctness-not-qualified",
        "completion_boundary": "incomplete-completion-boundary",
        "primary_timer": "missing-primary-timer",
    }

    for invalid_required_evidence, reason_code in expected_reasons.items():
        result = diagnose_run_bundle(
            _rewrite_bundle(
                tmp_path / invalid_required_evidence,
                adapter_id="capability-limited-test",
                device="Capability-limited test device",
                cohort_id="cohort-capability-limited-v1",
                candidate_id="test.matmul.limited",
                candidate_family="test-limited-matmul",
                invalid_required_evidence=invalid_required_evidence,
            )
        )

        assert result["adapter_contract"]["anchor_admission"] == {
            "status": "insufficient_evidence",
            "reason_codes": [reason_code],
        }
        assert result["axes"]["operator_achievable_frontier"]["status"] == (
            "unknown"
        )


def test_invalid_device_and_distributed_completion_protocols_block_anchor(
    tmp_path: Path,
) -> None:
    expected_reasons = {
        "device_without_stream_sync": "incomplete-completion-boundary",
        "device_host_timer": "invalid-primary-timer-protocol",
        "cross_rank_absolute_clock": "incomplete-completion-boundary",
        "cpu_wall_clock": "invalid-primary-timer-protocol",
    }
    for invalid_completion_protocol, reason_code in expected_reasons.items():
        result = diagnose_run_bundle(
            _rewrite_bundle(
                tmp_path / invalid_completion_protocol,
                adapter_id="capability-limited-test",
                device="Capability-limited test device",
                cohort_id="cohort-capability-limited-v1",
                candidate_id="test.matmul.limited",
                candidate_family="test-limited-matmul",
                invalid_completion_protocol=invalid_completion_protocol,
            )
        )

        assert result["adapter_contract"]["anchor_admission"] == {
            "status": "insufficient_evidence",
            "reason_codes": [reason_code],
        }
        assert result["axes"]["operator_achievable_frontier"]["status"] == (
            "unknown"
        )
        if invalid_completion_protocol == "device_host_timer":
            report = render_diagnostic_report(result)
            assert (
                "baseline completion: "
                "kind=device-event-stream-completion; "
                "timer=mach-continuous-time; stream=stream-001"
            ) in report


def test_missing_capability_manifest_cannot_leave_anchor_qualified(
    tmp_path: Path,
) -> None:
    result = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path,
            adapter_id="capability-limited-test",
            device="Capability-limited test device",
            cohort_id="cohort-capability-limited-v1",
            candidate_id="test.matmul.limited",
            candidate_family="test-limited-matmul",
            omit_capability_manifest=True,
        )
    )

    assert result["adapter_contract"] == {
        "status": "insufficient_evidence",
        "reason_codes": ["invalid-measurement-capability-manifest"],
    }
    assert result["axes"]["operator_achievable_frontier"] == {
        "status": "unknown",
        "reason_code": "no-qualified-active-exact-shape-anchor",
        "evidence_refs": [],
    }


def test_missing_cohort_evidence_cannot_leave_anchor_qualified(
    tmp_path: Path,
) -> None:
    result = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path,
            adapter_id="capability-limited-test",
            device="Capability-limited test device",
            cohort_id="cohort-capability-limited-v1",
            candidate_id="test.matmul.limited",
            candidate_family="test-limited-matmul",
            omit_cohort_evidence=True,
        )
    )

    assert result["adapter_contract"]["anchor_admission"] == {
        "status": "insufficient_evidence",
        "reason_codes": ["invalid-cohort-evidence"],
    }
    assert result["axes"]["operator_achievable_frontier"]["status"] == (
        "unknown"
    )


def test_derived_observation_requires_derivation_input_provenance(
    tmp_path: Path,
) -> None:
    result = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path,
            adapter_id="capability-limited-test",
            device="Capability-limited test device",
            cohort_id="cohort-capability-limited-v1",
            candidate_id="test.matmul.limited",
            candidate_family="test-limited-matmul",
            omit_derived_provenance=True,
        )
    )

    assert result["adapter_contract"] == {
        "status": "insufficient_evidence",
        "reason_codes": ["invalid-observation-field"],
    }
    assert result["axes"]["operator_achievable_frontier"]["status"] == (
        "unknown"
    )


def test_surface_from_another_cohort_cannot_be_reused_by_adapter(
    tmp_path: Path,
) -> None:
    result = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path,
            adapter_id="capability-limited-test",
            device="Capability-limited test device",
            cohort_id="cohort-capability-limited-v1",
            candidate_id="test.matmul.limited",
            candidate_family="test-limited-matmul",
            surface_cohort_id="cohort-foreign-v1",
        )
    )

    assert result["capability_surface_queries"][0]["status"] == "unknown"
    assert result["capability_surface_queries"][0]["reason_code"] == (
        "surface_cohort_mismatch"
    )


def test_paired_baseline_and_diagnostic_lanes_remain_independent_without_ablation(
    tmp_path: Path,
) -> None:
    result = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path,
            adapter_id="capability-limited-test",
            device="Capability-limited test device",
            cohort_id="cohort-capability-limited-v1",
            candidate_id="test.matmul.limited",
            candidate_family="test-limited-matmul",
        )
    )

    assert result["adapter_contract"]["lanes"] == {
        "pair_id": "lane-pair-capability-limited-test-q-proj-001",
        "baseline_lane_id": "baseline-capability-limited-test-q-proj-001",
        "diagnostic_lane_id": (
            "diagnostic-capability-limited-test-q-proj-001"
        ),
        "diagnostic_frontier_eligible": False,
        "reason_code": "profiling-overhead-ablation-missing",
    }
    assert result["evidence"]["baseline_timing_lane"]["raw_samples_ns"] == [
        1_560_000,
        1_600_000,
        1_640_000,
    ]
    assert result["evidence"]["diagnostic_profiling_lane"][
        "raw_samples_ns"
    ] == [100_000, 110_000, 120_000]
    assert result["axes"]["operator_achievable_frontier"]["value_ns"] == (
        1_200_000
    )
    report = render_diagnostic_report(result)
    assert "lane-pair-capability-limited-test-q-proj-001" in report
    assert "profiling-overhead-ablation-missing" in report


def test_self_reported_profiling_ablation_cannot_promote_diagnostic_timing(
    tmp_path: Path,
) -> None:
    result = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path,
            adapter_id="capability-limited-test",
            device="Capability-limited test device",
            cohort_id="cohort-capability-limited-v1",
            candidate_id="test.matmul.limited",
            candidate_family="test-limited-matmul",
            self_reported_profile_qualification=True,
        )
    )

    assert result["adapter_contract"]["lanes"][
        "diagnostic_frontier_eligible"
    ] is False
    assert result["adapter_contract"]["lanes"]["reason_code"] == (
        "profiling-overhead-ablation-unqualified"
    )


def test_independent_in_domain_profiling_ablation_can_qualify_its_timing(
    tmp_path: Path,
) -> None:
    result = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path,
            adapter_id="capability-limited-test",
            device="Capability-limited test device",
            cohort_id="cohort-capability-limited-v1",
            candidate_id="test.matmul.limited",
            candidate_family="test-limited-matmul",
            qualified_profile_ablation=True,
        )
    )

    assert result["adapter_contract"]["lanes"][
        "diagnostic_frontier_eligible"
    ] is True
    assert result["adapter_contract"]["lanes"]["reason_code"] is None
    assert result["axes"]["operator_achievable_frontier"]["value_ns"] == (
        1_200_000
    )


def test_stable_identity_changes_split_hardware_cohort_without_frontier_reuse(
    tmp_path: Path,
) -> None:
    for changed_dimension in (
        "device",
        "partition",
        "topology",
        "software",
        "numeric_execution",
        "timer_protocol",
        "power_clock",
        "execution_context",
        "communication",
    ):
        result = diagnose_run_bundle(
            _rewrite_bundle(
                tmp_path / changed_dimension,
                adapter_id="capability-limited-test",
                device="Capability-limited test device",
                cohort_id=f"cohort-capability-limited-{changed_dimension}-v2",
                reference_cohort_id="cohort-capability-limited-v1",
                candidate_id="test.matmul.limited",
                candidate_family="test-limited-matmul",
                cohort_change=changed_dimension,
            )
        )

        assert result["adapter_contract"]["cohort"] == {
            "status": "split",
            "cohort_id": f"cohort-capability-limited-{changed_dimension}-v2",
            "reference_cohort_id": "cohort-capability-limited-v1",
            "changed_dimensions": [changed_dimension],
            "retry": {"status": "not_required"},
            "evidence_refs": [
                "artifact://cohort/capability-limited-test.json"
            ],
        }
        assert result["adapter_contract"]["anchor_admission"]["status"] == (
            "eligible"
        )
        assert result["capability_surface_queries"][0]["cohort_id"] == result[
            "evidence"
        ]["cohort_id"]


def test_distributed_duration_reducer_change_splits_hardware_cohort(
    tmp_path: Path,
) -> None:
    result = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path,
            adapter_id="capability-limited-test",
            device="Capability-limited test device",
            cohort_id="cohort-capability-limited-reducer-v2",
            reference_cohort_id="cohort-capability-limited-v1",
            candidate_id="test.matmul.limited",
            candidate_family="test-limited-matmul",
            cohort_change="duration_reducer",
        )
    )

    assert result["adapter_contract"]["cohort"]["status"] == "split"
    assert result["adapter_contract"]["cohort"]["changed_dimensions"] == [
        "timer_protocol"
    ]
    assert result["adapter_contract"]["anchor_admission"]["status"] == (
        "eligible"
    )
    report = render_diagnostic_report(result)
    assert (
        "baseline completion: kind=distributed-rank-local-duration; "
        "timer=rank-local-monotonic; reducer=sum"
    ) in report
    assert (
        "diagnostic completion: kind=distributed-rank-local-duration; "
        "timer=rank-local-monotonic; reducer=sum"
    ) in report


def test_transient_health_failure_is_quarantined_and_retried_in_same_cohort(
    tmp_path: Path,
) -> None:
    result = diagnose_run_bundle(
        _rewrite_bundle(
            tmp_path,
            adapter_id="capability-limited-test",
            device="Capability-limited test device",
            cohort_id="cohort-capability-limited-v1",
            candidate_id="test.matmul.limited",
            candidate_family="test-limited-matmul",
            transient_failure="health",
        )
    )

    assert result["adapter_contract"]["cohort"] == {
        "status": "quarantined",
        "cohort_id": "cohort-capability-limited-v1",
        "reference_cohort_id": "cohort-capability-limited-v1",
        "changed_dimensions": [],
        "transient_failures": ["health"],
        "retry": {
            "status": "required",
            "attempt": 1,
            "maximum_attempts": 2,
            "policy_ref": "hardware-validity-cohort/v1",
        },
        "evidence_refs": [
            "artifact://cohort/capability-limited-test.json",
            "artifact://environment/health.json",
        ],
    }
    assert result["adapter_contract"]["anchor_admission"] == {
        "status": "insufficient_evidence",
        "reason_codes": ["cohort-quarantined"],
    }
    assert result["axes"]["operator_achievable_frontier"]["status"] == "unknown"


def test_cli_and_report_project_the_same_adapter_contract(tmp_path: Path) -> None:
    run = _rewrite_bundle(
        tmp_path,
        adapter_id="capability-limited-test",
        device="Capability-limited test device",
        cohort_id="cohort-capability-limited-v1",
        candidate_id="test.matmul.limited",
        candidate_family="test-limited-matmul",
    )
    direct = diagnose_run_bundle(run)

    machine = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    human = subprocess.run(
        [
            sys.executable,
            "-m",
            "groundupscale.cli",
            "diagnose",
            str(run),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert machine.returncode == 0, machine.stderr
    assert json.loads(machine.stdout) == direct
    assert human.returncode == 0, human.stderr
    assert (
        "measurement adapter: capability-limited-test@v1; "
        "protocol=exact-shape-diagnostic@v1; status=eligible"
    ) in human.stdout
    assert (
        "hardware validity cohort: matched; "
        "current=cohort-capability-limited-v1; "
        "reference=cohort-capability-limited-v1; changes=none; "
        "retry=not_required"
    ) in human.stdout
    assert "anchor admission: eligible" in human.stdout
    assert "counter.l2_cache_misses=unsupported" in human.stdout
    assert (
        "baseline completion: kind=synchronous-cpu-call-return; "
        "timer=mach-continuous-time"
    ) in human.stdout
    assert (
        "diagnostic completion: kind=synchronous-cpu-call-return; "
        "timer=mach-continuous-time"
    ) in human.stdout
