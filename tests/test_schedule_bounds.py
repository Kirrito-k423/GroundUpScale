from __future__ import annotations

import pytest

from groundupscale.scheduling import (
    BoundEvent,
    ConcurrencyGroup,
    ResourcePath,
    compose_schedule_bound,
)


def test_serialized_schedule_adds_local_event_floors() -> None:
    bound = compose_schedule_bound(
        (
            BoundEvent(
                event_id="a",
                predecessor_ids=(),
                local_duration_ns=3.0,
                resource_times_ns=(("compute", 3.0),),
            ),
            BoundEvent(
                event_id="b",
                predecessor_ids=("a",),
                local_duration_ns=5.0,
                resource_times_ns=(("memory", 5.0),),
            ),
        ),
        schedule="serialized",
        resource_paths=(
            ResourcePath("compute-path", "compute", ("a",), ("test://a",)),
            ResourcePath("memory-path", "memory", ("b",), ("test://b",)),
        ),
    )

    assert bound.serialized_duration_ns == pytest.approx(8.0)
    assert bound.critical_path_duration_ns == pytest.approx(8.0)
    assert bound.resource_duration_ns == pytest.approx(5.0)
    assert bound.ideal_dag_duration_ns == pytest.approx(8.0)
    assert bound.selected_duration_ns == pytest.approx(8.0)


def test_dependency_only_schedule_uses_critical_path_and_resource_load() -> None:
    bound = compose_schedule_bound(
        (
            BoundEvent("root", (), 2.0, (("compute", 2.0),)),
            BoundEvent("left", ("root",), 5.0, (("compute", 5.0),)),
            BoundEvent("right", ("root",), 4.0, (("memory", 4.0),)),
        ),
        schedule="dependency-only",
        concurrency_groups=(
            ConcurrencyGroup(
                "declared-left-right",
                ("left", "right"),
                ("test://declared-left-right",),
            ),
        ),
        resource_paths=(
            ResourcePath(
                "compute-path", "compute", ("root", "left"), ("test://compute",)
            ),
            ResourcePath(
                "memory-path", "memory", ("right",), ("test://memory",)
            ),
        ),
    )

    assert bound.serialized_duration_ns == pytest.approx(11.0)
    assert bound.critical_path_duration_ns == pytest.approx(7.0)
    assert bound.resource_duration_ns == pytest.approx(7.0)
    assert bound.ideal_dag_duration_ns == pytest.approx(7.0)
    assert bound.selected_duration_ns == pytest.approx(7.0)
    assert bound.limiting_resource == "compute"


def test_schedule_dependencies_must_be_acyclic() -> None:
    events = (
        BoundEvent("a", ("b",), 1.0),
        BoundEvent("b", ("a",), 1.0),
    )

    with pytest.raises(ValueError, match="cycle"):
        compose_schedule_bound(events, schedule="serialized")


def test_dependency_only_schedule_rejects_implicit_concurrency() -> None:
    events = (
        BoundEvent("root", (), 2.0),
        BoundEvent("left", ("root",), 5.0),
        BoundEvent("right", ("root",), 4.0),
    )

    with pytest.raises(ValueError, match="explicit concurrency"):
        compose_schedule_bound(events, schedule="dependency-only")


def test_resource_load_requires_an_explicit_resource_path() -> None:
    events = (
        BoundEvent("a", (), 2.0, (("compute", 2.0),)),
        BoundEvent("b", ("a",), 3.0, (("compute", 3.0),)),
    )

    with pytest.raises(ValueError, match="explicit resource path"):
        compose_schedule_bound(events, schedule="dependency-only")
