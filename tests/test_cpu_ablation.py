from __future__ import annotations

import pytest

from groundupscale.benchmark.cpu_ablation import fit_affine_latency


def test_affine_latency_fit_recovers_fixed_intercept_and_rate() -> None:
    points = [
        {"work_flops": float(work), "median_ns": 25.0 + 0.5 * work}
        for work in (10, 20, 40, 80, 160, 320)
    ]

    fit = fit_affine_latency(points)

    assert fit["intercept_ns"] == pytest.approx(25.0)
    assert fit["slope_ns_per_flop"] == pytest.approx(0.5)
    assert fit["rate_flops_per_second"] == pytest.approx(2_000_000_000.0)
    assert fit["r_squared"] == pytest.approx(1.0)
