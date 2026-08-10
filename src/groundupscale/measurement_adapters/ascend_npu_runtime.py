"""Versioned runtime contract for trusted Ascend NPU measurement."""

from __future__ import annotations

import re


ASCEND_NPU_RUNTIME_CONTRACT = {
    "contract_id": "ascend-npu-runtime-v1",
    "environment_kind": "trusted-hardware-calibration",
    "requirements": {
        "python_major_minor": "3.11",
        "torch_major_minor": "2.7",
        "torch_npu_major_minor": "2.7",
        "cann_major_minor": "8.5",
    },
    "separation_policy": (
        "pyproject torch pin governs portable compiler CI; this contract "
        "governs trusted Ascend measurement"
    ),
}


def _major_minor(version: object) -> str | None:
    match = re.match(r"^(\d+)\.(\d+)", str(version))
    if match is None:
        return None
    return f"{match.group(1)}.{match.group(2)}"


def assess_ascend_npu_runtime(
    *,
    python_version: object,
    torch_version: object,
    torch_npu_version: object,
    cann_version: object,
) -> dict[str, object]:
    observed = {
        "python": str(python_version),
        "torch": str(torch_version),
        "torch_npu": str(torch_npu_version),
        "cann": str(cann_version),
    }
    observed_major_minor = {
        name: _major_minor(value) for name, value in observed.items()
    }
    requirements = dict(ASCEND_NPU_RUNTIME_CONTRACT["requirements"])
    compatible = (
        observed_major_minor["python"] == requirements["python_major_minor"]
        and observed_major_minor["torch"] == requirements["torch_major_minor"]
        and observed_major_minor["torch_npu"]
        == requirements["torch_npu_major_minor"]
        and observed_major_minor["cann"] == requirements["cann_major_minor"]
    )
    return {
        "status": "compatible" if compatible else "incompatible",
        "contract_id": ASCEND_NPU_RUNTIME_CONTRACT["contract_id"],
        "environment_kind": ASCEND_NPU_RUNTIME_CONTRACT["environment_kind"],
        "requirements": requirements,
        "observed": observed,
        "rule": (
            "all observed major.minor versions equal the declared trusted "
            "NPU contract"
        ),
        "separation_policy": ASCEND_NPU_RUNTIME_CONTRACT[
            "separation_policy"
        ],
    }


__all__ = ["ASCEND_NPU_RUNTIME_CONTRACT", "assess_ascend_npu_runtime"]
