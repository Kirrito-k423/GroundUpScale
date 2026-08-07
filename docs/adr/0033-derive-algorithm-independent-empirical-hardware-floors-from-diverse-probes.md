# Derive algorithm-independent empirical hardware floors from diverse probes

GroundUpScale treats scalar, vector, matrix, reduction, memory, and
communication microbenchmarks as **probes of physical resource capacity**, not
as definitions of the probed operator or kernel's performance. Each probe runs
at least ten stratified Shapes for its Hardware Cohort, covering aligned and
non-aligned boundaries such as `511/512/513`, multiple working-set domains,
data types, thread counts, and access patterns. Repetitions are first reduced
to a robust per-Shape statistic; the distribution across eligible saturating
Shapes then produces a P80 robust-achievable capacity and a separately reported
P95 or stable-maximum optimistic capacity. Probe, Shape, algorithm, library,
compiler, and Run Bundle identities remain provenance, but the promoted
capability keys are hardware resource classes such as `scalar_fp32`,
`vector_fp32`, `matrix_fp32`, `dram_sequential_read`, or a concrete Fabric link.
No operator or current implementation name becomes the identity of the
resulting `HardwareCapabilityEnvelope`.

The empirical hardware floor is derived from algorithm-independent minimum
demands, not from the selected kernel's traffic or efficiency. Cost IR must
therefore distinguish minimum mathematical work, compulsory input/state/output
bytes, and minimum communication cut volume from implementation-added FLOPs,
materializations, conversions, recomputation, workspaces, and messages. For an
eligible execution horizon, GroundUpScale computes resource terms such as
`T_compute = minimum_work / measured_hardware_capacity`,
`T_memory = compulsory_bytes / measured_hardware_bandwidth`, and
`T_communication = minimum_volume / measured_link_bandwidth`. Their maximum is
the hardware floor only when those resources may overlap; shared-resource and
dependency constraints are composed through typed Resource Claims and the
critical path. This floor is independent of the currently selected operator
algorithm, while `ImplementationCandidate` duration models separately account
for Shape efficiency, fusion, tiling, materialization, dispatch, synchronization,
and scheduling.

GroundUpScale preserves four non-overwriting result layers: the vendor-theory
floor from comparable published physical peaks; the empirical hardware floor
from a promoted `HardwareCapabilityEnvelope`; the implementation and scheduled
prediction; and the immutable observation. P80 is labeled a robust achievable
reference rather than a mathematical absolute peak because some valid probes
may exceed it; P95 or a repeatable stable maximum supplies the more optimistic
empirical boundary. Every promoted envelope retains its raw probe evidence,
Shape coverage, statistics, environment and software fingerprints, uncertainty,
and validity domain. CI rejects missing aligned/non-aligned coverage, unstable
per-Shape measurements, incompatible cohort aggregation, and capability drift
beyond the configured 10% budget; comparison with a vendor figure is enforced
only when operation, dtype, resource, concurrency, and unit semantics match.
Measured envelopes never overwrite `HardwareSpec` vendor facts or silently
absorb a poor current algorithm into the definition of hardware capability.
