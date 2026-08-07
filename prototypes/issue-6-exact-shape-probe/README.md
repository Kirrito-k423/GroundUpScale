# PROTOTYPE — Exact-Shape Disambiguation Probe

This directory is a disposable logic prototype for the question in
“原型：Exact-Shape Disambiguation Probe 区分曲面失准与实现空间”:

> Can an exact-contract candidate probe distinguish implementation headroom
> and integration overhead while refusing a frontier shift when candidate or
> environment evidence is incomplete?

It is evidence, not a production runner. Nothing in this directory is imported
by `groundupscale`.

## Locked falsification protocol

- Primary anomaly: contiguous FP32 `257 x 257 x 257` MatMul, one CPU thread.
  The checked-in M4 observation reports 1.134376 TFLOP/s versus 1.394935
  TFLOP/s at the adjacent aligned `256^3` case (an 18.68% drop).
- Candidate contract: same semantic Shape, dtype, contiguous input/output
  layout, thread count, seed and hardware cohort. Padding is internal and its
  input/output copies remain inside the timed boundary.
- Correctness: FP64 NumPy reference with `atol=rtol=1e-4`, outside timing.
- Timing: three fresh processes, ten warmups, twelve randomized/interleaved
  windows, monotonic host clock, synchronous CPU completion.
- `implementation_headroom`: every session has a faster correct alternative;
  aggregate gain exceeds `max(5%, target IQR/median + candidate IQR/median)`;
  best-of-correct recovers at least 90% of the old `256^3` exact-anchor rate.
- `integration_overhead`: the standalone `256^3` operator remains within 10%
  of the old exact anchor; MatMul plus two explicit copies is slower by at least
  `max(10%, operator IQR/median + E2E IQR/median)`; a two-copy-only ablation
  explains the excess within 35%.
- `frontier_shift` is forbidden without C2/C3 candidate families, three
  sessions, stable neighbours and an eligible environment. PyTorch and NumPy
  both resolving to Accelerate count as C1 wrappers, not independent families.
- Removing alternatives, an ineligible environment, or a failed correctness
  oracle must produce `insufficient_evidence`.

The first locked run retained the `257^3` case as an
`insufficient_evidence` counterexample: no correct alternative beat the target.
Before any second benchmark, a separate implementation-headroom case was
locked from the repository's existing Context MatMul path:

- target: `einsum("bhqk,bkhd->bqhd").contiguous()`;
- alternative: a zero-copy values transpose, batched MatMul, output transpose,
  and contiguous result;
- exact external contract: FP32 probabilities `[1,8,512,512]`, values
  `[1,512,8,64]`, contiguous output `[1,512,8,64]`, four threads;
- verdict only if the correct alternative is faster in all three sessions and
  its aggregate gain exceeds `max(5%, combined IQR/median)`.

## Run

From the repository root:

```bash
python3.11 prototypes/issue-6-exact-shape-probe/run.py --batch
```

If the bare Python lacks project dependencies, the script re-executes the
repository's existing `.venv/bin/python`; it never installs packages. The run
prints each scenario's full decision state and writes raw evidence to
`results/raw-results.json`. It exits zero only when the locked assertions and
ticket Exit criteria pass.

Run without `--batch` for the tiny interactive evidence viewer.
