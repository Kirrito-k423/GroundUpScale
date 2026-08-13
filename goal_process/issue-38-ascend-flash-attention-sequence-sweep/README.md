# Issue #38 — bounded real Ascend TND FlashAttention sequence sweep

This directory contains the locked policy, resumable collection entrypoint,
qualification publisher, and the structured result for GitHub issue #38.

## Outcome

The bounded run currently publishes `unknown` with reason
`bounded-collection-stability-failed`; it does not publish a Capability
Surface or a global 4K Shape Regime Boundary.

On Hardware Cohort `ascend-npu-23b93a89d5fecc79` (machine `A2-AK-225`, host
`192.168.9.225`, actual login user `root`, remote hostname
`localhost.localdomain`), the real collection produced:

- main: 99 verified Run Bundles, all 33 declared sequence lengths times three
  independent process sessions;
- holdout: 99 verified Run Bundles, the same declared shapes and session count;
- independent validation: all 36 declared Run Bundles;
- supplemental: 0 Run Bundles; the main sweep did not meet the reviewed trigger.

Every observed main Run Bundle passed correctness, timing-quality, and its own
Run Bundle verification, with all 100 raw timing samples preserved. The
4095/4096/4097 medians did not establish a hard 4K boundary, so 4K remains only
a hypothesis for this fixed execution domain.

The UniVPN-backed SSH path temporarily accepted TCP/22 but closed before SSH
key exchange. After VPN recovery, the same process completed the remaining
validation bundles on the same Hardware Cohort.

The corpus still did not pass the 10% independent-session median-relative-range
gate. Holdout S=8 and S=255, plus validation S=48, S=160 and S=640, ranged from
10.43% to 15.78%. All Run Bundles themselves passed correctness and timing
quality; instability across independent sessions is what prevents promotion.
The bounded experiment therefore publishes `unknown` and stops, without
selectively dropping samples or adding another round.

## Replay and recovery

The remote workspace is:

```text
/home/t00906153/GroundUpScale-issue38-20260813
```

When the same machine is reachable, resume only the missing validation bundle
IDs. The collector verifies and skips every existing immutable Run Bundle:

```bash
cd /home/t00906153/GroundUpScale-issue38-20260813
bash goal_process/issue-38-ascend-flash-attention-sequence-sweep/collect_bounded_sequence_sweep.sh validation
```

Then publish through the public `OperatorFrontierBundleWriter` seam:

```bash
python goal_process/issue-38-ascend-flash-attention-sequence-sweep/publish_qualification.py \
  --workspace /home/t00906153/GroundUpScale-issue38-20260813
```

With an incomplete or unstable corpus this publisher emits a self-verifying structured unknown;
with all three complete lanes it evaluates the declared Error Budget and emits
either a qualified Surface or a rejected qualification. It never starts a
third experimental round.

## Evidence

- `evidence/qualification-unknown.json`: bounded stopping decision, confirmed
  inventory, failed stability gates, representative unknown queries, and exact
  replay commands.
- `evidence/runs/`: all 234 immutable measurement Run Bundles and the final
  self-contained `issue38-ascend-flash-attention-sequence-sweep-v2`
  qualification Run Bundle. From the repository root, `python -m
  groundupscale.cli verify-run <bundle> --json` passes without remote access.
- `RMB-Cost.md`: cache-hit, cache-miss, and output token cost estimate.
- `profilecodex-20260813-1303.md`: required >20-minute session latency profile.
