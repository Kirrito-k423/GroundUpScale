# Separate public compiler CI from trusted hardware calibration

GroundUpScale runs deterministic Schema, compiler, invariant, and formula tests
on ordinary public CI, while real CPU and GPU observations run only in trusted,
hardware-specific measurement environments. Immutable Observation Traces are
compared within a Hardware Cohort and may produce a Candidate Calibration
Profile, but never update an active profile silently; promotion requires
independent validation, Error Budgets, and explicit approval. A personal Mac is
not exposed as a general self-hosted runner for untrusted public pull requests.
