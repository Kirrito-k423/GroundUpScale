# Use immutable observations and controlled versioned calibration

Real executions produce immutable `ObservationTrace`s that enter versioned,
quality-checked datasets. Reproducible `CalibrationRun`s fit candidate profiles
and validate them against independent evidence and explicit error budgets before
promotion; profiles never overwrite base formulas, every calibrated prediction
retains its base value and evidence, out-of-domain use falls back explicitly,
and all profiles remain comparable and reversible.
