# Separate hardware specification, topology, backend, and calibration

GroundUpScale separates reusable static `HardwareSpec`s, concrete instance and
link topology in `FabricGraph`, implementation rules in `HardwareBackend`, and
versioned measured evidence in `CalibrationProfile`. Raw executions produce
immutable `ObservationTrace`s; theoretical capability, installed topology,
implementation selection, and environment-specific measured efficiency never
silently overwrite one another.
