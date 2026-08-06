# Separate definition, instance, and compilation identities

GroundUpScale distinguishes a reusable Definition ID, its user-facing Stable
Path at each instantiated model or workload location, and the immutable Node ID
of each entity in one compilation. Mapping keys and explicit list IDs provide
local stable names, while Model Repeat uses a deterministic key template. This
allows provenance to explain both where behavior was defined and where it was
instantiated without exposing random or array-position identities to users.
