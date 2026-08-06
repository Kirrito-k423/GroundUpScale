# Organize artifacts as immutable self-describing Run Bundles

GroundUpScale separates version-controlled YAML Specs, generated local or CI Run
Bundles, promoted evidence and Calibration Profiles, and small test fixtures.
Each Run Bundle is immutable and indexed by a Run Manifest containing artifact
roles, Schema versions, hashes, lineage, cohort, and validity; paths are storage
locations rather than identities. Machine artifacts use formats suited to their
access pattern, while the same manifest contract supports local files, CI
artifacts, and future object-store adapters.
