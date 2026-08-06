# Compose single-Spec YAML documents by explicit reference

Each YAML file represents one Spec Document and composes reusable inputs through
explicit Spec References rather than cross-file YAML anchors or implicit global
lookup. References may resolve to repository-relative files, plugin resources,
or supported URIs, must form an acyclic finite graph, and are pinned by version
or content digest whenever CI or reproducibility is required. Web tooling may
present an aggregated view but persists the same referenced YAML documents.
