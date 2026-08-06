# Use YAML as the only human-authored Spec format

GroundUpScale exposes one versioned, Schema-validated YAML format for every
human-authored Spec kind. Python is reserved for plugins and Lowering logic,
not a parallel Spec Builder, and Web tooling edits and emits the same YAML
documents instead of maintaining a separate JSON authoring contract;
intermediate IR serialization remains a separate inspection and interchange
concern.
