# Compose models in YAML and add primitive semantics through plugins

Users define Composite Modules entirely in YAML by composing registered module
kinds, parameter bindings, Model Repeats, and Structural Overrides. A new
Primitive Module is introduced only through a plugin that supplies its typed
semantic Lowering. This keeps ordinary model integration code-free without
allowing executable callbacks or hidden construction behavior inside YAML.
