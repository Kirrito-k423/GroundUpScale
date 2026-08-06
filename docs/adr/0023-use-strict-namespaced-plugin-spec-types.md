# Use strict namespaced plugin Spec types

Every Spec Document uses the same versioned envelope, and plugins extend its
typed fields through registered, namespaced, versioned Plugin Spec Types. Core
and plugin Schemas reject unknown fields, while each plugin owns its parameter
Schema, defaults, compatibility rules, and Lowering version; YAML selects and
configures the plugin but does not contain implementation code. This prevents
silent typo acceptance and avoids an untyped extension dictionary or a separate
Web-only data model.
