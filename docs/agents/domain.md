# Domain Docs

Before exploring or modifying the project, read:

- `CONTEXT.md`, when it exists
- Relevant ADRs under `docs/adr/`

This is a single-context repository.

`CONTEXT.md` defines the project’s shared domain vocabulary. Use its canonical
terms in code, tests, issues and documentation. Avoid synonyms explicitly
rejected by the glossary.

Domain documents are created lazily by `domain-modeling`, `grill-with-docs`
and `improve-codebase-architecture`; their initial absence is not an error.

If proposed work contradicts an existing ADR, surface the conflict explicitly
instead of silently overriding the decision.
