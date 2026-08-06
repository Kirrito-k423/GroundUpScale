# Resolve bindings by specificity and reject ambiguity

`DeploymentIntent` bindings inherit through scope containment, and a strictly
more-specific descendant may refine an inherited field. Independent fields
compose, placement constraints intersect, and overlapping bindings that are
not ordered by containment must not assign incompatible values to the same
field; GroundUpScale rejects such ambiguity instead of using file order or
last-wins behavior, and preserves binding provenance in the resolved plan.
