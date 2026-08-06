# Make explanations queryable before visualizing them

GroundUpScale derives a first-class Explanation Graph that connects every
reported metric to scopes, formulas, implementation choices, schedule causes,
uncertainty, calibration evidence, and provenance. CLI, standalone HTML, and Web
views are adapters over the same explanation interface rather than independent
explanation implementations. Metric Derivations declare whether contributions
are additive, inclusive, exclusive, shared, critical-path, or peak-live-set so
parallel time and aliased memory are not presented as misleading sums.
