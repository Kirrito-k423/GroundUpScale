# Assemble analysis inputs with an Analysis Plan

GroundUpScale uses an Analysis Plan as the human-authored entry point for one
reproducible analysis. It references the selected Workload Spec, Analysis Case,
Deployment Intent, Fabric Graph, and optional Calibration Profile without
duplicating their contents; generated predictions and reports remain separate
outputs so an input plan is never mutated by its results.
