# Use a restricted symbolic Spec expression language

Specs express derived parameters, shapes, assertions, and controlled expansion
conditions through a pure, strongly typed expression language over declared
values. Expressions may remain symbolic into later IRs, but cannot execute
Python, templates, shell commands, file or network access, implicit environment
lookups, side effects, or nondeterministic functions. This provides useful
model formulas without turning configuration parsing into code execution.
