# Implement execution strategies as multi-phase plugins

`DeploymentIntent` stores typed `StrategyConfiguration`s, while extensible
`StrategyPlugin`s validate them and contribute transformations at named
compiler phases. A plugin may span partitioning, state lifetimes, placement,
communication insertion, scheduling, and estimation, but must declare
preconditions, compatibility rules, generated effects, and explanation
provenance; the core compiler does not encode every strategy as feature flags.
