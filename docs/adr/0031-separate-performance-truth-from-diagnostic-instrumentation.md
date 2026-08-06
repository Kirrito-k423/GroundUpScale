# Separate performance truth from diagnostic instrumentation

Each measurable case supports a minimally instrumented benchmark run, a
structured diagnostic trace run, and optional targeted deep-probe runs. Hooks
emit scope correlation and tensor metadata while framework and runtime
collectors capture operator, device, memory, and scheduling events; per-module
device synchronization and unstructured printing are not accepted as E2E
performance truth. Raw evidence remains immutable, and an Alignment Map relates
normalized observations to compiled identities with explicit confidence and
unattributed buckets.
