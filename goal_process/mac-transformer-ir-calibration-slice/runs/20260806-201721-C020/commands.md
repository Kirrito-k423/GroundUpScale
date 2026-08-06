# C020 脱敏命令记录

```sh
pmset -g batt
pmset -g therm
uptime
sysctl -n hw.logicalcpu hw.physicalcpu
ps -A -o pid=,pcpu=,pmem=,comm=
uv run pytest <targeted-tests>
uv run groundupscale preflight --json
uv run pytest -q
git diff --check
```
