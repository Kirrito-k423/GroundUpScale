# C023 脱敏命令记录

```sh
ps -p 18974 -o pid=,state=,command=
kill -STOP 18974
ps -p 18974 -o pid=,state=,pcpu=,command=
lsof -nP -iTCP:8765 -sTCP:LISTEN
uv run groundupscale preflight --json
```
