# C024 脱敏命令记录

```sh
ps -p 50158,50161,64289,64290,18974 -o pid=,ppid=,state=,command=
kill -TERM 50158 50161
kill -TERM <remaining-current-workers>
kill -STOP 18974
lsof -nP -iTCP:8765 -sTCP:LISTEN
lsof -nP -iTCP:8766 -sTCP:LISTEN
uv run groundupscale preflight --json
```
