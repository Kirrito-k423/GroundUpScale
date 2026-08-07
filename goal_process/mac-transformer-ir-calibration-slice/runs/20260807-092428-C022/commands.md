# C022 脱敏命令记录

```sh
lsof -nP -iTCP:8765 -sTCP:LISTEN
lsof -nP -iTCP:8766 -sTCP:LISTEN
ps -p <resolved-pids> -o pid=,ppid=,state=,etime=,pcpu=,pmem=,command=
kill -TERM <resolved-listener-pids>
uv run groundupscale preflight --json
```
