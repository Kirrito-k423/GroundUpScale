# C025 脱敏命令记录

```sh
launchctl print gui/502/<exact-label>
launchctl bootout gui/502/com.autoresearch.qwen35-24h-20260714
launchctl bootout gui/502/com.autoresearch.qwen3vl8b-mopd-24h-20260720
launchctl print gui/502/<exact-label>
ps -A -o pid=,ppid=,state=,pcpu=,command=
lsof -nP -iTCP:8765 -sTCP:LISTEN
lsof -nP -iTCP:8766 -sTCP:LISTEN
uv run groundupscale preflight --json
```
