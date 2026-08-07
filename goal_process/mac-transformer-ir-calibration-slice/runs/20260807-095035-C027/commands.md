# C027 脱敏命令记录

```sh
uv run pytest tests/test_environment_validity.py::<target-test> -q
uv run pytest tests/test_environment_validity.py tests/test_run_bundle.py -q
uv run pytest -q
uv run groundupscale check-environment --json
git push origin main
gh run watch <run-id> --exit-status
```
