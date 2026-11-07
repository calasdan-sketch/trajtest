# Changelog

## 0.2.0 (unreleased)

Expert-review release. Backward compatible except where noted.

### Fixed
- **Async agents silently passed**: `async def` agents returned an un-awaited
  coroutine, `finished=True`, zero steps. They are now run to completion.
- `fs_glob` was exponential in the number of `**` segments (790 ms for six
  on a 30-deep tree); now linear. An agent-supplied pattern could hang CI.
- `never_called("typo")` passed forever. Names unknown to the run now fail
  with a did-you-mean. (Behaviour change: evals with a misspelled tool name
  in `never_called` / `at_most` / `called_times(...,0)` start failing, which
  is the point.)
- `trajtest run` printed a pytest assert-rewrite warning on every run.
- Repo hygiene: `.venv`, `__pycache__` and `egg-info` were committed; now
  untracked and ignored.

### Added
- Custom tools (`run(..., tools={...})`), `flaky()` fault injection,
  `traj.unused_tools()` defaulting to the run's own tool list.
- Assertions: `only_called`, `fs_writes_confined_to`, `http_confined_to`,
  `always_preceded_by`, `no_repeated_calls`, `no_tool_errors`.
- Trace import/export: `Trajectory.save/load/to_dict/from_dict`,
  `from_openai_messages`, `from_anthropic_messages`; `trajtest audit` CLI.
- `pass_rate` / `assert_pass_rate` for `sample()` results.
- Output hardening: control/bidi/zero-width/newline escaping, secret
  redaction, per-line truncation; FakeFS quota (`FSQuotaExceeded`);
  thread-safe step numbering.
- `bench/bench.py`, CI workflow, 50 new tests (135 -> 185).

### Round 3 (still the unreleased 0.2.0 line; no version bump)
Found by hands-on attacks (security red team, performance, trace-fuzz,
auditor, DX, competitor engineer), then re-attacked on the new code.
- **Security bug**: `never_called("fs_delete", path="/data/x")` was bypassed by
  `/data/./x`, `//data/x`, `/tmp/../data/x`; the file really was deleted.
  Path args now match raw OR normalised.
- **Security bug**: `fs_writes_confined_to` accepted `/out/..\\..\\etc` (escapes on
  Windows); and imported traces whose `bound` disagreed with `args` could hide a
  path/URL. Now fail closed on backslash/NUL/`%2e`, and every copy of the
  argument must pass. Behaviour change: such paths now fail the assertion.
- Trace loader: 9 shapes raised TypeError/AttributeError/RecursionError instead
  of ValueError (audit CLI crashed on a deeply nested file). Fuzzed 3000 random
  documents; only ValueError now.
- Output: printing a 20 MB value took 15 s (redaction ran before truncation) and
  `world_file` failures dumped the whole file; both bounded. Redaction now also
  masks URL query secrets, `user:pass@` URLs, PEM keys, `sk_live_`, `github_pat_`,
  `AIza`, `npm_` shapes.
- `always_preceded_by` was quadratic (7 s for 8k calls); now linear.
- Mock FS quota counts UTF-8 bytes (was characters); FS/HTTP/KV are lock-protected.
- Added: `http_hosts_confined_to`, `no_data_leak`, `no_secret_shapes`,
  `retried_after_error`, `same_tools_as` / `golden_diff`, `run/sample(timeout=)`,
  `AgentTimeout`, `trajtest diff`, `trajtest audit --policy/--json/--http-host-allow/--no-secret-shapes`,
  `python -m trajtest`, `py.typed`, did-you-mean on mistyped tool names,
  Windows-path hint. Behaviour change: `Trajectory.save()` redacts by default
  and writes atomically (`redact=False` for raw).
- Tests 185 -> 244.

### Performance
- Tool-call overhead 17.9 -> 5.5 us (signature inspection cached).
