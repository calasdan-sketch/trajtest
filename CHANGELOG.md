# Changelog

## 0.1.1 - 2026-11-08

First release on PyPI.

- pytest plugin and `trajtest` CLI for asserting on what an agent did (its tool calls and requests), not only its final text.
- `Expect` matchers, including `no_repeated_calls` loop detection.
- Load and save redbg traces (`load_trace`, `Trajectory.save_trace`).
