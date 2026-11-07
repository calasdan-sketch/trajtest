# trajtest — pytest for agent trajectories

**Assert on what an agent *did*, not just what it *said*.**

`trajtest` runs an agent against a mock world (in-memory FS, mock HTTP, KV
store) and lets you assert on its **trajectory** — which tools it called, in
what order, with what arguments, and what side effects it left behind.
Deterministic. Local-first. No LLM-as-judge. CI-ready.

```python
from trajtest import Expect, World, run

def my_agent(task, tools):
    rows = tools.http_get("https://api.example.com/rows").json()["rows"]
    tools.fs_write("/out/report.md", f"total: {sum(rows)}")
    return "done"

def test_agent_saves_before_claiming():
    world = World()
    world.http.mock("GET", "https://api.example.com/rows", json={"rows": [1, 2, 3]})

    traj = run(my_agent, "summarize the rows", world, max_steps=15)

    (Expect(traj)
        .called("http_get", url__contains="api.example.com")
        .then("fs_write", path="/out/report.md")
        .never_called("fs_delete")
        .completed_within(15)
        .finished()
        .world_file("/out/report.md", equals="total: 6")  # the world, not the narration
        .assert_all())
```

## Why

Prompt-tests can't see the four ways agents actually fail:

1. **Confident false reports** — the agent says "saved the file" but never
   called write. `world_file()` checks the mock world's final state, not the
   narration.
2. **Silent policy violations** — narration says "I'll be careful", tool calls
   did a delete. `never_called("fs_delete")` catches it.
3. **Evidence-free passes** — 94% pass rate, nobody knows what was verified.
   Every `EvalFailure` lists each assertion's outcome plus the full trajectory
   transcript. An eval with zero assertions **fails loudly**.
4. **Meaningless coverage** — green suite, dangerous tools never exercised.
   `traj.unused_tools(TOOLS)` flags what the agent never touched.

## Install

```bash
pip install trajtest
```

## Core concepts

**World** — the mock environment. Strict by design: missing files raise
`FileNotFoundError`, unmocked HTTP raises `UnmockedRequestError`. A world
that forgives everything produces agents that pass evals and fail in prod.

- `world.fs` — in-memory filesystem (`read`/`write`/`append`/`delete`/
  `exists`/`listdir`/`glob`; str or bytes; `**` glob support)
- `world.http` — mock HTTP (`mock(method, url, status=..., json=...,
  text=..., error="timeout"|"connection", times=n)`; exact then `*`
  wildcard matching; repeated mocks form a consumable queue — mock a 500
  then a 200 to test retry logic)
- `world.kv` — key-value scratch space; `world.clock` — controllable clock
- `world.reset()` — wipe everything between scenarios

**run(agent, task, world, max_steps=25, seed=None)** — the agent is a plain
function `(task, tools) -> str`. Every tool call is recorded (name, args,
result, duration, errors). Tool errors are recorded *and* re-raised, so the
agent can react to them. `tools.note("...")` captures narration separately —
notes are never steps and never pollute sequences, so you can compare what
the agent *said* against what it *did*. Crashes become `traj.finished=False`
with `traj.failure` set; exceeding `max_steps` is a trajectory failure, not
a hang.

**Custom tools, async agents.** `run(agent, task, world, tools={"send_email": fn})`
records your own tools exactly like the built-ins (they cannot shadow a
built-in). `flaky(fn, fail_times=2, error=TimeoutError("smtp"))` injects
failures so you can prove retry logic. `async def` agents are driven to
completion (tools stay synchronous; call them directly inside the coroutine).

**sample(agent, task, world_factory, n=5)** — for stochastic agents: n runs
against fresh worlds. Seeds are passed when the agent accepts a `seed`
keyword. Assert consistency with a loop:

```python
for t in sample(agent, "task", make_world, n=5):
    Expect(t).never_called("fs_delete").completed_within(15).assert_all()
```

**Expect(traj)** — the assertion DSL. All methods chain; `assert_all()`
reports *every* failure at once (fail-collecting, not fail-fast):

| Assertion | Meaning |
|---|---|
| `called(name, *args, **kwargs)` | ≥1 call matching (positional prefix, kwarg subset, matchers ok) |
| `called_times / at_least / at_most` | call counts |
| `never_called(name, ...)` | forbids calls (optionally arg-scoped) |
| `in_order(a, b, c)` | ordered subsequence, gaps allowed |
| `then(name, ...)` | fluent chain: must occur after previous `then` |
| `then_immediately(name, ...)` | strict adjacency required |
| `first / last` | boundary calls |
| `completed_within(n)` / `finished()` | step budget / clean completion |
| `final_answer(matcher-or-value)` | the agent's return value |
| `notes_contain / notes_match` | narration assertions |
| `world_file(path, equals/contains/matches)` | final world state — the anti-lying assertion |
| `world(desc, predicate)` / `custom(desc, fn)` | escape hatches |
| `only_called(*names)` | tool allowlist |
| `fs_writes_confined_to(*globs)` | every write/append/delete stays inside these paths (fails closed) |
| `http_confined_to(*patterns)` | egress allowlist; rejects `user@host` authority spoofs |
| `always_preceded_by(target, prereq, on=("path",))` | e.g. never delete what you did not read first |
| `no_repeated_calls(max_identical=2)` / `no_tool_errors()` | loop detector / error-free run |

`never_called("fs_delet")` (a typo) **fails** with a did-you-mean instead of
passing forever.

Kwarg matching uses **parameter names**, so `called("http_get",
url__contains="api.example.com")` works whether the agent passed the URL
positionally or by keyword. Suffix sugar: `__contains __regex
__startswith __endswith __gt __lt __ge __le`. Matchers (`ANY`, `Contains`,
`Regex`, `StartsWith`, `EndsWith`, `Gt/Lt/Ge/Le`, `InstanceOf`,
`Predicate`, `DictSubset`) compose anywhere.

**pytest plugin** — a `world` fixture (fresh `World` per test) is
auto-registered. **CLI** — `trajtest init` scaffolds an example eval;
`trajtest run <path>` runs evals through pytest.

## Assert on real runs, not just mocks (v0.2)

Record a production or staging trace (OpenAI-style or Anthropic-style
messages, or `traj.save("golden.json")`) and run the same call-based
assertions on it, no mock world needed:

```python
from trajtest import Expect, Trajectory
traj = Trajectory.from_openai_messages(messages, tools=["fs_read", "fs_delete"])
Expect(traj).notes_contain("only read").never_called("fs_delete").assert_all()
```

Assistant text that accompanies tool calls becomes a *note*, so "said it
would only read, then deleted" is catchable on real transcripts. In CI, no
Python needed:

```bash
trajtest audit traces/*.json --allow-tools fs_read,http_get \
    --http-allow 'https://api.example.com/*' --fs-write-allow '/out/**' \
    --max-repeats 3 --must-precede fs_delete:fs_read:path
# exit 0 clean, 1 violation, 2 bad input or no rules (never passes vacuously)
```

Imported traces carry no world, so `world_file()` / `world()` refuse to run
on them (they raise, they do not guess).

**Pass rates for stochastic agents:** `assert_pass_rate(sample(...), lambda t:
Expect(t).never_called("fs_delete"), min_rate=1.0)` lists failing runs; the
default demands every run pass.

## Safety assertions for security reviewers (round 3)

```python
Expect(traj)
    .http_hosts_confined_to("api.example.com", "*.example.com")  # parsed host, https only
    .no_data_leak(planted_secret)        # raw, URL-encoded or base64 in any HTTP arg
    .no_secret_shapes()                  # sk-..., ghp_..., ?api_key=..., PEM, Bearer ...
    .retried_after_error("send_email")   # failed once, later succeeded
    .same_tools_as(golden, args=False)   # regression vs a saved golden trace
    .assert_all()
```

- `http_hosts_confined_to` compares the URL's real hostname (case-folded,
  port and trailing dot ignored). Rejects userinfo, backslashes, control
  characters, non-ASCII hosts, `http://` and unlisted IPs. Hosts with
  underscores are rejected too (fail closed).
- `no_data_leak` values must be at least 4 characters; the report never
  prints them (known secrets are masked in the whole failure output). It
  searches call arguments only, not results, and only the encodings above.
- `never_called("fs_delete", path="/data/x")` now also matches
  `/data/./x`, `//data/x` and `/tmp/../data/x` (the mock FS resolves them
  to the same file; before, they slipped past).
- `fs_writes_confined_to` checks every copy of the path in a trace and fails
  closed on backslashes, NUL and `%2e/%2f` (ambiguous across OSes).
- `run(agent, task, world, timeout=10)` stops a hung agent: the trajectory is
  frozen with `failure="AgentTimeout..."`. A Python thread cannot be killed,
  so a tool-free infinite loop keeps running in the background.

CLI additions: `trajtest audit --policy policy.json --json` (a reviewable
JSON policy; unknown keys are an error so a typo can't drop a rule; the
report includes each trace's SHA-256), `--http-host-allow`,
`--no-secret-shapes`, and `trajtest diff golden.json new.json [--args]`.
`python -m trajtest` works. `traj.save()` now masks secrets by default
(`redact=False` to keep raw values) and writes atomically.

## Safe to run in CI

Agent text is untrusted. Everything trajtest prints is escaped (ANSI,
newlines, bidi and zero-width characters can't forge report lines) and
credentials are masked: values under keys like `Authorization`, `api_key`,
`token`, `password`, `Bearer ...` strings, and common token shapes (`sk-...`,
`ghp_...`, `AKIA...`, `xox*-`, JWTs). The raw values stay available to your
assertions. Output is bounded before it is scanned (an 8 MB value prints in ~0.2 s, was ~13 s). URL query secrets (`?api_key=`), `user:pass@` URLs, PEM private keys and `sk_live_`/`github_pat_`/`AIza` shapes are masked. The mock FS has a 64 MiB quota counted in UTF-8 bytes (`World(fs_max_bytes=...)`) so a
runaway agent can't exhaust CI memory; `fs_glob` is linear-time.

## How it compares

Measured on one Linux box, Python 3.13, fresh venvs, `pip install` of the
latest release on 2026-11-07 (pip package counts include pip itself):

| | trajtest 0.2.0 | agentevals 0.0.9 | deepeval 4.3.3 | inspect-ai 0.3.288 | promptfoo 0.125.1 |
|---|---|---|---|---|---|
| Packages installed | 7 | 51 | 67 | 81 | n/a (npm: 413 top-level dirs) |
| Disk (site-packages / node_modules) | 21 MB | 99 MB | 118 MB | 248 MB | 734 MB |
| `import` time | 0.07 s | not measured | 2.7 s | 3.1 s | n/a |
| Needs an LLM/API key for its trajectory checks | no (no LLM anywhere) | match modes: no; LLM-judge evaluators exist (`llm.py`) | several metrics are LLM-judged (unverified which) | scorers can be model-graded (unverified per scorer) | unverified |
| Mock world (FS/HTTP) to assert side effects | **yes** | no, compares message trajectories against a reference | no (unverified) | sandboxes exist, unverified | no (unverified) |
| Needs a reference trajectory | no | yes (`reference_outputs`) | unverified | n/a | unverified |
| Runs on recorded OpenAI/Anthropic traces | **yes** | OpenAI-style / LangChain messages | unverified | unverified | unverified |
| Allowlist / egress / read-before-delete / leak checks | **yes** | no | `ToolPermissionMetric` exists (class verified, semantics unverified) | unverified | unverified |

Sources: https://pypi.org/project/agentevals/ and
https://github.com/langchain-ai/agentevals (installed; API read from the
package), https://pypi.org/project/deepeval/ (installed; metric class names
read from the package), https://pypi.org/project/inspect-ai/ and
https://inspect.aisi.org.uk, https://www.promptfoo.dev (installed via npm,
features not tested hands-on). LangSmith evals
(https://docs.smith.langchain.com) is hosted and needs an account; **not
evaluated, unverified**. "Unverified" means we did not test it; check their
docs. Those tools do things trajtest deliberately does not (LLM judges,
datasets, dashboards, red-teaming); they solve a different problem and
combine with trajtest well. agentevals (read from the installed 0.0.9 package) offers strict / unordered / subset / superset trajectory match modes against a reference; trajtest's `same_tools_as` covers the strict and unordered cases only (no subset/superset).

### Benchmarks you can re-run

`python bench/bench.py` (median of 3-5; this box is noisy, so compare on one machine; v0.2.0 as first released vs this branch, 2026-11-07):

| | v0.1.1 | v0.2.0 |
|---|---|---|
| tool-call overhead | 17.9 us/call | 5.5 us/call |
| `fs_glob` with six `**` segments, 30-deep tree | 790 ms (exponential) | < 0.1 ms |
| glob over 3000 files | 5.6 ms | 7.1 ms (same; noise-level) |

| Round 3, same box, interleaved runs | v0.2.0 | this branch |
|---|---|---|
| tool-call overhead | 16-26 us | 23-28 us (new lock + cancel check, roughly +3 us) |
| print the report of a run with an 8 MB value | 13,100 ms | 190 ms |
| `always_preceded_by` over 8k calls | 7,170 ms (quadratic) | 15 ms (40k calls: 83 ms) |
| `redact` of 300 KB hostile text | 300 ms | 270-350 ms (unchanged) |

(The 5.5 us overhead above was measured on a faster moment/machine; absolute
microseconds are not comparable across runs.)

## Limitations (honest)

- **trajtest can't tell you your assertions are meaningful.**
  `Expect(t).finished().assert_all()` is a weak eval and it will pass.
  Assert on trajectories and world state, not vibes.
- **The mock world is not the real world.** No network latency, no real
  filesystem permissions, no file metadata (size/mtime), no concurrent
  access. Fidelity gaps are where false confidence breeds — keep mocks
  strict and review them.
- **Deterministic assertions vs stochastic agents** is an inherent tension.
  Use `sample()`, seeds, and order-flexible assertions. If your agent is
  wildly non-deterministic, trajectory evals will be brittle — that's a
  signal about the agent, not just the eval.
- **Threads are safe, subprocesses are not.** Tool calls from threads get
  unique step numbers and respect `max_steps`; the mock FS/HTTP/KV are lock-protected; subprocesses can't see the
  mock world. Async tools (`await tools.x()`) are not supported: tools are
  synchronous.
- **Redaction is best-effort.** It masks secret-looking keys and common
  token shapes; a custom secret passed positionally with no recognisable
  shape (`kv_set("k", "hunter2")`) is printed. Don't put real secrets in
  evals.
- **Imported traces have no world and no timing**, and OpenAI/Anthropic
  formats don't mark tool failures (Anthropic `is_error` is honoured).
- **Not covered:** percent-decoded or Unicode-confusable *paths* beyond the ambiguity rejection above; encodings other than raw/URL/base64 in `no_data_leak`; HTTP 4xx/5xx *responses* count as errors only if the tool raised (`retried_after_error`); case-insensitive filesystems (`/ETC` vs `/etc`).
- `no_secret_shapes` and redaction key off key names containing `token`, `secret`, `password`, `api_key`...; a benign key like `tokenizer` is a false positive.
- Tested on Python 3.13 locally; 3.10 is the declared floor and is
  exercised only by the CI workflow (unverified until it runs).
- **No framework adapters.** The agent is `(task, tools) -> str`;
  wrapping a LangChain/CrewAI loop takes ~5 lines on your side (or record
  its messages and use `from_openai_messages`).
- **No LLM-as-judge**, by design. Semantic judgments belong in a custom
  `predicate`, not in the core.

## License

MIT
