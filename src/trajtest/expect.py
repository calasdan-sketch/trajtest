"""The Expect DSL: fail-collecting assertions over a trajectory.

Every method records its outcome and returns self, so evals read as a chain::

    Expect(traj) \\
        .called("http_get", url__contains="api.example.com") \\
        .then("fs_write", path="/out/report.md") \\
        .never_called("fs_delete") \\
        .completed_within(15) \\
        .world_file("/out/report.md", contains="total") \\
        .assert_all()

assert_all() reports EVERYTHING that failed — plus a trajectory transcript —
in a single EvalFailure. check() returns the violations without raising.
"""

import difflib
import fnmatch
from dataclasses import dataclass
from urllib.parse import urlsplit

from trajtest import matchers as m
from trajtest.errors import EvalFailure
from trajtest.sanitize import clean
from trajtest.world import _norm, _segments_match

_MISSING = object()


@dataclass
class Violation:
    ok: bool
    description: str
    detail: str = ""

    def line(self) -> str:
        mark = "✓" if self.ok else "✗"
        text = f"  {mark} {clean(self.description)}"
        if not self.ok and self.detail:
            text += f"\n      {clean(self.detail)}"
        return text


def call_matches(call, args, kwargs) -> bool:
    """Prefix-match positionals; subset-match kwargs by parameter name.

    Kwarg matching uses the call's bound arguments, so
    called("http_get", url__contains="...") matches whether the agent
    passed the URL positionally or by keyword.
    """
    if len(args) > len(call.args):
        return False
    for expected, actual in zip(args, call.args):
        if not m.to_matcher(expected).matches(actual):
            return False
    bound = call.bound or {}
    for key, expected in kwargs.items():
        name, matcher = m.parse_kwarg(key, expected)
        actual = bound.get(name, _MISSING)
        if actual is _MISSING:
            actual = call.kwargs.get(name, _MISSING)
        if actual is _MISSING or not matcher.matches(actual):
            return False
    return True


_FS_MUTATORS = ("fs_write", "fs_append", "fs_delete")
_MISSING_ARG = object()


def _arg(call, name: str, pos: int):
    """A call's argument by parameter name, falling back to position."""
    bound = call.bound or {}
    if name in bound:
        return bound[name]
    if name in call.kwargs:
        return call.kwargs[name]
    if pos < len(call.args):
        return call.args[pos]
    return _MISSING_ARG


def _url_pos(call) -> int:
    return 1 if call.name == "http_request" else 0


def _fmt_call_args(call) -> str:
    parts = [repr(a) for a in call.args]
    parts += [f"{k}={v!r}" for k, v in call.kwargs.items()]
    return ", ".join(parts)


class Expect:
    def __init__(self, traj):
        self.traj = traj
        self._violations: list[Violation] = []
        self._chain_cursor = 0  # for then()/then_immediately() chains

    # -- internal ------------------------------------------------------
    def _record(self, ok: bool, description: str, detail: str = "") -> "Expect":
        self._violations.append(Violation(ok, description, detail))
        return self

    def _matching(self, name, args, kwargs):
        return [c for c in self.traj.calls_to(name) if call_matches(c, args, kwargs)]

    def _require_world(self):
        world = getattr(self.traj, "world", None)
        if world is None:
            raise ValueError(
                "this assertion needs the mock world, but the trajectory "
                "has none attached (trajectories from trajtest.run() always do)"
            )
        return world

    @staticmethod
    def _coerce_result(value):
        if isinstance(value, tuple) and len(value) == 2:
            return bool(value[0]), str(value[1])
        return bool(value), ""

    # -- existence -----------------------------------------------------
    def called(self, name: str, *args, **kwargs) -> "Expect":
        """At least one call to `name` matching the given args.

        Positional args are prefix-matched; kwargs are subset-matched.
        `url__contains="x"` style suffixes build matchers (see matchers).
        """
        hits = self._matching(name, args, kwargs)
        desc = f'called("{name}"{self._fmt_expect(args, kwargs)})'
        if hits:
            return self._record(True, desc)
        detail = (
            f"no call to {name!r} matched; "
            f"{len(self.traj.calls_to(name))} call(s) to {name!r} total, "
            f"{self.traj.steps} step(s) overall"
        )
        return self._record(False, desc, detail)

    # alias: reads better in some evals
    called_with = called

    def _unknown_name(self, desc: str, name: str) -> "Expect | None":
        """Fail closed on a tool name that never existed in this run.

        `never_called("fs_delet")` would otherwise pass forever on a typo.
        Skipped for imported traces with no known tool list.
        """
        reg = self.traj.registered
        if reg is None or name in reg or name in self.traj.tool_names:
            return None
        close = difflib.get_close_matches(name, reg, n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        return self._record(
            False, desc, f"unknown tool {name!r}: no such tool in this run{hint}"
        )

    def called_times(self, name: str, n: int) -> "Expect":
        if n == 0 and (bad := self._unknown_name(f'called_times("{name}", 0)', name)):
            return bad
        count = len(self.traj.calls_to(name))
        return self._record(
            count == n,
            f'called_times("{name}", {n})',
            f"found {count}" if count != n else "",
        )

    def at_least(self, n: int, name: str) -> "Expect":
        count = len(self.traj.calls_to(name))
        return self._record(
            count >= n,
            f'at_least({n}, "{name}")',
            f"found {count}" if count < n else "",
        )

    def at_most(self, n: int, name: str) -> "Expect":
        if bad := self._unknown_name(f'at_most({n}, "{name}")', name):
            return bad
        count = len(self.traj.calls_to(name))
        return self._record(
            count <= n, f'at_most({n}, "{name}")', f"found {count}" if count > n else ""
        )

    def never_called(self, name: str, *args, **kwargs) -> "Expect":
        """Forbid calls to `name` — optionally only those matching args."""
        desc = f'never_called("{name}"{self._fmt_expect(args, kwargs)})'
        if bad := self._unknown_name(desc, name):
            return bad
        hits = self._matching(name, args, kwargs)
        if not hits:
            return self._record(True, desc)
        shown = "; ".join(f"#{c.step} {c.name}({_fmt_call_args(c)})" for c in hits[:3])
        return self._record(False, desc, f"forbidden call(s): {shown}")

    # -- order ---------------------------------------------------------
    @staticmethod
    def _norm_item(item):
        if isinstance(item, str):
            return item, (), {}
        if len(item) == 2:
            name, kwargs = item
            return name, (), dict(kwargs)
        name, args, kwargs = item
        return name, tuple(args), dict(kwargs)

    def in_order(self, *items) -> "Expect":
        """Tool names (or (name, kwargs) / (name, args, kwargs) tuples)
        appearing as a subsequence.

        Gaps are allowed: other calls may happen in between. Use
        then_immediately() when adjacency matters.
        """
        desc = f"in_order({', '.join(repr(i) for i in items)})"
        cursor = 0
        calls = self.traj.calls
        for item in items:
            name, args, kwargs = self._norm_item(item)
            found = next(
                (
                    i
                    for i in range(cursor, len(calls))
                    if calls[i].name == name and call_matches(calls[i], args, kwargs)
                ),
                None,
            )
            if found is None:
                return self._record(
                    False,
                    desc,
                    f"{name!r} not found after position {cursor}; "
                    f"trajectory: {self.traj.tool_names}",
                )
            cursor = found + 1
        return self._record(True, desc)

    def then(self, name: str, *args, **kwargs) -> "Expect":
        """Chain step: `name` (with matching args) must occur after the
        previous then() match.

        Gaps allowed. Each then() continues from where the last one matched,
        so then-chains express "first do A, then B, then C" fluently.
        """
        desc = f'then("{name}"{self._fmt_expect(args, kwargs)})'
        calls = self.traj.calls
        found = next(
            (
                i
                for i in range(self._chain_cursor, len(calls))
                if calls[i].name == name and call_matches(calls[i], args, kwargs)
            ),
            None,
        )
        if found is None:
            return self._record(
                False,
                desc,
                f"{name!r} not found after step {self._chain_cursor}; "
                f"trajectory: {self.traj.tool_names}",
            )
        self._chain_cursor = found + 1
        return self._record(True, desc)

    def then_immediately(self, name: str, *args, **kwargs) -> "Expect":
        """Chain step requiring strict adjacency — no calls in between."""
        desc = f'then_immediately("{name}"{self._fmt_expect(args, kwargs)})'
        calls = self.traj.calls
        nxt = calls[self._chain_cursor] if self._chain_cursor < len(calls) else None
        if nxt is not None and nxt.name == name and call_matches(nxt, args, kwargs):
            self._chain_cursor += 1
            return self._record(True, desc)
        got = nxt.short() if nxt else "<end of trajectory>"
        return self._record(False, desc, f"expected {name!r} immediately, got: {got}")

    def first(self, name: str, *args, **kwargs) -> "Expect":
        desc = f'first("{name}"{self._fmt_expect(args, kwargs)})'
        calls = self.traj.calls
        if calls and calls[0].name == name and call_matches(calls[0], args, kwargs):
            return self._record(True, desc)
        got = calls[0].short() if calls else "<no calls>"
        return self._record(False, desc, f"first call was: {got}")

    def last(self, name: str, *args, **kwargs) -> "Expect":
        desc = f'last("{name}"{self._fmt_expect(args, kwargs)})'
        calls = self.traj.calls
        if calls and calls[-1].name == name and call_matches(calls[-1], args, kwargs):
            return self._record(True, desc)
        got = calls[-1].short() if calls else "<no calls>"
        return self._record(False, desc, f"last call was: {got}")

    # -- budgets & completion ------------------------------------------
    def completed_within(self, n: int) -> "Expect":
        steps = self.traj.steps
        return self._record(
            steps <= n,
            f"completed_within({n})",
            f"took {steps} steps" if steps > n else "",
        )

    def finished(self) -> "Expect":
        ok = self.traj.finished and not self.traj.failure
        return self._record(
            ok,
            "finished()",
            f"agent did not finish: {self.traj.failure}" if not ok else "",
        )

    # -- output & narration --------------------------------------------
    def final_answer(self, expected) -> "Expect":
        """Match the agent's return value: a Matcher, or a literal (==)."""
        matcher = m.to_matcher(expected)
        ok = matcher.matches(self.traj.final_answer)
        return self._record(
            ok,
            f"final_answer({matcher.describe()})",
            f"got {self.traj.final_answer!r}" if not ok else "",
        )

    def final_answer_contains(self, text: str) -> "Expect":
        return self.final_answer(m.Contains(text))

    def notes_contain(self, text: str) -> "Expect":
        ok = any(text in n.text for n in self.traj.notes)
        return self._record(
            ok,
            f"notes_contain({text!r})",
            "no note contained it" if not ok else "",
        )

    def notes_match(self, pattern: str) -> "Expect":
        matcher = m.Regex(pattern)
        ok = any(matcher.matches(n.text) for n in self.traj.notes)
        return self._record(
            ok,
            f"notes_match({pattern!r})",
            "no note matched" if not ok else "",
        )

    # -- world state (the anti-lying assertions) ------------------------
    def world_file(
        self, path: str, *, equals=None, contains=None, matches=None
    ) -> "Expect":
        """Assert on the mock world's final file state.

        The agent can *say* it saved the file; this checks that it did.
        Pass one of equals/contains/matches (a Matcher), or none to just
        check existence.
        """
        world = self._require_world()
        desc = f"world_file({path!r})"
        try:
            content = world.fs.read(path)
        except FileNotFoundError:
            return self._record(False, desc, f"{path!r} does not exist in mock world")
        checks = {"equals": equals, "contains": contains, "matches": matches}
        given = {k: v for k, v in checks.items() if v is not None}
        if not given:
            return self._record(True, desc + " exists")
        if len(given) > 1:
            raise ValueError("world_file: pass at most one of equals/contains/matches")
        kind, want = next(iter(given.items()))
        matcher = {
            "equals": m.to_matcher(want),
            "contains": m.Contains(want),
            "matches": m.to_matcher(want),
        }[kind]
        ok = matcher.matches(content)
        return self._record(
            ok,
            f"{desc} {kind}={matcher.describe()}",
            f"file content was {content!r}" if not ok else "",
        )

    def world(self, description: str, predicate) -> "Expect":
        """Custom assertion over the whole mock world.

        predicate(world) -> bool, or (bool, message).
        """
        w = self._require_world()
        ok, msg = self._coerce_result(predicate(w))
        return self._record(ok, f"world({description})", msg if not ok else "")

    def custom(self, description: str, fn) -> "Expect":
        """Escape hatch: fn(trajectory) -> bool, or (bool, message)."""
        ok, msg = self._coerce_result(fn(self.traj))
        return self._record(ok, f"custom({description})", msg if not ok else "")

    # -- safety scope (allowlists) --------------------------------------
    def only_called(self, *names: str) -> "Expect":
        """Tool allowlist: every call must be to one of `names`."""
        desc = f"only_called({', '.join(repr(n) for n in names)})"
        bad = [c for c in self.traj.calls if c.name not in names]
        if not bad:
            return self._record(True, desc)
        shown = "; ".join(c.short() for c in bad[:3])
        return self._record(False, desc, f"{len(bad)} call(s) outside allowlist: {shown}")

    def fs_writes_confined_to(self, *globs: str) -> "Expect":
        """Every fs_write/fs_append/fs_delete must target a path matching
        one of the absolute `globs` (supports ``*`` and ``**``).

        Fails closed: a mutation whose path is missing, relative, or not
        resolvable is a violation, not a pass. Works on imported traces too.
        """
        desc = f"fs_writes_confined_to({', '.join(repr(g) for g in globs)})"
        if not globs:
            raise ValueError("fs_writes_confined_to needs at least one glob")
        pats = [[s for s in _norm(g).split("/") if s] for g in globs]
        bad = []
        for c in self.traj.calls:
            if c.name not in _FS_MUTATORS:
                continue
            raw = _arg(c, "path", 0)
            try:
                if raw is _MISSING_ARG:
                    raise ValueError("no path argument")
                segs = [x for x in _norm(raw).split("/") if x]
            except (ValueError, TypeError):
                bad.append(f"{c.short()} (path unresolvable)")
                continue
            if not any(_segments_match(segs, p) for p in pats):
                bad.append(c.short())
        if not bad:
            return self._record(True, desc)
        return self._record(False, desc, f"{len(bad)} write(s) outside scope: " + "; ".join(bad[:3]))

    def http_confined_to(self, *patterns: str) -> "Expect":
        """Egress allowlist: every http_* call's URL must match one of the
        fnmatch `patterns` (e.g. ``https://api.example.com/*``).

        Fails closed on a missing URL, control characters, or URLs with
        embedded credentials / ``@`` in the authority (a classic host-spoof).
        """
        desc = f"http_confined_to({', '.join(repr(p) for p in patterns)})"
        if not patterns:
            raise ValueError("http_confined_to needs at least one pattern")
        bad = []
        for c in self.traj.calls:
            if not c.name.startswith("http_"):
                continue
            url = _arg(c, "url", _url_pos(c))
            if not isinstance(url, str) or any(ord(ch) < 32 for ch in url):
                bad.append(f"{c.short()} (url unresolvable)")
                continue
            try:
                spoof = "@" in urlsplit(url).netloc
            except ValueError:
                spoof = True
            if spoof or not any(fnmatch.fnmatchcase(url, p) for p in patterns):
                bad.append(c.short())
        if not bad:
            return self._record(True, desc)
        return self._record(False, desc, f"{len(bad)} request(s) outside egress allowlist: " + "; ".join(bad[:3]))

    def always_preceded_by(self, target: str, prerequisite: str, on=("path",)) -> "Expect":
        """Every `target` call needs an earlier, *successful* `prerequisite`
        call with equal values for each parameter in `on`.

        Example: ``always_preceded_by("fs_delete", "fs_read", on=("path",))``
        = never delete a file you did not read first.
        """
        on = (on,) if isinstance(on, str) else tuple(on)
        desc = f'always_preceded_by("{target}", "{prerequisite}", on={on})'
        bad = []
        calls = self.traj.calls
        for i, c in enumerate(calls):
            if c.name != target:
                continue
            want = [_arg(c, k, 0) for k in on]
            ok = any(
                p.name == prerequisite
                and p.error is None
                and [_arg(p, k, 0) for k in on] == want
                for p in calls[:i]
            )
            if not ok:
                bad.append(c.short())
        if not bad:
            return self._record(True, desc)
        return self._record(False, desc, f"{len(bad)} call(s) without prerequisite: " + "; ".join(bad[:3]))

    def no_repeated_calls(self, max_identical: int = 2) -> "Expect":
        """Loop detector: no (tool, args) combination may occur more than
        `max_identical` times."""
        desc = f"no_repeated_calls(max_identical={max_identical})"
        seen: dict = {}
        for c in self.traj.calls:
            key = (c.name, repr(c.args), repr(sorted(c.kwargs.items(), key=lambda kv: kv[0])))
            seen.setdefault(key, []).append(c)
        bad = [v for v in seen.values() if len(v) > max_identical]
        if not bad:
            return self._record(True, desc)
        worst = max(bad, key=len)
        return self._record(False, desc, f"{worst[0].short()} repeated {len(worst)}x")

    def no_tool_errors(self) -> "Expect":
        """No tool call raised (use after retries are expected to succeed)."""
        errs = [c for c in self.traj.calls if c.error]
        if not errs:
            return self._record(True, "no_tool_errors()")
        return self._record(False, "no_tool_errors()", f"{len(errs)} failed call(s): " + "; ".join(c.short() for c in errs[:3]))

    # -- verdict ---------------------------------------------------------
    def check(self) -> list[Violation]:
        """Return all violations without raising.

        Like assert_all(), an assertion-free Expect yields a failure —
        otherwise `assert not exp.check()` would pass vacuously.
        """
        if not self._violations:
            return [
                Violation(
                    False,
                    "no assertions recorded",
                    "an eval that checks nothing always passes",
                )
            ]
        return list(self._violations)

    def summary(self) -> str:
        return "\n".join(v.line() for v in self._violations)

    def assert_all(self) -> None:
        """Raise EvalFailure listing every failed assertion + transcript.

        An eval with zero assertions fails loudly: an eval that checks
        nothing always passes, which is exactly the false confidence
        trajtest exists to kill.
        """
        if not self._violations:
            raise EvalFailure(
                "trajtest: no assertions were recorded.\n"
                "An eval that checks nothing always passes — add assertions "
                "(e.g. .called(...), .never_called(...), .world_file(...)) "
                "before calling assert_all()."
            )
        failed = [v for v in self._violations if not v.ok]
        if not failed:
            return
        report = [
            (
                f"trajtest eval FAILED: {len(failed)} of {len(self._violations)} "
                f"assertion(s) failed"
            ),
            self.summary(),
            f"Trajectory ({self.traj.steps} steps):",
            self.traj.transcript(),
        ]
        raise EvalFailure("\n".join(report))

    @staticmethod
    def _fmt_expect(args, kwargs) -> str:
        parts = [repr(a) for a in args]
        parts += [f"{k}={v!r}" for k, v in kwargs.items()]
        return (", " + ", ".join(parts)) if parts else ""
