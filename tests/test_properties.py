"""Round 5: property-based tests (hypothesis). Skipped if hypothesis is absent.

Each property states an invariant a reviewer would rely on; the fixed seed
makes CI runs reproducible.
"""

import contextlib
import fnmatch
import json
import posixpath
import unicodedata

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from trajtest import Expect, Trajectory, World, run, traces
from trajtest.expect import golden_diff
from trajtest.sanitize import clean, redact, safe

SET = settings(max_examples=150, deadline=None, derandomize=True,
               suppress_health_check=[HealthCheck.too_slow])

names = st.sampled_from(["fs_read", "fs_write", "http_get", "kv_get", "kv_set"])
text = st.text(max_size=60)


def _agent_for(plan):
    def agent(task, tools):
        for name, arg in plan:
            with contextlib.suppress(Exception):  # tool errors are part of the trajectory
                if name == "fs_read":
                    tools.fs_read("/f/" + arg)
                elif name == "fs_write":
                    tools.fs_write("/f/" + arg, arg)
                elif name == "http_get":
                    tools.http_get("https://a.test/" + arg)
                elif name == "kv_get":
                    tools.kv_get(arg, None)
                else:
                    tools.kv_set(arg, 1)
        return "done"
    return agent


def _world():
    w = World()
    w.http.mock("GET", "*", text="ok")
    return w


plans = st.lists(st.tuples(names, st.text(alphabet="abc", min_size=1, max_size=3)), max_size=15)


@SET
@given(plans)
def test_recorded_steps_equal_calls_made_and_golden_is_reflexive(plan):
    traj = run(_agent_for(plan), "t", _world(), max_steps=50)
    assert traj.steps == len(plan)
    assert golden_diff(traj, traj) == []
    assert golden_diff(traj, traj, ordered=False, args=True) == []


@SET
@given(plan=plans)
def test_save_load_roundtrip_preserves_tool_sequence(plan, tmp_path_factory):
    traj = run(_agent_for(plan), "t", _world(), max_steps=50)
    p = tmp_path_factory.mktemp("rt") / "t.json"
    traj.save(p)
    again = Trajectory.load(p)
    assert again.tool_names == traj.tool_names
    assert golden_diff(traj, again, args=True) == []


@SET
@given(plans, plans)
def test_golden_diff_empty_iff_same_sequence(a, b):
    ta = run(_agent_for(a), "t", _world(), max_steps=50)
    tb = run(_agent_for(b), "t", _world(), max_steps=50)
    same = ta.tool_names == tb.tool_names
    assert (golden_diff(ta, tb) == []) == same
    same_bag = sorted(ta.tool_names) == sorted(tb.tool_names)
    assert (golden_diff(ta, tb, ordered=False) == []) == same_bag


@SET
@given(text)
def test_safe_and_clean_output_has_no_control_or_bidi_characters(s):
    for out in (safe(s), clean(s)):
        assert "\n" not in out and "\r" not in out and "\x1b" not in out
        assert not any(unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp", "Cs", "Co", "Cn")
                       for c in out), repr(out)
        assert not any(chr(c) in out for c in (0x202E, 0x202D, 0x2066, 0x200B))


@SET
@given(text)
def test_redact_never_raises_and_is_idempotent(s):
    once = redact(s)
    assert redact(once) == once


@SET
@given(st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40))
def test_redact_removes_embedded_sk_key(prefix):
    secret = "sk-" + "A1b2C3d4E5" * 3
    assert secret not in redact(f"{prefix} {secret} tail")


@SET
@given(st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=True) | st.text(max_size=10),
    lambda c: st.lists(c, max_size=4) | st.dictionaries(st.text(max_size=5), c, max_size=4),
    max_leaves=25))
def test_trace_loader_only_raises_valueerror(doc):
    try:
        traces.detect_and_load(doc)
    except ValueError:
        pass


@SET
@given(st.lists(st.text(alphabet="ab/", min_size=1, max_size=6), max_size=8),
       st.sampled_from(["/f/*", "/f/**", "/f/a*", "/**/b", "/*"]))
def test_fs_glob_results_exist_and_are_sorted_unique(paths, pattern):
    w = World()
    for p in paths:
        with contextlib.suppress(Exception):  # odd paths may be rejected; that is fine here
            w.fs.write("/f/" + p.strip("/") or "/f/x", "x")
    got = w.fs.glob(pattern)
    assert got == sorted(set(got))
    assert all(w.fs.exists(g) for g in got)
    if "**" not in pattern:
        # without ** a glob must agree with fnmatch on single-segment depth
        for g in got:
            assert len(g.split("/")) == len(pattern.split("/"))
            assert fnmatch.fnmatchcase(posixpath.normpath(g), pattern.replace("**", "*"))


@SET
@given(st.lists(st.text(alphabet="abc", min_size=1, max_size=3), max_size=10))
def test_fs_quota_accounting_returns_to_zero_after_deleting_everything(keys):
    w = World()
    for k in keys:
        w.fs.write("/q/" + k, k * 5)
    for k in set(keys):
        w.fs.delete("/q/" + k)
    assert w.fs._used == 0


@SET
@given(plans)
def test_expect_never_called_agrees_with_recorded_names(plan):
    traj = run(_agent_for(plan), "t", _world(), max_steps=50)
    for n in ["fs_read", "fs_write", "http_get", "kv_get", "kv_set"]:
        ex = Expect(traj).never_called(n)
        ok = all(v.ok for v in ex.check())
        assert ok == (n not in traj.tool_names)


@SET
@given(name=st.text(alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="/\x00"), min_size=1, max_size=12))
def test_audit_json_is_valid_json_for_hostile_file_names(name, tmp_path_factory):
    import contextlib
    import io

    from trajtest import cli
    p = tmp_path_factory.mktemp("a") / (name.strip(".") or "x")
    p = p.with_name(p.name + ".json")
    traces.save(run(_agent_for([("kv_set", "a")]), "t", _world()), p)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["audit", str(p), "--max-repeats", "3", "--json"])
    assert rc == 0
    assert json.loads(buf.getvalue())["ok"] is True
