"""Regression tests for the v0.2 expert-review findings."""

import asyncio
import json
import random
import threading

import pytest

from trajtest import (
    EvalFailure,
    Expect,
    FSQuotaExceeded,
    Trajectory,
    World,
    assert_pass_rate,
    flaky,
    pass_rate,
    run,
    sample,
)
from trajtest.cli import main
from trajtest.sanitize import clean, redact, safe
from trajtest.world import _segments_match

# Fake credentials are assembled at runtime so no secret-shaped literal sits
# in the source tree.
FAKE = "sk" + "-" + "t3st" + "0" * 12


# -- async agents (v0.1 silently passed with an un-awaited coroutine) -------
def test_async_agent_is_driven_to_completion():
    async def agent(task, tools):
        await asyncio.sleep(0)
        tools.fs_write("/a", "x")
        return "ok"

    traj = run(agent, "t", World())
    assert traj.finished and traj.final_answer == "ok" and traj.steps == 1


def test_async_agent_crash_is_a_trajectory_failure():
    async def agent(task, tools):
        raise RuntimeError("boom")

    traj = run(agent, "t", World())
    assert not traj.finished and "boom" in traj.failure


def test_async_agent_inside_running_loop():
    async def agent(task, tools):
        tools.kv_set("k", 1)
        return "ok"

    async def outer():
        return run(agent, "t", World())

    assert asyncio.run(outer()).finished


def test_async_agent_respects_max_steps():
    async def agent(task, tools):
        for i in range(10):
            tools.kv_set(i, i)

    traj = run(agent, "t", World(), max_steps=3)
    assert not traj.finished and "max_steps" in traj.failure


# -- custom tools -----------------------------------------------------------
def test_custom_tools_are_recorded_and_matchable():
    sent = []
    tools = {"send_email": lambda to, body="": sent.append(to) or "queued"}

    def agent(task, t):
        return t.send_email("a@example.com", body="hi")

    traj = run(agent, "t", World(), tools=tools)
    assert sent == ["a@example.com"] and traj.final_answer == "queued"
    Expect(traj).called("send_email", to__contains="@example.com").assert_all()
    assert "send_email" in traj.registered


def test_custom_tool_cannot_shadow_builtin_or_use_bad_name():
    with pytest.raises(ValueError, match="shadow"):
        run(lambda t, x: "ok", "t", World(), tools={"fs_write": print})
    with pytest.raises(ValueError, match="invalid"):
        run(lambda t, x: "ok", "t", World(), tools={"__class__": print})
    with pytest.raises(TypeError):
        run(lambda t, x: "ok", "t", World(), tools={"x": 5})


def test_flaky_injects_failures_then_succeeds():
    tools = {"ping": flaky(lambda: "pong", fail_times=2, error=TimeoutError("t"))}

    def agent(task, t):
        for _ in range(5):
            try:
                return t.ping()
            except TimeoutError:
                continue

    traj = run(agent, "t", World(), tools=tools)
    assert traj.final_answer == "pong" and traj.steps == 3
    assert [bool(c.error) for c in traj.calls] == [True, True, False]


def test_unused_tools_defaults_to_this_runs_tools():
    traj = run(lambda t, x: x.kv_set("a", 1), "t", World(), tools={"danger": print})
    assert "danger" in traj.unused_tools() and "kv_set" not in traj.unused_tools()


# -- unknown tool names fail closed -----------------------------------------
def test_never_called_typo_fails_instead_of_passing_forever():
    traj = run(lambda t, x: "ok", "t", World())
    with pytest.raises(EvalFailure, match="did you mean 'fs_delete'"):
        Expect(traj).never_called("fs_delet").assert_all()
    Expect(traj).never_called("fs_delete").assert_all()


def test_unknown_name_check_skipped_without_tool_list():
    traj = Trajectory.from_openai_messages([{"role": "assistant", "content": "hi"}])
    Expect(traj).never_called("anything").assert_all()


# -- glob: linear time and same answers as the old recursive matcher --------
def _old_match(path_segs, pat_segs):
    import fnmatch

    if not pat_segs:
        return not path_segs
    if pat_segs[0] == "**":
        return any(_old_match(path_segs[i:], pat_segs[1:]) for i in range(len(path_segs) + 1))
    if not path_segs or not fnmatch.fnmatchcase(path_segs[0], pat_segs[0]):
        return False
    return _old_match(path_segs[1:], pat_segs[1:])


def test_glob_dp_matches_reference_on_random_inputs():
    rng = random.Random(7)
    for _ in range(3000):
        path = [rng.choice("ab") + rng.choice("ab.") for _ in range(rng.randint(0, 5))]
        pat = [rng.choice(["**", "*", "a*", "?b", "ab", "b."]) for _ in range(rng.randint(0, 5))]
        assert _segments_match(path, pat) == _old_match(path, pat), (path, pat)


def test_glob_many_double_stars_does_not_hang():
    w = World()
    w.fs.write("/" + "/".join("d" * 1 for _ in range(60)) + "/x.txt", "1")
    assert w.fs.glob("/" + "/".join(["**"] * 40) + "/y.txt") == []


# -- filesystem quota --------------------------------------------------------
def test_fs_quota_blocks_runaway_writes_and_appends():
    w = World(fs_max_bytes=100)
    w.fs.write("/a", "x" * 60)
    with pytest.raises(FSQuotaExceeded):
        w.fs.write("/b", "x" * 60)
    w.fs.delete("/a")
    w.fs.write("/b", "x" * 60)  # space reclaimed
    with pytest.raises(FSQuotaExceeded):
        w.fs.append("/b", "x" * 60)
    w.fs.write("/b", "y")  # overwrite shrinks
    w.fs.reset()
    w.fs.write("/c", "x" * 100)


def test_fs_quota_can_be_disabled():
    w = World(fs_max_bytes=None)
    w.fs.write("/a", "x" * 1000)


# -- sanitising untrusted text ----------------------------------------------
def test_note_cannot_forge_report_lines_or_inject_escapes():
    def agent(task, tools):
        tools.note("done\n  ✓ finished()\x1b[2J\u202ehidden\u200b")

    traj = run(agent, "t", World())
    out = traj.transcript()
    assert "\x1b" not in out and "\u202e" not in out and "\u200b" not in out
    assert not [ln for ln in out.splitlines() if ln.lstrip().startswith("✓")]
    assert "\\x1b" in out and "\\u202e" in out


def test_error_message_and_failure_are_sanitised():
    def agent(task, tools):
        tools.http_get("https://x.test/\x1b[31m\nFAKE")

    traj = run(agent, "t", World())
    with pytest.raises(EvalFailure) as ei:
        Expect(traj).finished().assert_all()
    assert "\x1b" not in str(ei.value)


def test_safe_leaves_normal_unicode_alone():
    assert safe("héllo — 日本語 ✓") == "héllo — 日本語 ✓"


def test_secrets_redacted_in_transcript_but_not_in_data():
    def agent(task, tools):
        try:
            tools.http_get("https://x.test", headers={"Authorization": "Bearer " + FAKE})
        except Exception:  # noqa: BLE001, S110
            pass
        tools.kv_set("api_key", FAKE)  # positional value: key name is data
        tools.http_post("https://x.test", json={"password": "hunter2hunter2"})

    traj = run(agent, "t", World())
    text = traj.transcript()
    assert FAKE not in text and "hunter2" not in text
    assert "«redacted»" in text
    assert traj.calls[0].kwargs["headers"]["Authorization"].endswith(FAKE)  # assertions still see it


def test_redact_forms():
    assert FAKE not in redact(f"Authorization: Bearer {FAKE}")
    assert FAKE not in redact(f"requests.get(url, api_key='{FAKE}')")
    assert FAKE not in redact(json.dumps({"x-api-key": FAKE}))
    assert redact("plain text, token count 5") == "plain text, token count 5"


def test_clean_is_linear_on_hostile_input():
    import time

    start = time.perf_counter()
    clean("'" * 200_000 + "api_key=" * 20_000 + "{'token': '" * 5_000)
    assert time.perf_counter() - start < 5


def test_huge_argument_is_truncated_in_transcript():
    traj = run(lambda t, x: x.kv_set("k", "A" * 500_000), "t", World())
    assert len(traj.transcript()) < 3000


# -- threads -----------------------------------------------------------------
def test_threaded_tool_calls_get_unique_step_numbers():
    def agent(task, tools):
        def work():
            for i in range(300):
                tools.kv_set(i, i)

        ts = [threading.Thread(target=work) for _ in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]

    traj = run(agent, "t", World(), max_steps=10_000)
    assert sorted(c.step for c in traj.calls) == list(range(1, 1801))


def test_max_steps_enforced_under_threads():
    def agent(task, tools):
        errs = []

        def work():
            try:
                for i in range(200):
                    tools.kv_set(i, i)
            except Exception as e:  # noqa: BLE001 - MaxStepsExceeded
                errs.append(e)

        ts = [threading.Thread(target=work) for _ in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]

    traj = run(agent, "t", World(), max_steps=50)
    assert traj.steps == 50


# -- safety-scope assertions -------------------------------------------------
def _agent_escape(task, tools):
    tools.fs_write("/out/ok.md", "1")
    tools.fs_write("/etc/passwd", "pwned")
    tools.fs_delete("/out/../etc/passwd")
    tools.http_get("https://api.example.com/v1")
    try:
        tools.http_get("https://api.example.com@evil.test/v1")
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        tools.http_request("POST", "https://evil.test/x")
    except Exception:  # noqa: BLE001, S110
        pass


def _world_all_http():
    w = World()
    w.http.mock("GET", "*", text="ok")
    w.http.mock("POST", "*", text="ok")
    return w


def test_fs_writes_confined_to_catches_escape_and_traversal():
    traj = run(_agent_escape, "t", _world_all_http())
    with pytest.raises(EvalFailure, match="outside scope"):
        Expect(traj).fs_writes_confined_to("/out/**").assert_all()
    ok = run(lambda t, x: x.fs_write("/out/a/b.md", "1"), "t", World())
    Expect(ok).fs_writes_confined_to("/out/**").assert_all()


def test_fs_writes_confined_fails_closed_on_relative_path():
    traj = Trajectory.from_openai_messages([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "function": {"name": "fs_write", "arguments": json.dumps({"path": "rel.txt"})}}]},
        {"role": "assistant", "content": "done"},
    ])
    with pytest.raises(EvalFailure, match="unresolvable"):
        Expect(traj).fs_writes_confined_to("/**").assert_all()


def test_http_confined_to_blocks_exfil_and_authority_spoof():
    traj = run(_agent_escape, "t", _world_all_http())
    with pytest.raises(EvalFailure) as ei:
        Expect(traj).http_confined_to("https://api.example.com/*").assert_all()
    msg = str(ei.value)
    assert "2 request(s) outside" in msg  # spoof + http_request POST evil


def test_always_preceded_by_requires_matching_successful_read():
    def sloppy(task, tools):
        tools.fs_write("/a", "1")
        tools.fs_write("/b", "1")
        tools.fs_read("/a")
        tools.fs_delete("/a")
        tools.fs_delete("/b")

    traj = run(sloppy, "t", World())
    with pytest.raises(EvalFailure, match="without prerequisite"):
        Expect(traj).always_preceded_by("fs_delete", "fs_read", on="path").assert_all()

    def careful(task, tools):
        tools.fs_write("/a", "1")
        tools.fs_read("/a")
        tools.fs_delete("/a")

    Expect(run(careful, "t", World())).always_preceded_by("fs_delete", "fs_read").assert_all()


def test_always_preceded_ignores_failed_prerequisite():
    def agent(task, tools):
        try:
            tools.fs_read("/missing")
        except FileNotFoundError:
            pass
        tools.fs_write("/missing", "x")
        tools.fs_delete("/missing")

    traj = run(agent, "t", World())
    with pytest.raises(EvalFailure):
        Expect(traj).always_preceded_by("fs_delete", "fs_read").assert_all()


def test_only_called_no_repeats_no_errors():
    def looper(task, tools):
        for _ in range(4):
            tools.kv_set("a", 1)
        try:
            tools.fs_read("/nope")
        except FileNotFoundError:
            pass

    traj = run(looper, "t", World())
    with pytest.raises(EvalFailure) as ei:
        (Expect(traj).only_called("kv_set").no_repeated_calls(2).no_tool_errors().assert_all())
    msg = str(ei.value)
    assert "outside allowlist" in msg and "repeated 4x" in msg and "failed call" in msg
    Expect(traj).only_called("kv_set", "fs_read").no_repeated_calls(4).assert_all()


# -- trace import / export ---------------------------------------------------
def test_roundtrip_json(tmp_path):
    traj = run(lambda t, x: (x.fs_write("/a", "1"), x.note("hi"), "done")[2], "t", World())
    p = tmp_path / "t.json"
    traj.save(p)
    back = Trajectory.load(p)
    assert back.tool_names == ["fs_write"] and back.final_answer == "done"
    assert back.notes[0].text == "hi" and back.world is None
    Expect(back).called("fs_write", path="/a").never_called("fs_delete").assert_all()


def test_world_assertions_refuse_imported_traces():
    back = Trajectory.from_dict(run(lambda t, x: "ok", "t", World()).to_dict())
    with pytest.raises(ValueError, match="mock world"):
        Expect(back).world_file("/x").assert_all()


@pytest.mark.parametrize("bad", [
    [], {"schema": 2, "calls": []}, {"schema": 1, "calls": "x"},
    {"schema": 1, "calls": [{"name": 5}]}, {"schema": 1, "calls": [{"name": "a", "args": {}}]},
    {"schema": 1, "calls": [], "registered": [1]},
])
def test_malformed_native_trace_rejected(bad):
    with pytest.raises(ValueError):
        Trajectory.from_dict(bad)


def test_non_json_values_degrade_to_repr():
    traj = run(lambda t, x: x.fs_write("/b", b"\x00\x01"), "t", World())
    json.dumps(traj.to_dict())


OPENAI = [
    {"role": "user", "content": "clean up"},
    {"role": "assistant", "content": "I will only read.", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "fs_read", "arguments": '{"path": "/a"}'}},
        {"id": "c2", "type": "function", "function": {"name": "fs_delete", "arguments": '{"path": "/a"}'}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "data"},
    {"role": "tool", "tool_call_id": "c2", "content": "deleted"},
    {"role": "assistant", "content": "All done."},
]


def test_openai_import_and_narration_vs_action():
    traj = Trajectory.from_openai_messages(OPENAI, tools=["fs_read", "fs_delete"])
    assert traj.finished and traj.final_answer == "All done."
    assert traj.notes[0].text == "I will only read."
    assert traj.calls[0].result_repr == "'data'"
    with pytest.raises(EvalFailure):
        Expect(traj).notes_contain("only read").never_called("fs_delete").assert_all()
    Expect(traj).in_order("fs_read", "fs_delete").always_preceded_by("fs_delete", "fs_read").assert_all()


def test_openai_malformed_arguments_are_flagged_not_guessed():
    msgs = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "function": {"name": "x", "arguments": "{not json"}}]}]
    traj = Trajectory.from_openai_messages(msgs)
    assert traj.calls[0].error and not traj.finished
    with pytest.raises(EvalFailure):
        Expect(traj).no_tool_errors().assert_all()


def test_trace_ending_on_tool_use_is_unfinished():
    traj = Trajectory.from_openai_messages(OPENAI[:3])
    assert not traj.finished and "without a final" in traj.failure


ANTHROPIC = [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": [
        {"type": "text", "text": "Reading first."},
        {"type": "tool_use", "id": "t1", "name": "fs_read", "input": {"path": "/a"}}]},
    {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "boom"}], "is_error": True}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Could not read."}]},
]


def test_anthropic_import_marks_errors():
    traj = Trajectory.from_anthropic_messages(ANTHROPIC)
    assert traj.finished and traj.calls[0].error and traj.notes[0].text == "Reading first."
    assert traj.final_answer == "Could not read."


def test_import_rejects_garbage():
    for fn in (Trajectory.from_openai_messages, Trajectory.from_anthropic_messages):
        with pytest.raises(ValueError):
            fn("nope")
        with pytest.raises(ValueError):
            fn([1, 2])
    with pytest.raises(ValueError):
        Trajectory.from_openai_messages([{"role": "assistant", "tool_calls": [{"function": {}}]}])


# -- pass rate ---------------------------------------------------------------
def test_pass_rate_and_threshold():
    def agent(task, tools, seed=0):
        if seed % 4 == 0:
            tools.fs_delete if False else tools.kv_set("bad", 1)
        return "ok"

    trajs = sample(agent, "t", World, n=8)

    def build(t):
        return Expect(t).never_called("kv_set")

    res = pass_rate(trajs, build)
    assert (res.passed, res.total) == (6, 8) and "75%" in str(res)
    assert_pass_rate(trajs, build, min_rate=0.75)
    with pytest.raises(EvalFailure, match=r"failing runs: \[0, 4\]"):
        assert_pass_rate(trajs, build)  # default demands every run


def test_pass_rate_empty_expect_fails_and_zero_trajs_raise():
    trajs = sample(lambda t, x: "ok", "t", World, n=2)
    assert pass_rate(trajs, lambda t: Expect(t)).passed == 0
    with pytest.raises(EvalFailure):
        assert_pass_rate([], lambda t: Expect(t))
    with pytest.raises(ValueError):
        assert_pass_rate(trajs, lambda t: Expect(t), min_rate=2)


# -- CLI audit ---------------------------------------------------------------
def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def test_cli_audit_clean_and_dirty(tmp_path, capsys):
    good = _write(tmp_path, "g.json", OPENAI[:1] + [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "1", "function": {"name": "fs_read", "arguments": '{"path": "/a"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "x"},
        {"role": "assistant", "content": "ok"}])
    bad = _write(tmp_path, "b.json", OPENAI)
    assert main(["audit", good, "--allow-tools", "fs_read"]) == 0
    assert main(["audit", good, bad, "--allow-tools", "fs_read"]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "outside allowlist" in out
    assert main(["audit", bad, "--must-precede", "fs_delete:fs_read:path"]) == 0


def test_cli_audit_requires_a_rule_and_valid_input(tmp_path, capsys):
    f = _write(tmp_path, "a.json", OPENAI)
    assert main(["audit", f]) == 2
    assert main(["audit", str(tmp_path / "missing.json"), "--no-tool-errors"]) == 2
    (tmp_path / "junk.json").write_text("{not json")
    assert main(["audit", str(tmp_path / "junk.json"), "--no-tool-errors"]) == 2
    assert main(["audit", f, "--must-precede", "bad"]) == 2


def test_cli_audit_native_and_anthropic_formats(tmp_path):
    native = tmp_path / "n.json"
    run(lambda t, x: x.fs_write("/out/a", "1"), "t", World()).save(native)
    assert main(["audit", str(native), "--fs-write-allow", "/out/**"]) == 0
    assert main(["audit", str(native), "--fs-write-allow", "/tmp/**"]) == 1
    an = _write(tmp_path, "an.json", ANTHROPIC)
    assert main(["audit", an, "--no-tool-errors"]) == 1


def test_cli_audit_rejects_oversized_trace(tmp_path, monkeypatch):
    from trajtest import traces

    monkeypatch.setattr(traces, "MAX_FILE_BYTES", 10)
    f = _write(tmp_path, "a.json", OPENAI)
    assert main(["audit", f, "--no-tool-errors"]) == 2


def test_cli_run_has_no_assert_rewrite_warning(tmp_path, capfd):
    f = tmp_path / "test_ok.py"
    f.write_text("def test_ok():\n    assert True\n")
    assert main(["run", str(f)]) == 0
    assert "cannot be rewritten" not in capfd.readouterr().out
