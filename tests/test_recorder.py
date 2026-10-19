"""Tests for the trajectory recorder: run(), sample(), ToolCall, Trajectory."""

import pytest

from trajtest.errors import MaxStepsExceeded
from trajtest.recorder import _Tools, run, sample
from trajtest.world import World


def good_agent(task, tools):
    tools.http_get("https://api.example.com/rows")
    tools.fs_write("/out/report.md", "total: 6")
    tools.note("saved the report")
    return "done"


def make_world():
    world = World()
    world.http.mock("GET", "https://api.example.com/rows", json={"rows": [1, 2, 3]})
    return world


def test_run_records_calls_in_order():
    traj = run(good_agent, "task", make_world())
    assert traj.finished
    assert traj.final_answer == "done"
    assert traj.tool_names == ["http_get", "fs_write"]
    assert traj.steps == 2


def test_tool_call_details():
    traj = run(good_agent, "task", make_world())
    call = traj.calls[0]
    assert call.step == 1
    assert call.name == "http_get"
    assert call.args == ("https://api.example.com/rows",)
    assert call.duration_ms >= 0
    assert call.error is None


def test_notes_are_not_steps():
    traj = run(good_agent, "task", make_world())
    assert len(traj.notes) == 1
    assert traj.notes[0].text == "saved the report"
    assert traj.notes[0].step == 2  # written after 2 tool calls
    assert traj.steps == 2  # notes don't inflate the step budget


def test_world_attached_to_trajectory():
    world = make_world()
    traj = run(good_agent, "task", world)
    assert traj.world is world


def test_agent_exception_marks_unfinished():
    def boom(task, tools):
        tools.note("starting")
        raise RuntimeError("agent gave up")

    traj = run(boom, "task", World())
    assert not traj.finished
    assert "RuntimeError" in traj.failure
    assert traj.steps == 0  # the note isn't a step; no tool was called


def test_failed_tool_call_recorded_with_error():
    def boom(task, tools):
        tools.fs_read("/missing.txt")

    traj = run(boom, "task", World())
    assert not traj.finished
    assert "FileNotFoundError" in traj.failure
    # the failed tool call is still recorded, with its error
    assert traj.calls[0].name == "fs_read"
    assert "FileNotFoundError" in traj.calls[0].error


def test_max_steps_enforced():
    def loopy(task, tools):
        while True:
            tools.kv_set("n", 1)

    traj = run(loopy, "task", World(), max_steps=5)
    assert not traj.finished
    assert "max_steps=5" in traj.failure
    assert traj.steps == 5


def test_max_steps_invalid():
    with pytest.raises(ValueError):
        run(good_agent, "task", World(), max_steps=0)


def test_seed_injected_when_accepted():
    seen = {}

    def seeded(task, tools, seed=None):
        seen["seed"] = seed
        return "ok"

    traj = run(seeded, "task", World(), seed=42)
    assert traj.finished
    assert seen["seed"] == 42


def test_seed_rejected_when_not_accepted():
    with pytest.raises(TypeError, match="seed"):
        run(good_agent, "task", World(), seed=1)


def test_tool_errors_propagate_to_agent_but_are_recorded():
    handled = {}

    def resilient(task, tools):
        try:
            tools.http_get("https://flaky.example/")
        except TimeoutError:
            handled["yes"] = True
            tools.note("request timed out, continuing")
        return "recovered"

    world = World()
    world.http.mock("GET", "https://flaky.example/", error="timeout")
    traj = run(resilient, "task", world)
    assert traj.finished
    assert handled["yes"] is True
    assert traj.calls[0].error.startswith("TimeoutError")


def test_unknown_tool_raises_attribute_error():
    traj_holder = {}

    def sneaky(task, tools):
        traj_holder["tools"] = tools
        return "x"

    run(sneaky, "task", World())
    with pytest.raises(AttributeError):
        traj_holder["tools"].launch_missiles()


def test_result_repr_truncated():
    def big(task, tools):
        tools.kv_set("k", "x" * 10000)
        return "x"

    traj = run(big, "task", World())
    assert len(traj.calls[0].result_repr) <= 501


def test_unused_tools():
    from trajtest import TOOLS

    traj = run(good_agent, "task", make_world())
    assert set(TOOLS) == set(_Tools.REGISTERED)
    unused = traj.unused_tools(TOOLS)
    assert "http_get" not in unused
    assert "fs_write" not in unused
    assert "fs_delete" in unused


def test_transcript_format():
    traj = run(good_agent, "task", make_world())
    text = traj.transcript()
    assert "#1 http_get('https://api.example.com/rows')" in text
    assert "(note @ step 2) saved the report" in text
    assert "finished=True" in text


def test_sample_runs_n_fresh_worlds():
    worlds = []

    def factory():
        w = make_world()
        worlds.append(w)
        return w

    def agent(task, tools, seed=None):
        tools.kv_set("seed", seed)
        return f"s{seed}"

    trajs = sample(agent, "task", factory, n=3, seed_start=10)
    assert len(trajs) == 3
    assert len({id(w) for w in worlds}) == 3  # fresh world each run
    assert [t.final_answer for t in trajs] == ["s10", "s11", "s12"]
    # each trajectory sees only its own world's state
    assert trajs[0].world.kv.get("seed") == 10


def test_sample_invalid_n():
    with pytest.raises(ValueError):
        sample(good_agent, "task", make_world, n=0)


def test_sample_world_factory_called_each_time():
    calls = []

    def factory():
        calls.append(1)
        return make_world()

    # agents without a seed kwarg still work: seeds are simply not passed
    sample(good_agent, "task", factory, n=2)
    assert len(calls) == 2


def test_sample_seeds_only_when_accepted():
    seen = []

    def seeded(task, tools, seed=None):
        seen.append(seed)
        return "ok"

    sample(seeded, "task", make_world, n=2, seed_start=5)
    assert seen == [5, 6]

    seen_plain = []
    sample(lambda t, tools: seen_plain.append("ran") or "ok", "task", make_world, n=2)
    assert seen_plain == ["ran", "ran"]


def test_max_steps_exceeded_directly_is_exception():
    # MaxStepsExceeded is an Exception (agents could theoretically catch it)
    assert issubclass(MaxStepsExceeded, Exception)
