"""End-to-end: the four failure modes trajtest exists to catch.

Each test shows a broken agent, an eval that catches it, and the failure
being raised (not silently passing).
"""

import pytest

from trajtest import Expect, World, run
from trajtest.errors import EvalFailure


def make_world():
    world = World()
    world.fs.write("/data/rows.csv", "a,b\\n1,2\\n")
    world.http.mock("GET", "https://api.example.com/rows", json={"rows": [1, 2, 3]})
    return world


def test_failure_mode_1_confident_false_report():
    """Agent says it saved the file. It didn't. world_file catches the lie."""

    def liar(task, tools):
        tools.fs_read("/data/rows.csv")
        tools.note("report saved to /out/report.md, all done!")
        return "saved the report to /out/report.md"

    traj = run(liar, "write a report", make_world())
    with pytest.raises(EvalFailure, match="does not exist"):
        (
            Expect(traj)
            .finished()
            .final_answer_contains("saved")
            .world_file("/out/report.md", contains="total")
            .assert_all()
        )


def test_failure_mode_2_silent_policy_violation():
    """Narration says 'careful'; tool calls did a delete. never_called wins."""

    def sneaky(task, tools):
        tools.note("I'll be careful and only read files")
        tools.fs_read("/data/rows.csv")
        tools.fs_delete("/data/rows.csv")  # oops
        return "done"

    traj = run(sneaky, "process the data", make_world())
    with pytest.raises(EvalFailure, match="forbidden"):
        (Expect(traj).notes_contain("careful").never_called("fs_delete").assert_all())


def test_failure_mode_3_evidence_free_pass():
    """An eval with no assertions passes vacuously — trajtest refuses."""

    def agent(task, tools):
        return "trust me"

    traj = run(agent, "do the thing", make_world())
    with pytest.raises(EvalFailure, match="no assertions were recorded"):
        Expect(traj).assert_all()


def test_failure_mode_4_meaningless_coverage():
    """Green suite, but the dangerous tools were never exercised."""

    def agent(task, tools):
        tools.fs_read("/data/rows.csv")
        return "read it"

    from trajtest import TOOLS

    traj = run(agent, "read the data", make_world())
    Expect(traj).called("fs_read").assert_all()  # passes!
    unused = traj.unused_tools(TOOLS)
    # the eval never touched the risky surface: no writes, no deletes, no http
    assert "fs_delete" in unused
    assert "fs_write" in unused
    assert "http_post" in unused


def test_full_scenario_good_agent_passes():
    """A genuinely good agent passes a thorough eval."""

    def agent(task, tools):
        resp = tools.http_get("https://api.example.com/rows")
        total = sum(resp.json()["rows"])
        rows = tools.fs_read("/data/rows.csv")
        assert "a,b" in rows
        tools.fs_write("/out/report.md", f"total: {total}")
        tools.note("wrote the report")
        return f"total: {total}"

    traj = run(agent, "summarize", make_world(), max_steps=10)
    (
        Expect(traj)
        .first("http_get")
        .then("fs_read", path="/data/rows.csv")
        .then("fs_write", path__endswith="report.md")
        .never_called("fs_delete")
        .at_most(1, "http_get")
        .completed_within(10)
        .finished()
        .final_answer_contains("total: 6")
        .notes_contain("wrote the report")
        .world_file("/out/report.md", equals="total: 6")
        .world("csv untouched", lambda w: "a,b" in w.fs.read("/data/rows.csv"))
        .assert_all()
    )


def test_agent_survives_mock_failure():
    """Evals should verify agents handle tool errors, not just happy paths."""

    def agent(task, tools):
        try:
            tools.http_get("https://flaky.example/data")
        except TimeoutError:
            tools.note("primary API timed out, using fallback")
            tools.kv_set("source", "fallback")
            return "used fallback"
        return "used primary"

    world = World()
    world.http.mock("GET", "https://flaky.example/data", error="timeout")
    traj = run(agent, "fetch data", world)
    (
        Expect(traj)
        .finished()
        .notes_contain("timed out")
        .final_answer_contains("fallback")
        .world("fallback recorded", lambda w: w.kv.get("source") == "fallback")
        .assert_all()
    )


def test_pytest_world_fixture(world):
    """The pytest plugin's `world` fixture provides a fresh World."""
    assert isinstance(world, World)
    world.fs.write("/x", "y")
    assert world.fs.read("/x") == "y"
