"""Tests for the Expect assertion DSL."""

import pytest

from trajtest import ANY, Contains, Expect, Regex, World, run
from trajtest.errors import EvalFailure
from trajtest.recorder import Trajectory


def agent(task, tools):
    tools.http_get("https://api.example.com/rows", headers={"A": "b"})
    tools.fs_read("/docs/in.md")
    tools.fs_write("/out/report.md", "total: 6")
    tools.note("all done, cleaned up nothing")
    return "total: 6"


@pytest.fixture
def traj():
    world = World()
    world.fs.write("/docs/in.md", "in")
    world.http.mock("GET", "https://api.example.com/rows", json={"rows": [1, 2, 3]})
    return run(agent, "task", world)


def test_called_basic(traj):
    Expect(traj).called("http_get").assert_all()  # no raise


def test_called_with_arg_matchers(traj):
    (
        Expect(traj)
        .called("http_get", "https://api.example.com/rows")
        .called("http_get", url__contains="api.example.com")
        .called("fs_write", "/out/report.md", Contains("total"))
        .called("http_get", headers=ANY)
        .assert_all()
    )


def test_called_failure_message(traj):
    with pytest.raises(EvalFailure) as exc:
        Expect(traj).called("fs_delete").assert_all()
    assert 'called("fs_delete")' in str(exc.value)
    assert "no call" in str(exc.value)


def test_called_prefix_positional_matching(traj):
    # extra actual args are fine; expected positionals are a prefix
    Expect(traj).called("http_get", "https://api.example.com/rows").assert_all()
    with pytest.raises(EvalFailure):
        Expect(traj).called("http_get", "https://wrong.example/").assert_all()


def test_called_with_alias(traj):
    Expect(traj).called_with("fs_read", "/docs/in.md").assert_all()


def test_called_times_at_least_at_most(traj):
    (
        Expect(traj)
        .called_times("http_get", 1)
        .at_least(1, "fs_write")
        .at_most(2, "fs_write")
        .assert_all()
    )
    with pytest.raises(EvalFailure):
        Expect(traj).called_times("http_get", 5).assert_all()
    with pytest.raises(EvalFailure):
        Expect(traj).at_most(0, "http_get").assert_all()


def test_never_called(traj):
    Expect(traj).never_called("fs_delete").assert_all()
    with pytest.raises(EvalFailure) as exc:
        Expect(traj).never_called("fs_write").assert_all()
    assert "forbidden call(s)" in str(exc.value)


def test_never_called_with_args(traj):
    # forbidding only specific args: other writes are fine
    Expect(traj).never_called("fs_write", "/etc/passwd").assert_all()
    with pytest.raises(EvalFailure):
        Expect(traj).never_called("fs_write", "/out/report.md").assert_all()


def test_in_order(traj):
    Expect(traj).in_order("http_get", "fs_read", "fs_write").assert_all()


def test_in_order_with_gaps_allowed():
    def gappy(task, tools):
        tools.note("starting")
        tools.http_get("https://x.example/")
        tools.kv_set("a", 1)
        tools.fs_write("/o", "x")

    world = World()
    world.http.mock("GET", "https://x.example/", text="t")
    t = run(gappy, "t", world)
    Expect(t).in_order("http_get", "fs_write").assert_all()  # gap ok


def test_in_order_failure(traj):
    with pytest.raises(EvalFailure) as exc:
        Expect(traj).in_order("fs_write", "http_get").assert_all()
    assert "in_order" in str(exc.value)


def test_in_order_with_kwarg_tuples(traj):
    (
        Expect(traj)
        .in_order("http_get", ("fs_write", {"path": "/out/report.md"}))
        .assert_all()
    )
    with pytest.raises(EvalFailure):
        (Expect(traj).in_order(("fs_write", {"path": "/wrong.md"})).assert_all())


def test_then_chain(traj):
    (
        Expect(traj)
        .then("http_get")
        .then("fs_write", path__endswith="report.md")
        .assert_all()
    )


def test_then_chain_failure_advances_cursor(traj):
    with pytest.raises(EvalFailure):
        (
            Expect(traj)
            .then("fs_write")
            .then("http_get")  # http_get happened BEFORE fs_write
            .assert_all()
        )


def test_then_immediately(traj):
    (
        Expect(traj)
        .then("http_get")
        .then_immediately("fs_read")
        .then_immediately("fs_write")
        .assert_all()
    )


def test_then_immediately_failure(traj):
    with pytest.raises(EvalFailure) as exc:
        (
            Expect(traj)
            .then("http_get")
            .then_immediately("fs_write")  # fs_read is in between
            .assert_all()
        )
    assert "immediately" in str(exc.value)


def test_first_last(traj):
    Expect(traj).first("http_get").last("fs_write").assert_all()
    with pytest.raises(EvalFailure):
        Expect(traj).first("fs_write").assert_all()
    with pytest.raises(EvalFailure):
        Expect(traj).last("http_get").assert_all()


def test_first_last_empty_trajectory():
    t = run(lambda task, tools: "nothing", "t", World())
    with pytest.raises(EvalFailure):
        Expect(t).first("http_get").assert_all()
    with pytest.raises(EvalFailure):
        Expect(t).last("http_get").assert_all()


def test_completed_within(traj):
    Expect(traj).completed_within(3).assert_all()
    Expect(traj).completed_within(99).assert_all()
    with pytest.raises(EvalFailure):
        Expect(traj).completed_within(2).assert_all()


def test_finished(traj):
    Expect(traj).finished().assert_all()


def test_finished_failure():
    def boom(task, tools):
        raise RuntimeError("kaput")

    t = run(boom, "t", World())
    with pytest.raises(EvalFailure) as exc:
        Expect(t).finished().assert_all()
    assert "kaput" in str(exc.value)


def test_final_answer(traj):
    Expect(traj).final_answer("total: 6").assert_all()
    Expect(traj).final_answer(Regex(r"total: \d+")).assert_all()
    Expect(traj).final_answer_contains("total").assert_all()
    with pytest.raises(EvalFailure):
        Expect(traj).final_answer("wrong").assert_all()


def test_notes(traj):
    Expect(traj).notes_contain("cleaned up").assert_all()
    Expect(traj).notes_match(r"all done").assert_all()
    with pytest.raises(EvalFailure):
        Expect(traj).notes_contain("deleted everything").assert_all()


def test_world_file(traj):
    (
        Expect(traj)
        .world_file("/out/report.md")
        .world_file("/out/report.md", contains="total")
        .world_file("/out/report.md", equals="total: 6")
        .world_file("/out/report.md", matches=Regex(r"total: \d+"))
        .assert_all()
    )


def test_world_file_missing(traj):
    with pytest.raises(EvalFailure) as exc:
        Expect(traj).world_file("/out/nope.md").assert_all()
    assert "does not exist" in str(exc.value)


def test_world_file_content_mismatch(traj):
    with pytest.raises(EvalFailure) as exc:
        Expect(traj).world_file("/out/report.md", contains="WRONG").assert_all()
    assert "file content was" in str(exc.value)


def test_world_file_multiple_checks_rejected(traj):
    with pytest.raises(ValueError):
        Expect(traj).world_file("/out/report.md", contains="a", equals="b")


def test_world_predicate(traj):
    (
        Expect(traj)
        .world("report exists", lambda w: w.fs.exists("/out/report.md"))
        .world("http was hit", lambda w: (len(w.http.requests) == 1, "no requests"))
        .assert_all()
    )
    with pytest.raises(EvalFailure) as exc:
        Expect(traj).world("bad", lambda w: False).assert_all()
    assert "world(bad)" in str(exc.value)


def test_world_predicate_tuple_message(traj):
    with pytest.raises(EvalFailure) as exc:
        Expect(traj).world("x", lambda w: (False, "custom reason")).assert_all()
    assert "custom reason" in str(exc.value)


def test_custom(traj):
    (Expect(traj).custom("few steps", lambda t: t.steps <= 5).assert_all())
    with pytest.raises(EvalFailure):
        Expect(traj).custom("one step", lambda t: t.steps == 1).assert_all()


def test_world_assertion_without_world():
    t = Trajectory()  # hand-built, no world attached
    with pytest.raises(ValueError, match="mock world"):
        Expect(t).world_file("/x")


def test_fail_collecting_reports_all_violations(traj):
    with pytest.raises(EvalFailure) as exc:
        (
            Expect(traj)
            .called("http_get")  # passes
            .called("fs_delete")  # fails
            .never_called("fs_write")  # fails
            .completed_within(1)  # fails
            .assert_all()
        )
    text = str(exc.value)
    assert "3 of 4 assertion(s) failed" in text
    assert '✓ called("http_get")' in text
    assert '✗ called("fs_delete")' in text
    assert '✗ never_called("fs_write")' in text
    assert "Trajectory (3 steps):" in text
    assert "#1 http_get" in text


def test_check_does_not_raise(traj):
    violations = Expect(traj).called("nope_tool").check()
    assert len(violations) == 1
    assert not violations[0].ok
    assert "nope_tool" in violations[0].description


def test_check_flags_zero_assertions(traj):
    violations = Expect(traj).check()
    assert len(violations) == 1
    assert not violations[0].ok
    assert "no assertions" in violations[0].description


def test_bound_args_match_positional_by_name(traj):
    # agent passed url/path positionally; kwarg-style matching still works
    (
        Expect(traj)
        .called("http_get", url="https://api.example.com/rows")
        .called("fs_write", path="/out/report.md", content__contains="total")
        .assert_all()
    )


def test_empty_eval_fails_loudly(traj):
    with pytest.raises(EvalFailure, match="no assertions were recorded"):
        Expect(traj).assert_all()


def test_summary_marks(traj):
    exp = Expect(traj).called("http_get").called("nope")
    summary = exp.summary()
    assert "✓" in summary and "✗" in summary


def test_chaining_returns_self(traj):
    exp = Expect(traj)
    assert exp.called("http_get") is exp
    assert exp.then("http_get") is exp
