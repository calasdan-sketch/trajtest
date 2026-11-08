"""load_trace / save_trace: assert on redbg recordings, export to redbg."""

import gzip
import shutil
from pathlib import Path

import pytest

from trajtest import Expect, load_trace
from trajtest.errors import EvalFailure

SAMPLE = Path(__file__).parent / "fixtures" / "redbg-sample-a.jsonl"


def test_load_trace_maps_steps_and_answer():
    traj = load_trace(SAMPLE)
    assert traj.tool_names == ["read_file", "write_file"]
    assert traj.calls[0].kwargs == {"path": "main.py"}
    assert traj.calls[0].error is None
    assert "print" in traj.calls[0].result_repr
    assert traj.final_answer == "main.py defines main(), which prints 'hello'."
    assert traj.finished is True and traj.failure is None


def test_assertions_work_on_loaded_trace():
    traj = load_trace(SAMPLE)
    (
        Expect(traj)
        .called("read_file", path="main.py")
        .in_order("read_file", "write_file")
        .never_called("delete_file")
        .assert_all()
    )
    with pytest.raises(EvalFailure):
        Expect(traj).never_called("write_file").assert_all()


def test_world_file_on_loaded_trace_fails_clearly():
    traj = load_trace(SAMPLE)
    with pytest.raises(EvalFailure, match="world_file needs a mock World; this "
                       "trajectory was loaded from a recording"):
        Expect(traj).world_file("/summary.txt")


def test_gz_trace_loads_identically(tmp_path):
    gz = tmp_path / "a.jsonl.gz"
    with open(SAMPLE, "rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    a, b = load_trace(SAMPLE), load_trace(gz)
    assert [(c.name, c.kwargs, c.result_repr) for c in a.calls] == [
        (c.name, c.kwargs, c.result_repr) for c in b.calls
    ]
    assert a.final_answer == b.final_answer


def test_failed_and_unanswered_calls(tmp_path):
    p = tmp_path / "t.jsonl"
    base = '"v":1,"ts":"t","mono":0'
    p.write_text("\n".join([
        f'{{{base},"id":"s","type":"session_start","agent":"x"}}',
        f'{{{base},"id":"c1","type":"tool_call","step":1,"tool":"http_get","args":{{"url":"u"}}}}',
        f'{{{base},"id":"r1","type":"tool_result","step":1,"call_id":"c1","tool":"http_get","ok":false,"error":"boom"}}',
        f'{{{base},"id":"c2","type":"tool_call","step":2,"tool":"fs_write","args":{{}}}}',
        f'{{{base},"id":"n","type":"note","text":"hmm"}}',
    ]) + "\n")
    traj = load_trace(p)
    assert traj.calls[0].error == "boom"
    assert traj.calls[1].error == "no result recorded"
    assert [n.text for n in traj.notes] == ["hmm"] and traj.notes[0].step == 2
    assert traj.finished is False and "session_end" in traj.failure


def _agent(task, tools):
    tools.http_get("https://api.example.com/rows")
    tools.note("got rows")
    tools.fs_write("/out/r.md", content="total: 6")
    try:
        tools.fs_read("/missing")
    except FileNotFoundError:
        pass
    return "total: 6"


def _run():
    from trajtest import World, run

    world = World()
    world.http.mock("GET", "https://api.example.com/rows", json={"rows": [1]})
    return run(_agent, "task", world)


def test_save_trace_round_trip(tmp_path):
    traj = _run()
    path = traj.save_trace(tmp_path / "run.jsonl")
    back = load_trace(path)
    assert back.tool_names == traj.tool_names == ["http_get", "fs_write", "fs_read"]
    assert [c.kwargs for c in back.calls] == [
        {"url": "https://api.example.com/rows"},
        {"path": "/out/r.md", "content": "total: 6"},
        {"path": "/missing"},
    ]
    assert back.calls[2].error and "FileNotFoundError" in back.calls[2].error
    assert back.final_answer == "total: 6" and back.finished
    assert [(n.step, n.text) for n in back.notes] == [(1, "got rows")]
    Expect(back).in_order("http_get", "fs_write").assert_all()


def test_save_trace_is_redbg_shaped_and_gz(tmp_path):
    import json

    path = _run().save_trace(tmp_path / "run.jsonl.gz")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh]
    start, end = events[0], events[-1]
    assert start["type"] == "session_start"
    assert start["agent"] == "trajtest" and start["source"] == "trajtest"
    assert end["type"] == "session_end" and end["status"] == "ok"
    for e in events:
        assert {"v", "id", "ts", "mono", "type"} <= set(e)
    results = [e for e in events if e["type"] == "tool_result"]
    calls = {e["id"] for e in events if e["type"] == "tool_call"}
    assert all(r["call_id"] in calls for r in results)
