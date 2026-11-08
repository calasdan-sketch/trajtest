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
