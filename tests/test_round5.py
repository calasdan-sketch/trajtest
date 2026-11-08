"""Round 5 regressions found by running the CLI in hostile environments."""

import os
import pathlib
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
TRACE = os.path.join(HERE, "golden", "trace_b.json")


def _cli(*args, env=None, cwd=None):
    e = {**os.environ, **(env or {})}
    return subprocess.run([sys.executable, "-m", "trajtest", *args],
                          capture_output=True, text=True, env=e, cwd=cwd, check=False)


def test_ascii_console_does_not_crash_on_non_ascii_path(tmp_path):
    p = tmp_path / ("tr" + chr(0xE9) + " " + chr(0x202E) + ".json")
    p.write_bytes(pathlib.Path(TRACE).read_bytes())
    r = _cli("audit", str(p), "--max-repeats", "3",
             env={"PYTHONIOENCODING": "ascii", "PYTHONUTF8": "0"})
    assert r.returncode == 0, r.stderr
    assert "Traceback" not in r.stderr and "ok" in r.stdout


def test_init_writes_utf8_under_ascii_locale(tmp_path):
    r = _cli("init", "--dir", str(tmp_path / "sp ace"),
             env={"PYTHONUTF8": "0", "PYTHONIOENCODING": "ascii", "LC_ALL": "C"})
    assert r.returncode == 0, r.stderr
    body = (tmp_path / "sp ace" / "test_trajtest_example.py").read_bytes()
    body.decode("utf-8")
    assert b"\r\n" not in body


def test_init_unwritable_target_is_plain_error(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    r = _cli("init", "--dir", str(blocker / "sub"))
    assert r.returncode == 2 and "Traceback" not in r.stderr and "cannot" in r.stderr


def test_json_piped_into_closed_reader_is_quiet(tmp_path):
    big = tmp_path / "big.json"
    big.write_bytes(pathlib.Path(TRACE).read_bytes())
    args = [sys.executable, "-m", "trajtest", "audit", str(big), "--max-repeats", "3", "--json"]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdout.close()
    err = p.stderr.read().decode()
    p.wait()
    assert "Traceback" not in err


@pytest.mark.skipif(os.name == "nt", reason="posix permission semantics")
def test_audit_on_directory_or_fifo_like_path_is_plain_error(tmp_path):
    r = _cli("audit", str(tmp_path), "--max-repeats", "3")
    assert r.returncode == 2 and "Traceback" not in r.stderr


# -- boundary tests added after mutation-style sanity found them unpinned ----

def _traj(names, **kw):
    from trajtest import World, run

    def agent(task, tools):
        for n in names:
            tools.kv_set(n, 1)
        return "x"
    return run(agent, "t", World(), **kw)


def test_finished_fails_when_a_loaded_trace_carries_a_failure_even_if_marked_finished():
    from trajtest import Expect
    t = _traj(["a"])
    assert t.finished
    t.failure = "RuntimeError: boom"
    assert not all(v.ok for v in Expect(t).finished().check())


@pytest.mark.parametrize("reps,limit,ok", [(2, 2, True), (3, 2, False), (1, 1, True), (2, 1, False)])
def test_no_repeated_calls_boundary_is_exactly_max_identical(reps, limit, ok):
    from trajtest import Expect, World, run

    def agent(task, tools):
        for _ in range(reps):
            tools.kv_set("same", 1)
        return "x"
    t = run(agent, "t", World())
    assert all(v.ok for v in Expect(t).no_repeated_calls(limit).check()) is ok


@pytest.mark.parametrize("steps,limit,ok", [(3, 3, True), (4, 3, False), (2, 3, True)])
def test_completed_within_boundary(steps, limit, ok):
    from trajtest import Expect
    t = _traj(["k"] * steps)
    assert all(v.ok for v in Expect(t).completed_within(limit).check()) is ok


def test_bearer_and_basic_tokens_without_a_known_shape_are_masked():
    from trajtest.sanitize import redact
    out = redact("server said Bearer plainTok3n99xyz and Basic dXNlcjpwYXNz here")
    assert "plainTok3n99xyz" not in out and "dXNlcjpwYXNz" not in out


def test_clip_bounds_huge_text_before_regexes_run():
    from trajtest.sanitize import clean
    out = clean("x" * 5_000_000, 1000)
    assert len(out) < 1200 and "chars" in out
