"""Round 5: golden-file tests for every machine/human output format of the CLI.

Regenerate after an intentional format change with:
    TRAJTEST_UPDATE_GOLDEN=1 pytest tests/test_golden.py
and review the diff in git: these files are the output contract CI scripts
parse.
"""

import contextlib
import io
import os
import pathlib

import pytest

from trajtest import cli

GOLD = pathlib.Path(__file__).parent / "golden"

CASES = {
    "audit_fail_text": (["audit", "trace_a.json", "--http-host-allow", "ok.test",
                         "--fs-write-allow", "/out/*", "--no-tool-errors"], 1),
    "audit_ok_json": (["audit", "trace_b.json", "--fs-write-allow", "/out/*", "--json"], 0),
    "audit_fail_json": (["audit", "trace_a.json", "--no-secret-shapes", "--no-tool-errors", "--json"], 1),
    "diff_different": (["diff", "trace_b.json", "trace_a.json"], 1),
    "diff_same": (["diff", "trace_a.json", "trace_a.json"], 0),
    "diff_unordered_args": (["diff", "trace_b.json", "trace_a.json", "--unordered", "--args"], 1),
    "audit_verbose_ok": (["audit", "trace_b.json", "--max-repeats", "2", "-v"], 0),
}


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    cwd = os.getcwd()
    os.chdir(GOLD)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main(argv)
    finally:
        os.chdir(cwd)
    return rc, out.getvalue(), err.getvalue()


@pytest.mark.parametrize("name", sorted(CASES))
def test_cli_output_matches_golden(name):
    argv, want_rc = CASES[name]
    rc, out, err = _run(argv)
    assert rc == want_rc
    assert err == ""
    path = GOLD / f"{name}.txt"
    if os.environ.get("TRAJTEST_UPDATE_GOLDEN"):
        path.write_text(out, encoding="utf-8", newline="\n")
    assert out == path.read_text(encoding="utf-8"), f"output changed; see {path.name}"


def test_junit_matches_golden_and_is_wellformed(tmp_path):
    import xml.etree.ElementTree as ET

    j = tmp_path / "out.xml"
    rc, _, _ = _run(["audit", "trace_a.json", "--no-tool-errors", "--fs-write-allow", "/out/*",
                     "--junit", str(j)])
    assert rc == 1
    root = ET.fromstring(j.read_bytes())
    assert root.get("tests") == "2" and root.get("failures") == "2"
    path = GOLD / "audit.junit.xml"
    if os.environ.get("TRAJTEST_UPDATE_GOLDEN"):
        path.write_bytes(j.read_bytes())
    assert j.read_bytes() == path.read_bytes()


def test_junit_written_even_when_clean_and_no_temp_left(tmp_path):
    j = tmp_path / "ok.xml"
    rc, _, _ = _run(["audit", "trace_b.json", "--max-repeats", "2", "--junit", str(j)])
    assert rc == 0 and 'failures="0"' in j.read_text(encoding="utf-8")
    assert [p.name for p in tmp_path.iterdir()] == ["ok.xml"]


def test_junit_unwritable_path_is_plain_error_exit_2(tmp_path):
    rc, _, err = _run(["audit", "trace_b.json", "--max-repeats", "2",
                         "--junit", str(tmp_path / "no" / "such" / "dir.xml")])
    assert rc == 2 and "cannot write" in err and "Traceback" not in err
