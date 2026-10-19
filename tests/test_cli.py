"""Tests for the CLI: init scaffolding and run passthrough."""

import subprocess
import sys

import pytest

from trajtest.cli import main

VENV_PY = sys.executable


def test_init_creates_missing_dir(tmp_path):
    target_dir = tmp_path / "newdir" / "nested"
    assert main(["init", "--dir", str(target_dir)]) == 0
    assert (target_dir / "test_trajtest_example.py").exists()


def test_init_scaffolds(tmp_path):
    code = main(["init", "--dir", str(tmp_path)])
    assert code == 0
    target = tmp_path / "test_trajtest_example.py"
    assert target.exists()
    text = target.read_text()
    assert "from trajtest import EvalFailure, Expect, World, run" in text
    assert "def test_agent_fetches_then_saves" in text
    assert "def test_agent_does_not_lie_about_saving" in text


def test_init_refuses_overwrite(tmp_path):
    assert main(["init", "--dir", str(tmp_path)]) == 0
    assert main(["init", "--dir", str(tmp_path)]) == 1
    assert main(["init", "--dir", str(tmp_path), "--force"]) == 0


def test_run_passes_on_example(tmp_path):
    """The whole scaffold file runs green: the liar demo *expects* the
    EvalFailure, so `init && run` is a clean first experience."""
    assert main(["init", "--dir", str(tmp_path)]) == 0
    code = main(["run", str(tmp_path / "test_trajtest_example.py")])
    assert code == 0


def test_run_reports_failure(tmp_path):
    target = tmp_path / "test_bad.py"
    target.write_text(
        "from trajtest import Expect, World, run\n"
        "\n"
        "def test_nope(world):\n"
        "    traj = run(lambda t, tools: 'x', 't', world)\n"
        "    Expect(traj).called('http_get').assert_all()\n"
    )
    code = main(["run", str(target)])
    assert code != 0


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "trajtest" in capsys.readouterr().out


def test_scaffold_tests_execute_under_pytest(tmp_path):
    """The scaffolded file runs green under real pytest (both tests)."""
    assert main(["init", "--dir", str(tmp_path)]) == 0
    result = subprocess.run(
        [
            VENV_PY,
            "-m",
            "pytest",
            str(tmp_path / "test_trajtest_example.py"),
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_scaffold_world_signature_order(tmp_path):
    """Regression: the scaffold must call world(description, predicate) in
    the right order — swapped args used to raise TypeError: 'str' object
    is not callable instead of demonstrating the liar catch."""
    assert main(["init", "--dir", str(tmp_path)]) == 0
    text = (tmp_path / "test_trajtest_example.py").read_text()
    assert ".world(lambda" not in text  # lambda must not be the 1st arg
    assert "pytest.raises(EvalFailure)" in text
