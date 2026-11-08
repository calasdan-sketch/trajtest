"""Round 5: pin author, license and version so they cannot drift."""

import pathlib
import re
import subprocess
import sys

import trajtest

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUTHOR = "Calas Automations Inc."
pytestmark = __import__("pytest").mark.skipif(
    not (ROOT / "pyproject.toml").exists(), reason="not running from a source checkout")


def _pyproject():
    return (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_author_string_is_pinned_everywhere():
    assert trajtest.__author__ == AUTHOR
    text = _pyproject()
    assert f'authors = [{{ name = "{AUTHOR}" }}]' in text
    assert f'maintainers = [{{ name = "{AUTHOR}" }}]' in text
    assert f"Copyright (c) 2026 {AUTHOR}" in (ROOT / "LICENSE").read_text(encoding="utf-8")


def test_no_stale_author_variants():
    for f in [ROOT / "pyproject.toml", ROOT / "LICENSE", ROOT / "README.md", ROOT / "CHANGELOG.md",
              *(ROOT / "src" / "trajtest").glob("*.py")]:
        body = f.read_text(encoding="utf-8")
        assert not re.search(r"Calas Automations(?! Inc)\b(?!\.com)", body), f
        assert "Dan Calas" not in body and "calasdan" not in body, f


def test_version_consistent_between_package_and_pyproject_and_changelog():
    m = re.search(r'^version = "([^"]+)"', _pyproject(), re.MULTILINE)
    assert m and m.group(1) == trajtest.__version__
    assert f"## {trajtest.__version__}" in (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")


def test_cli_version_reports_version_and_author():
    out = subprocess.run([sys.executable, "-m", "trajtest", "--version"],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == f"trajtest {trajtest.__version__} ({AUTHOR})"
