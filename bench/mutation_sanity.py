"""Mutation-style sanity check: break key conditions in a COPY of the source
and confirm the test suite notices. Usage: python bench/mutation_sanity.py

Not part of the normal test run (it re-runs the suite once per mutation, ~1 min
each batch). A surviving mutant means a rule can be silently wrong while all
tests pass: add a test. Exit 0 only if every mutant is killed.
"""

import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (file, exact old text, replacement). Each old text must occur exactly once.
MUTANTS = [
    ("expect.py", "steps <= n,", "steps < n,"),
    ("expect.py", "ok = self.traj.finished and not self.traj.failure", "ok = self.traj.finished"),
    ("expect.py", "bad = [v for v in seen.values() if len(v) > max_identical]",
     "bad = [v for v in seen.values() if len(v) >= max_identical + 2]"),
    ("expect.py", "errs = [c for c in self.traj.calls if c.error]", "errs = []"),
    ("expect.py", "if not hits:\n            return self._record(True, desc)\n        shown",
     "if hits:\n            return self._record(True, desc)\n        shown"),
    ("expect.py", 'if not c.name.startswith("http_"):\n                continue\n            for url in _arg_all(c, "url", _url_pos(c)):\n                if not _host_ok',
     'if not c.name.startswith("http_"):\n                continue\n            for url in _arg_all(c, "url", _url_pos(c)):\n                if False and not _host_ok'),
    ("world.py", 'if part == "..":\n            if parts:\n                parts.pop()\n            continue',
     'if part == "..":\n            continue'),
    ("world.py", 'if "\\x00" in path:', 'if False:'),
    ("world.py", "self._used -= _size(self._files[path])", "pass"),
    ("sanitize.py", "if len(text) <= limit:\n        return text", "if True:\n        return text"),
    ("sanitize.py", '"Cc", "Cf", "Zl", "Zp", "Cs", "Co", "Cn"', '"Cc", "Zl", "Zp", "Cs", "Co", "Cn"'),
    ("sanitize.py", "text = _BEARER.sub(", "text = text if True else _BEARER.sub("),
    ("traces.py", "os.replace(tmp, path)", "os.rename(tmp, path + '.x')"),
    ("cli.py", "worst = max(worst, 1 if bad else 0)", "worst = 0"),
    ("cli.py", 'err("trajtest audit: no rules given; refusing to pass vacuously")\n        return 2',
     'err("trajtest audit: no rules given; refusing to pass vacuously")\n        return 0'),
]


def run_one(m):
    fname, old, new = m
    tmp = tempfile.mkdtemp(prefix="trajtest-mut-")
    try:
        dst = os.path.join(tmp, "repo")
        shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", "*.egg-info", ".ruff_cache", ".pytest_cache", "dist", "build"))
        path = os.path.join(dst, "src", "trajtest", fname)
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        if src.count(old) != 1:
            return m, "STALE"
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(src.replace(old, new))
        env = {**os.environ, "PYTHONPATH": os.path.join(dst, "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider",
                            os.path.join(dst, "tests")], cwd=dst, env=env,
                           capture_output=True, text=True, check=False)
        return m, ("killed" if r.returncode != 0 else "SURVIVED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    from concurrent.futures import ThreadPoolExecutor

    workers = int(os.environ.get("MUTATION_JOBS", "4"))
    only = sys.argv[1:]  # optional substrings: run just the matching mutants
    todo = [m for m in MUTANTS if not only or any(o in m[0] + m[1] for o in only)]
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(run_one, todo))
    survivors = 0
    for (fname, old, _), status in results:
        print(f"{status:8} {fname}: {old[:60]!r}")
        survivors += status != "killed"
    print(f"{len(todo) - survivors}/{len(todo)} mutants killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
