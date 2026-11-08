# Round 5 notes (also covers the round-4 style "first hour" packaging pass, because docs/ROUND4.md did not exist)

Honest method note: no sub-agent tool was available in this session, so the
three specialists below were played in sequence by one reviewer. Every finding
names the command that proved it. Nothing was run on Windows.

## Specialists and findings

### LLM-agent evaluation researcher
- Re-checked the two strongest competitors read-only: `pip index versions agentevals` -> 0.0.9, `pip index versions deepeval` -> 4.3.3. Both match the README comparison table, so it is still current (2026-11-08). Their semantics beyond what the README already says were not re-tested.
- Gap vs agentevals: subset/superset trajectory match modes (`same_tools_as` only does strict/unordered). Not built (NOT-DONE row 12).
- Gap: no pass@k / pass^k helper on top of `pass_rate` (row 13).

### Test-methodology expert
- Property-based tests, `tests/test_properties.py` (hypothesis, derandomized, skipped if absent): 11 properties (golden diff reflexive / empty iff same sequence, save-load round trip, redact idempotent and removes `sk-` keys, loader only raises ValueError on random JSON, glob sorted/unique/existing, quota returns to zero, `never_called` agrees with recorded names, audit `--json` valid for hostile file names). Command: `pytest tests/test_properties.py`. Result: no product defect found. One wrong assumption in my own test (NBSP is not a control character; the escaper is right to leave it).
- Mutation-style sanity, `python bench/mutation_sanity.py` (copies the tree, flips 15 conditions, expects the suite to fail). First run: 13/15 killed. Survivors showed real test gaps: `no_repeated_calls` limit boundary, `finished()` ignoring a loaded trace's `failure`, `Bearer`/`Basic` masking only covered by other patterns. Added boundary/regression tests in `tests/test_round5.py`; one survivor (`\r` escape) was an equivalent mutant (still escaped via the control-char branch) and was replaced by a `clip()` mutant. Re-run of the changed/surviving mutants: all killed (15 of the final 15 killed across the two runs; the final set was not re-run in one piece because the machine was heavily loaded and a full run took about 20 minutes).
- Golden-file tests, `tests/test_golden.py`: 7 CLI outputs + JUnit XML pinned byte for byte in `tests/golden/`. `TRAJTEST_UPDATE_GOLDEN=1 pytest tests/test_golden.py` regenerates.
- Differential tests against a reference: none exist for this domain (no reference implementation of the trajectory assertions). Not done.

### CI integrator
- Missing: JUnit output for CI dashboards. Added `trajtest audit --junit FILE` (atomic write; exit 2 and a plain message if unwritable). Proof: `tests/test_golden.py::test_junit_*`.
- Missing: the CI workflow never built or installed the package. Added a `package` job (build, `twine check`, install the wheel into a clean venv, run `init` + `run`). Not run on GitHub; run locally in the equivalent steps below.
- Golden files would break on a Windows checkout with CRLF conversion (the trace SHA-256 is printed in output). Added `.gitattributes` (`tests/golden/* -text`).
- Real crashes on a narrow console, proven with `PYTHONIOENCODING=ascii trajtest audit "tré.json" --max-repeats 3` (UnicodeEncodeError traceback) and `LC_ALL=C PYTHONUTF8=0 trajtest init` (same, in the file write). Fixed (`backslashreplace` console fallback, UTF-8 + LF for files written). Also `audit --json | head` printed a BrokenPipe traceback and `init` into an unwritable path printed one; now quiet / exit 2 with a plain message. Tests: `tests/test_round5.py`.
- Lint drift: CI installs unpinned `ruff`; the current release enables far more rules than classic defaults (it flagged 1 existing line, fixed with a targeted `noqa`). Pinning ruff is recommended (row 17, NOT-DONE).

### Packaging / release engineer (first-hour pass)
- Baseline: 244 tests passed; `python bench/bench.py` run. Build: `python -m build --outdir <scratch>`; `twine check` PASSED for wheel and sdist.
- Defect: no `LICENSE` file anywhere although metadata said MIT. Added (`Copyright (c) 2026 Calas Automations Inc.`), SPDX `license = "MIT"` + `license-files` (build needs setuptools >= 77), classifiers, maintainers, `[dev]` extra. Proof: `unzip -p dist/*.whl '*METADATA'` shows `Author: Calas Automations Inc.`, `Maintainer: Calas Automations Inc.`, `License-Expression: MIT`, `License-File: LICENSE`; wheel contains `licenses/LICENSE`, `py.typed`, entry points.
- Defect: sdist lacked CHANGELOG and the golden files. `MANIFEST.in` added; the sdist's own test suite then passed against the installed wheel: 284 passed (`pytest` run from the unpacked sdist, wheel installed in a clean venv).
- Clean-venv install (not editable) on Python 3.13, 3.12 and 3.11, run from a directory whose name contains spaces and a non-ASCII letter: `trajtest --version` prints `trajtest 0.2.0 (Calas Automations Inc.)`; `init` + `run` pass. README's first code block was extracted and run with pytest from the clean venv: passed. Python 3.10 (the declared minimum) was not available: unverified.
- Dependencies: `pytest>=7` only; `import trajtest` works in a venv holding just the wheel plus pytest.
- Registry: `pip index versions trajtest` and `pip download trajtest` find nothing. 0.2.0 has not been published, or the name is not on the index this machine can see. README "Install" now says so and shows install-from-checkout. Registry upgrade test: skipped by instruction.
- Author line: all hits for old variants re-grepped across the tree; test `tests/test_packaging.py` pins the author in pyproject, `__author__`, LICENSE, `--version`, and forbids personal-name and handle variants in sources and docs. The only project URL is the company site `https://calasautomations.com`; no GitHub URL is placed in package metadata (the repository's visibility is not known to me, so nothing claims it is public).

### Static Windows review (not run on Windows)
- `os.replace` for atomic writes (traces.py, junit): fine on Windows when the target is not open elsewhere; a reader holding the file open could make it raise PermissionError, which surfaces as OSError -> exit 2 for `--junit`.
- Console encoding: fixed (see above). File encodings: all explicit UTF-8 now. No `shell=True`, no `signal` use.
- Paths: the mock FS is deliberately POSIX; Windows-style paths give a hint message. Trace file paths use `open()`/`os`, which accept both separators.

## Scored ideas (value / differentiation / effort / risk, 1-5; effort and risk: low is better)

| # | Idea | Value | Diff. | Effort | Risk | Status |
|---|------|-------|-------|--------|------|--------|
| 1 | LICENSE file + SPDX metadata | 5 | 1 | 1 | 1 | DONE |
| 2 | Author pinned everywhere + test | 4 | 1 | 1 | 1 | DONE |
| 3 | `audit --junit` | 4 | 3 | 2 | 1 | DONE |
| 4 | Console encoding fallback | 4 | 2 | 1 | 1 | DONE |
| 5 | `init` writes UTF-8/LF, plain error on failure | 3 | 1 | 1 | 1 | DONE |
| 6 | BrokenPipe handled quietly | 2 | 1 | 1 | 1 | DONE |
| 7 | Property-based tests | 4 | 2 | 2 | 1 | DONE |
| 8 | Golden-file tests for all CLI formats | 4 | 2 | 2 | 1 | DONE |
| 9 | Mutation sanity script | 4 | 3 | 2 | 1 | DONE |
| 10 | CI `package` job (build, twine, clean install) | 4 | 1 | 1 | 1 | DONE (not run on GitHub) |
| 11 | sdist includes CHANGELOG, tests, golden files | 3 | 1 | 1 | 1 | DONE |
| 12 | subset/superset trajectory match modes | 4 | 4 | 3 | 2 | NOT-DONE |
| 13 | pass@k / pass^k helpers | 3 | 3 | 2 | 1 | NOT-DONE |
| 14 | SARIF output for `audit` (code-scanning tab) | 3 | 4 | 3 | 2 | NOT-DONE (needs the SARIF schema read first; not invented) |
| 15 | GitHub `::error` annotations from `audit` | 3 | 3 | 2 | 1 | NOT-DONE |
| 16 | mypy clean (about 10 annotation findings) | 2 | 1 | 2 | 1 | NOT-DONE, listed in README limits |
| 17 | Pin ruff version in CI | 3 | 1 | 1 | 1 | NOT-DONE (needs a decision on the version) |
| 18 | LangChain/CrewAI adapters | 4 | 3 | 4 | 3 | NOT-DONE (README limit) |
| 19 | Async tools (`await tools.x()`) | 3 | 2 | 4 | 3 | NOT-DONE (README limit) |
| 20 | Case-insensitive FS option in mock FS | 2 | 2 | 3 | 2 | NOT-DONE |
| 21 | Percent-decoded / confusable path matching | 3 | 3 | 4 | 3 | NOT-DONE (README limit) |
| 22 | HTTP 4xx/5xx responses count as errors in `retried_after_error` | 3 | 2 | 2 | 2 | NOT-DONE (behaviour change; needs a decision) |
| 23 | Differential test vs a reference implementation | 2 | 2 | 5 | 1 | NOT-DONE (no reference exists) |
| 24 | Run the CI matrix on Windows | 4 | 1 | 1 | 1 | NOT-DONE (needs Dan to enable; free on public repos, check private-repo minutes) |

## What I could not test
- Windows (any version), Python 3.10, macOS.
- The GitHub Actions workflow itself (not pushed; its steps were run by hand).
- The registry upgrade path (skipped by instruction; the package is not on the registry anyway).
- A single full mutation run of the final mutant set; it was run in two parts (see above).
- Competitor behaviour beyond version numbers.
