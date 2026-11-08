# Publish checklist for trajtest (plain words)

Nothing here has been run for you. Uploading is your step; you are signed in to PyPI yourself.

**Version to release: 0.2.0.** It has never been published (a search of the package index finds nothing), so no version bump is needed. Nothing in this release breaks existing users.

## Before you upload
1. Make sure you are in the project folder and the tests pass:
   `pip install -e ".[dev]"` then `pytest -q` (expect 284 passed, about 15 seconds) and `ruff check src tests bench` (expect "All checks passed").
2. Build the files to upload into a fresh folder:
   `python -m build --outdir dist`
3. Check them:
   `twine check dist/*` (both must say PASSED).
4. Look at the author line:
   `unzip -p dist/*.whl '*METADATA' | head -8` should show `Author: Calas Automations Inc.` and `License-Expression: MIT`.
5. Try it in a clean place: `python -m venv /tmp/try && /tmp/try/bin/pip install dist/*.whl && /tmp/try/bin/trajtest --version` should print `trajtest 0.2.0 (Calas Automations Inc.)`.
6. Check the name `trajtest` is still free on pypi.org (search the site). If someone else took it, stop and rename before uploading.

## Upload (your step)
`twine upload dist/*`
A PyPI upload cannot be undone or re-used for the same version number. After it works, open the project page on pypi.org and check the README shows properly, then change the README "Install" note (it currently says it is not released yet).

## After
Tag the release in git yourself if you want one (`git tag v0.2.0`). Decide whether to turn on a Windows test run in CI (the code has never been run on Windows).
