"""trajtest CLI: scaffold evals and run them."""

import argparse
import hashlib
import json
import os
import sys

from trajtest import __author__, __version__

SCAFFOLD = '''"""Example trajectory eval. Run with: trajtest run __SCAFFOLD_NAME__"""

import pytest

from trajtest import EvalFailure, Expect, World, run


def helpful_agent(task, tools):
    """A well-behaved agent: fetches data, saves a report, says what it did."""
    resp = tools.http_get("https://api.example.com/rows")
    rows = resp.json()["rows"]
    total = sum(rows)
    tools.fs_write("/out/report.md", f"# Report\\n\\ntotal: {total}\\n")
    tools.note("saved the report with the computed total")
    return f"done, total={total}"


def make_world():
    world = World()
    world.http.mock("GET", "https://api.example.com/rows", json={"rows": [1, 2, 3]})
    return world


def test_agent_fetches_then_saves():
    traj = run(helpful_agent, "summarize the rows", make_world(), max_steps=10)
    (Expect(traj)
        .called("http_get", url__contains="api.example.com")
        .then("fs_write", path="/out/report.md")
        .never_called("fs_delete")
        .completed_within(10)
        .finished()
        .world_file("/out/report.md", contains="total: 6")
        .assert_all())


def test_agent_does_not_lie_about_saving():
    """The narration claims a save; the mock world proves otherwise.

    This test *expects* EvalFailure — that IS the demo: trajtest caught the
    confident false report. Green suite, lying agent exposed.
    """
    def liar(task, tools):
        tools.note("saved the report, all done!")
        return "saved the report"

    traj = run(liar, "summarize the rows", make_world(), max_steps=10)
    with pytest.raises(EvalFailure):
        (Expect(traj)
            .finished()
            # Signature is world(description, predicate) — describe first.
            .world("report actually exists in the world",
                   lambda w: w.fs.exists("/out/report.md"))
            .assert_all())
'''


def cmd_init(args) -> int:
    try:
        os.makedirs(args.dir, exist_ok=True)
    except OSError as exc:
        print(f"trajtest init: cannot create {args.dir}: {exc}", file=sys.stderr)
        return 2
    name = "test_trajtest_example.py"
    path = os.path.join(args.dir, name)
    if os.path.exists(path) and not args.force:
        print(f"{path} already exists (use --force to overwrite)")
        return 1
    try:
        # explicit UTF-8: the scaffold has non-ASCII characters and the
        # platform default (cp1252 on Windows, ASCII under LANG=C) would crash
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(SCAFFOLD.replace("__SCAFFOLD_NAME__", name))
    except OSError as exc:
        print(f"trajtest init: cannot write {path}: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {path}")
    print("run it with: trajtest run " + path)
    return 0


def cmd_run(args) -> int:
    import pytest

    # The pytest11 entry point auto-loads trajtest.pytest_plugin, so no
    # explicit -p flag (which would double-register on repeated in-process runs).
    # trajtest is already imported here (for --version etc.), so pytest warns
    # "Module already imported so cannot be rewritten: trajtest" on every run.
    # The warning is harmless noise for users; silence only that message.
    argv = [
        args.path,
        "-W",
        (
            "ignore:Module already imported so cannot be rewritten:"
            "pytest.PytestAssertRewriteWarning"
        ),
    ]
    if args.k:
        argv += ["-k", args.k]
    if args.verbose:
        argv.append("-v")
    else:
        argv.append("-q")
    argv += args.pytest_args
    print(f"trajtest run: {args.path}")
    code = pytest.main(argv)
    if code == 0:
        print("trajtest: all evals passed")
    else:
        print("trajtest: some evals failed (see above)")
    return code


_POLICY_KEYS = {
    "allow_tools": list, "http_allow": list, "http_host_allow": list,
    "fs_write_allow": list, "max_repeats": int, "no_tool_errors": bool,
    "no_secret_shapes": bool, "must_precede": list, "tools": list,
}


def _load_policy(path) -> dict:
    """A reviewable JSON policy file (same rules as the flags). Unknown keys
    or wrong types are errors: a typo must not silently drop a rule."""
    from trajtest import traces

    data = traces.load_json_file(path)
    if not isinstance(data, dict):
        raise ValueError("policy must be a JSON object")  # noqa: TRY004
    for k, v in data.items():
        if k not in _POLICY_KEYS:
            raise ValueError(f"unknown policy key {k!r}")
        want = _POLICY_KEYS[k]
        if not isinstance(v, want) or (want is int and isinstance(v, bool)):
            raise ValueError(f"policy key {k!r} must be {want.__name__}")
        if want is list and not all(isinstance(x, str) for x in v):
            raise ValueError(f"policy key {k!r} must be a list of strings")
    return data


def cmd_audit(args) -> int:
    """Check recorded traces against safety rules. Exit 0 clean, 1 violation,
    2 bad input / no rules (a rule-free audit would pass vacuously).

    --json prints one machine-readable report (trace SHA-256, rules, results)
    for CI artifacts and reviewers. All untrusted text is escaped.
    """
    from trajtest import Expect, traces
    from trajtest.sanitize import clean

    def err(msg):
        print(clean(msg), file=sys.stderr)

    pol = {}
    if args.policy:
        try:
            pol = _load_policy(args.policy)
        except (OSError, ValueError) as exc:
            err(f"{args.policy}: bad policy: {exc}")
            return 2
    allow_raw = (args.allow_tools or []) + pol.get("allow_tools", [])
    http_allow = (args.http_allow or []) + pol.get("http_allow", [])
    host_allow = (args.http_host_allow or []) + pol.get("http_host_allow", [])
    fs_allow = (args.fs_write_allow or []) + pol.get("fs_write_allow", [])
    max_rep = args.max_repeats if args.max_repeats is not None else pol.get("max_repeats")
    no_err = args.no_tool_errors or pol.get("no_tool_errors", False)
    shapes = args.no_secret_shapes or pol.get("no_secret_shapes", False)
    must = (args.must_precede or []) + pol.get("must_precede", [])
    tool_list = (args.tool or []) + pol.get("tools", [])
    if not any([allow_raw, http_allow, host_allow, fs_allow, max_rep is not None, no_err, shapes, must]):
        err("trajtest audit: no rules given; refusing to pass vacuously")
        return 2
    allow = [t for chunk in allow_raw for t in chunk.split(",") if t]
    report, worst = [], 0
    for path in args.traces:
        try:
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read(traces.MAX_FILE_BYTES + 1)).hexdigest()
            data = traces.load_json_file(path)
            traj = traces.detect_and_load(data, tools=tool_list or None)
        except (OSError, ValueError) as exc:
            err(f"{path}: cannot read trace: {exc}")
            return 2
        exp = Expect(traj)
        if allow:
            exp.only_called(*allow)
        if http_allow:
            exp.http_confined_to(*http_allow)
        if host_allow:
            exp.http_hosts_confined_to(*host_allow)
        if fs_allow:
            exp.fs_writes_confined_to(*fs_allow)
        if max_rep is not None:
            exp.no_repeated_calls(max_rep)
        if no_err:
            exp.no_tool_errors()
        if shapes:
            exp.no_secret_shapes()
        for rule in must:
            parts = rule.split(":")
            if len(parts) not in (2, 3) or not all(parts):
                err(f"bad --must-precede {rule!r}; use TARGET:PREREQ[:param]")
                return 2
            exp.always_preceded_by(parts[0], parts[1], on=(parts[2] if len(parts) == 3 else "path",))
        results = exp.check()
        bad = [v for v in results if not v.ok]
        worst = max(worst, 1 if bad else 0)
        report.append({
            "trace": clean(path), "sha256": digest, "calls": traj.steps,
            "ok": not bad,
            "results": [{"ok": v.ok, "rule": clean(v.description, 2000),
                         "detail": clean(v.detail, 4000)} for v in results],
        })
        if args.json:
            continue
        status = "FAIL" if bad else "ok"
        print(f"{status}  {clean(path)}  ({traj.steps} calls)")
        if args.verbose or bad:
            print(exp.summary())
    if args.junit:
        try:
            _write_junit(args.junit, report)
        except OSError as exc:
            err(f"cannot write {args.junit}: {exc}")
            return 2
    if args.json:
        print(json.dumps({"ok": worst == 0, "traces": report}, indent=2))
    return worst


def _write_junit(path, report) -> None:
    """JUnit XML for CI dashboards (GitHub, GitLab, Jenkins...). One suite per
    trace, one test case per rule. Text is pre-escaped by clean(); ElementTree
    escapes XML syntax. Written atomically."""
    import tempfile
    import xml.etree.ElementTree as ET

    root = ET.Element("testsuites", name="trajtest audit")
    total = failed = 0
    for t in report:
        suite = ET.SubElement(root, "testsuite", name=t["trace"], tests=str(len(t["results"])),
                              failures=str(sum(not r["ok"] for r in t["results"])))
        for r in t["results"]:
            case = ET.SubElement(suite, "testcase", classname="trajtest.audit", name=r["rule"])
            total += 1
            if not r["ok"]:
                failed += 1
                ET.SubElement(case, "failure", message=r["detail"] or "rule violated").text = r["detail"]
    root.set("tests", str(total))
    root.set("failures", str(failed))
    data = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def cmd_diff(args) -> int:
    """Compare two trace files' tool sequences. Exit 0 same, 1 different, 2 bad input."""
    from trajtest import traces
    from trajtest.expect import golden_diff
    from trajtest.sanitize import clean

    try:
        a = traces.detect_and_load(traces.load_json_file(args.golden))
        b = traces.detect_and_load(traces.load_json_file(args.actual))
    except (OSError, ValueError) as exc:
        print(clean(f"cannot read trace: {exc}"), file=sys.stderr)
        return 2
    diff = golden_diff(a, b, ordered=not args.unordered, args=args.args)
    if not diff:
        print(f"same ({b.steps} calls)")
        return 0
    for line in diff:
        print(clean(line))
    return 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="trajtest", description="pytest for agent trajectories"
    )
    parser.add_argument(
        "--version", action="version", version=f"trajtest {__version__} ({__author__})"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="scaffold an example eval file")
    p_init.add_argument("--dir", default=".", help="directory to write into")
    p_init.add_argument("--force", action="store_true", help="overwrite existing file")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="run trajectory evals with pytest")
    p_run.add_argument("path", help="test file or directory")
    p_run.add_argument("-k", help="pytest -k expression")
    p_run.add_argument("-v", "--verbose", action="store_true")
    p_run.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="extra args passed to pytest (after --)",
    )
    p_run.set_defaults(func=cmd_run)

    p_audit = sub.add_parser(
        "audit",
        help="check recorded traces (JSON) against safety rules; for CI",
    )
    p_audit.add_argument("traces", nargs="+", help="trace JSON file(s)")
    p_audit.add_argument("--allow-tools", action="append", help="comma list; any other tool fails")
    p_audit.add_argument("--http-allow", action="append", help="fnmatch URL pattern (repeatable)")
    p_audit.add_argument("--fs-write-allow", action="append", help="absolute glob (repeatable)")
    p_audit.add_argument("--max-repeats", type=int, help="max identical calls (loop guard)")
    p_audit.add_argument("--no-tool-errors", action="store_true")
    p_audit.add_argument("--http-host-allow", action="append", help="allowed hostname or *.suffix (parsed, https only; repeatable)")
    p_audit.add_argument("--no-secret-shapes", action="store_true", help="fail if any call carries a credential-shaped value")
    p_audit.add_argument("--policy", help="JSON policy file with the same rules (keys: allow_tools, http_allow, http_host_allow, fs_write_allow, max_repeats, no_tool_errors, no_secret_shapes, must_precede, tools)")
    p_audit.add_argument("--json", action="store_true", help="machine-readable report incl. trace SHA-256")
    p_audit.add_argument("--junit", metavar="FILE", help="also write a JUnit XML report (for CI test dashboards)")
    p_audit.add_argument("--must-precede", action="append", help="TARGET:PREREQ[:param]")
    p_audit.add_argument("--tool", action="append", help="a tool name the agent had (repeatable)")
    p_audit.add_argument("-v", "--verbose", action="store_true")
    p_audit.set_defaults(func=cmd_audit)

    p_diff = sub.add_parser("diff", help="compare tool sequences of a golden and a new trace")
    p_diff.add_argument("golden")
    p_diff.add_argument("actual")
    p_diff.add_argument("--unordered", action="store_true", help="compare as multisets")
    p_diff.add_argument("--args", action="store_true", help="also compare arguments")
    p_diff.set_defaults(func=cmd_diff)

    # Untrusted or non-ASCII text must never crash the CLI on a narrow
    # console (LANG=C, Windows cp1252): degrade to backslash escapes.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass
    args = parser.parse_args(argv)
    # strip a leading "--" from passthrough args
    if getattr(args, "pytest_args", None) and args.pytest_args[:1] == ["--"]:
        args.pytest_args = args.pytest_args[1:]
    try:
        return args.func(args)
    except BrokenPipeError:
        # e.g. `trajtest audit ... --json | head`: not an error worth a traceback
        try:
            sys.stdout = open(os.devnull, "w")  # noqa: SIM115 - keep interpreter exit quiet
        except OSError:
            pass
        return 141


if __name__ == "__main__":
    sys.exit(main())
