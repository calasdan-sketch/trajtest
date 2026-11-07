"""trajtest CLI: scaffold evals and run them."""

import argparse
import sys

from trajtest import __version__

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
    import os

    os.makedirs(args.dir, exist_ok=True)
    name = "test_trajtest_example.py"
    path = os.path.join(args.dir, name)
    if os.path.exists(path) and not args.force:
        print(f"{path} already exists (use --force to overwrite)")
        return 1
    with open(path, "w") as f:
        f.write(SCAFFOLD.replace("__SCAFFOLD_NAME__", name))
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


def cmd_audit(args) -> int:
    """Check recorded traces against safety rules. Exit 0 clean, 1 violation,
    2 bad input / no rules (a rule-free audit would pass vacuously)."""
    from trajtest import Expect, traces

    rules = [
        args.allow_tools, args.http_allow, args.fs_write_allow,
        args.max_repeats is not None, args.no_tool_errors, args.must_precede,
    ]
    if not any(rules):
        print("trajtest audit: no rules given; refusing to pass vacuously", file=sys.stderr)
        return 2
    allow = [t for chunk in (args.allow_tools or []) for t in chunk.split(",") if t]
    worst = 0
    for path in args.traces:
        try:
            data = traces.load_json_file(path)
            traj = traces.detect_and_load(data, tools=args.tool or None)
        except (OSError, ValueError) as exc:
            print(f"{path}: cannot read trace: {exc}", file=sys.stderr)
            return 2
        exp = Expect(traj)
        if allow:
            exp.only_called(*allow)
        if args.http_allow:
            exp.http_confined_to(*args.http_allow)
        if args.fs_write_allow:
            exp.fs_writes_confined_to(*args.fs_write_allow)
        if args.max_repeats is not None:
            exp.no_repeated_calls(args.max_repeats)
        if args.no_tool_errors:
            exp.no_tool_errors()
        for rule in args.must_precede or []:
            parts = rule.split(":")
            if len(parts) not in (2, 3) or not all(parts):
                print(f"bad --must-precede {rule!r}; use TARGET:PREREQ[:param]", file=sys.stderr)
                return 2
            exp.always_preceded_by(parts[0], parts[1], on=(parts[2] if len(parts) == 3 else "path",))
        bad = [v for v in exp.check() if not v.ok]
        status = "FAIL" if bad else "ok"
        print(f"{status}  {path}  ({traj.steps} calls)")
        if args.verbose or bad:
            print(exp.summary())
        worst = max(worst, 1 if bad else 0)
    return worst


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="trajtest", description="pytest for agent trajectories"
    )
    parser.add_argument(
        "--version", action="version", version=f"trajtest {__version__}"
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
    p_audit.add_argument("--must-precede", action="append", help="TARGET:PREREQ[:param]")
    p_audit.add_argument("--tool", action="append", help="a tool name the agent had (repeatable)")
    p_audit.add_argument("-v", "--verbose", action="store_true")
    p_audit.set_defaults(func=cmd_audit)

    args = parser.parse_args(argv)
    # strip a leading "--" from passthrough args
    if getattr(args, "pytest_args", None) and args.pytest_args[:1] == ["--"]:
        args.pytest_args = args.pytest_args[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
