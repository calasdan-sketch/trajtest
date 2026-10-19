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
    argv = [args.path]
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

    args = parser.parse_args(argv)
    # strip a leading "--" from passthrough args
    if getattr(args, "pytest_args", None) and args.pytest_args[:1] == ["--"]:
        args.pytest_args = args.pytest_args[1:]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
