"""Re-runnable micro-benchmarks. Usage: python bench/bench.py

Prints median-of-5 timings for things the experts found slow or
unbounded. Numbers depend on your machine; compare before/after
on the SAME machine.
"""

import statistics
import time

from trajtest import Expect, Trajectory, World, run
from trajtest.sanitize import redact


def timeit(fn, repeat=5):
    vals = []
    for _ in range(repeat):
        s = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - s)
    return statistics.median(vals)


def bench_call_overhead(n=20000):
    w = World()
    w.fs.write("/f", "hi")

    def agent(task, tools):
        for _ in range(n):
            tools.fs_read("/f")

    t = timeit(lambda: run(agent, "t", w, max_steps=n + 1))
    return t / n * 1e6


def bench_glob(stars=6, depth=30):
    w = World()
    w.fs.write("/" + "/".join("d" for _ in range(depth)) + "/x.txt", "1")
    pat = "/" + "/".join(["**"] * stars) + "/y.txt"
    return timeit(lambda: w.fs.glob(pat), repeat=3)


def bench_glob_many_files(files=3000):
    w = World()
    for i in range(files):
        w.fs.write(f"/src/pkg{i % 50}/mod{i}.py", "x")
    return timeit(lambda: w.fs.glob("/src/**/mod*.py"), repeat=3)


def bench_big_value_report(mb=8):
    """Print the transcript of a run that read and wrote an `mb` MB value."""
    big = "x" * (mb * 1024 * 1024)

    def agent(task, tools):
        tools.fs_write("/a", big)
        tools.fs_read("/a")
        return big

    def go():
        t = run(agent, "t", World())
        t.transcript()

    return timeit(go, repeat=3)


def bench_preceded_by(n=20000):
    calls = [{"name": "fs_read", "args": [f"/f{i}"]} for i in range(n)]
    calls += [{"name": "fs_delete", "args": [f"/f{i}"]} for i in range(n)]
    t = Trajectory.from_dict({"schema": 1, "calls": calls})
    return timeit(lambda: Expect(t).always_preceded_by("fs_delete", "fs_read").check(), repeat=3)


def bench_redact_hostile(n=30000):
    text = "token='x\" " * n
    return timeit(lambda: redact(text), repeat=3)


if __name__ == "__main__":
    print(f"tool-call overhead      : {bench_call_overhead():8.2f} us/call")
    print(f"glob 6x'**' 30 deep     : {bench_glob() * 1000:8.1f} ms")
    print(f"glob over 3000 files    : {bench_glob_many_files() * 1000:8.1f} ms")
    print(f"report for 8 MB value   : {bench_big_value_report() * 1000:8.1f} ms")
    print(f"preceded_by 40k calls   : {bench_preceded_by() * 1000:8.1f} ms")
    print(f"redact 300 KB hostile   : {bench_redact_hostile() * 1000:8.1f} ms")
