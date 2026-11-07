"""Re-runnable micro-benchmarks. Usage: python bench/bench.py

Prints median-of-5 timings for three things the experts found slow or
unbounded in v0.1. Numbers depend on your machine; compare before/after
on the SAME machine.
"""

import statistics
import time

from trajtest import World, run


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


if __name__ == "__main__":
    print(f"tool-call overhead      : {bench_call_overhead():8.2f} us/call")
    print(f"glob 6x'**' 30 deep     : {bench_glob() * 1000:8.1f} ms")
    print(f"glob over 3000 files    : {bench_glob_many_files() * 1000:8.1f} ms")
