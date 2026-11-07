"""Round 3 regressions: expert findings from hands-on attacks."""

import base64
import json
import time

import pytest

from trajtest import (
    AgentTimeout,
    EvalFailure,
    Expect,
    Trajectory,
    World,
    flaky,
    run,
    traces,
)
from trajtest.cli import main
from trajtest.sanitize import redact


def bad(exp):
    return [v for v in exp.check() if not v.ok]


def trace(*calls):
    return Trajectory.from_dict({"schema": 1, "calls": list(calls)})


# -- path normalisation (never_called bypass) --------------------------------
@pytest.mark.parametrize("p", ["/data/./keep.txt", "//data/keep.txt", "/tmp/../data/keep.txt", "/data/keep.txt/"])
def test_never_called_sees_normalised_paths(p):
    w = World()
    w.fs.write("/data/keep.txt", "a")
    t = run(lambda task, tools: tools.fs_delete(p), "t", w)
    assert not w.fs.exists("/data/keep.txt")  # it really was deleted
    assert bad(Expect(t).never_called("fs_delete", path="/data/keep.txt"))
    assert bad(Expect(t).never_called("fs_delete", "/data/keep.txt"))


def test_called_matches_normalised_path_too():
    w = World()
    t = run(lambda task, tools: tools.fs_write("/a/./b.txt", "x"), "t", w)
    assert not bad(Expect(t).called("fs_write", path="/a/b.txt"))


# -- confinement fail-closed --------------------------------------------------
@pytest.mark.parametrize("p", ["/out/..\\..\\etc\\x", "/out/a\x00b", "/out/%2e%2e/x", "C:\\out\\x"])
def test_confinement_rejects_ambiguous_paths(p):
    assert bad(Expect(trace({"name": "fs_write", "args": [p, "x"]})).fs_writes_confined_to("/out/**"))


def test_confinement_checks_all_copies_of_an_argument():
    t = trace({"name": "fs_write", "args": ["/etc/passwd", "a"], "bound": {"path": "/out/x"}})
    assert bad(Expect(t).fs_writes_confined_to("/out/**"))
    t = trace({"name": "http_get", "args": ["https://evil.net/"], "bound": {"url": "https://api.example.com/x"}})
    assert bad(Expect(t).http_confined_to("https://api.example.com/*"))


def test_windows_path_error_has_hint():
    with pytest.raises(ValueError, match="Windows"):
        World().fs.write("C:\\out\\x", "a")


# -- host allowlist -----------------------------------------------------------
@pytest.mark.parametrize("url,ok", [
    ("https://api.example.com/x", True),
    ("HTTPS://API.Example.COM:443/x", True),
    ("https://v1.api.example.com/x", True),
    ("https://api.example.com./x", True),
    ("https://api.example.com.evil.net/", False),
    ("https://evil.net/?h=api.example.com", False),
    ("https://api.example.com@evil.net/", False),
    ("https://evil.net\\@api.example.com/", False),
    ("http://api.example.com/x", False),
    ("https://example.com/", False),
    ("https://xn--pple-43d.example/", False),
    ("https://api.example.com:notaport/", False),
    ("https://\u0430pi.example.com/", False),
    ("https://2130706433/", False),
    ("api.example.com/x", False),
    ("https://api.example.com/x\n", False),
])
def test_host_allowlist(url, ok):
    t = trace({"name": "http_get", "args": [url]})
    assert (not bad(Expect(t).http_hosts_confined_to("api.example.com", "*.api.example.com"))) is ok


# -- data leak / secret shapes -----------------------------------------------
def test_no_data_leak_detects_raw_encoded_and_nested():
    secret = "s3cr" + "et-value-" + "42"
    for payload in [secret, base64.b64encode(secret.encode()).decode(), secret.replace("-", "%2D")]:
        t = trace({"name": "http_post", "args": ["https://x/"], "kwargs": {"json": {"a": [{"b": payload}]}}})
        assert bad(Expect(t).no_data_leak(secret)), payload
    clean = trace({"name": "http_post", "args": ["https://x/"], "kwargs": {"json": {"a": 1}}})
    assert not bad(Expect(clean).no_data_leak(secret))
    with pytest.raises(ValueError):
        Expect(clean).no_data_leak("ab")


def test_no_data_leak_report_does_not_print_value():
    secret = "pa55" + "word-xyz"
    t = trace({"name": "http_get", "args": ["https://x/?q=" + secret]})
    with pytest.raises(EvalFailure) as e:
        Expect(t).no_data_leak(secret).assert_all()
    assert secret not in str(e.value)


def test_no_secret_shapes():
    key = "sk-" + "a1B2c3D4e5F6g7H8"
    assert bad(Expect(trace({"name": "http_get", "args": ["https://x/?api_key=abc123"]})).no_secret_shapes())
    assert bad(Expect(trace({"name": "kv_set", "args": ["k", key]})).no_secret_shapes())
    assert not bad(Expect(trace({"name": "kv_set", "args": ["k", "hello"]})).no_secret_shapes())


# -- redaction -----------------------------------------------------------------
def test_redaction_new_shapes():
    out = redact("GET https://a.io/x?api_key=ABC123&b=1 https://u:pw@h.io " + "sk_" + "live_" + "a" * 20)
    assert "ABC123" not in out and "pw@" not in out and "a" * 20 not in out and "b=1" in out
    pem = "-----BEGIN RSA " + "PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----"
    assert "MIIabc" not in redact("x " + pem + " y")


def test_huge_values_print_fast():
    big = "x" * (8 * 1024 * 1024)

    def agent(task, tools):
        tools.fs_write("/a", big)
        tools.fs_read("/a")
        return big

    t0 = time.perf_counter()
    t = run(agent, "t", World())
    with pytest.raises(EvalFailure) as e:
        Expect(t).world_file("/a", equals="nope").assert_all()
    t.transcript()
    assert time.perf_counter() - t0 < 3
    assert len(str(e.value)) < 20000


# -- trace loader fuzz ---------------------------------------------------------
@pytest.mark.parametrize("data", [
    {"schema": 1, "calls": [], "max_steps": [1]},
    {"schema": 1, "calls": [], "max_steps": float("nan")},
    {"schema": 1, "calls": [], "notes": "x"},
    {"schema": 1, "calls": [{"name": "a", "duration_ms": [1]}]},
    {"schema": 1, "calls": [], "notes": [{"text": "a", "step": [1]}]},
    [{"role": "assistant", "tool_calls": ["x"]}],
    [{"role": "assistant", "tool_calls": 5}],
    [{"role": "assistant", "tool_calls": [{"function": 3}]}],
    [{"role": "assistant", "content": [{"type": "tool_use", "name": "a", "input": [], "id": 1}]}],
])
def test_loader_only_raises_valueerror(data):
    with pytest.raises(ValueError):
        traces.detect_and_load(data)


def test_deeply_nested_file_is_valueerror(tmp_path, capsys):
    p = tmp_path / "deep.json"
    p.write_text("[" * 200000)
    assert main(["audit", str(p), "--no-tool-errors"]) == 2


def test_loader_random_fuzz_never_crashes():
    import random

    rnd = random.Random(7)
    atoms = [None, True, 0, -1, 1.5, "a", "", [], {}, [None], {"name": 1}, {"function": None}]

    def gen(d=0):
        if d > 3 or rnd.random() < 0.3:
            return rnd.choice(atoms)
        if rnd.random() < 0.5:
            return [gen(d + 1) for _ in range(rnd.randint(0, 3))]
        keys = ["role", "tool_calls", "content", "calls", "name", "args", "kwargs", "bound", "schema",
                "notes", "text", "step", "id", "function", "arguments", "type", "input", "messages"]
        return {rnd.choice(keys): gen(d + 1) for _ in range(rnd.randint(0, 4))}

    for _ in range(3000):
        try:
            traces.detect_and_load(gen())
        except ValueError:
            pass


# -- save redaction + atomic ---------------------------------------------------
def test_save_redacts_by_default(tmp_path):
    key = "ghp_" + "A" * 24
    t = run(lambda task, tools: tools.kv_set("api_key", key) or tools.http_get("https://x.io/?token=abc"), "t",
            World(strict_http=False))
    p = tmp_path / "g.json"
    t.save(p)
    raw = p.read_text()
    assert key not in raw and "token=abc" not in raw
    assert not [f for f in tmp_path.iterdir() if f.name.startswith(".trajtest-")]
    t.save(p, redact=False)
    assert key in p.read_text()


# -- timeout --------------------------------------------------------------------
def test_run_timeout_freezes_trajectory():
    def hang(task, tools):
        tools.kv_set("a", 1)
        time.sleep(0.6)
        tools.kv_set("b", 2)  # after cancellation: raises, not recorded
        return "x"

    t = run(hang, "t", World(), timeout=0.2)
    assert not t.finished and "AgentTimeout" in t.failure
    time.sleep(0.6)
    assert t.steps == 1
    assert issubclass(AgentTimeout, BaseException)


def test_timeout_passthrough_and_errors():
    t = run(lambda task, tools: tools.kv_set("a", 1) or "ok", "t", World(), timeout=5)
    assert t.finished and t.final_answer == "ok"
    t = run(lambda task, tools: 1 / 0, "t", World(), timeout=5)
    assert "ZeroDivisionError" in t.failure
    with pytest.raises(ValueError):
        run(lambda a, b: 1, "t", World(), timeout=0)


# -- retries, golden diff, audit ------------------------------------------------
def test_retried_after_error():
    def agent(task, tools):
        for _ in range(3):
            try:
                return tools.send("x")
            except TimeoutError:
                pass

    t = run(agent, "t", World(), tools={"send": flaky(lambda m: "sent", fail_times=1, error=TimeoutError("t"))})
    assert not bad(Expect(t).retried_after_error("send"))
    t2 = run(lambda task, tools: tools.send("x"), "t", World(), tools={"send": lambda m: "sent"})
    assert bad(Expect(t2).retried_after_error("send"))


def test_same_tools_as_golden():
    g = trace({"name": "a"}, {"name": "b"}, {"name": "c"})
    same = trace({"name": "a"}, {"name": "b"}, {"name": "c"})
    other = trace({"name": "a"}, {"name": "c"}, {"name": "b"}, {"name": "d"})
    assert not bad(Expect(same).same_tools_as(g))
    assert bad(Expect(other).same_tools_as(g))
    assert bad(Expect(other).same_tools_as(g, ordered=False))
    assert not bad(Expect(trace({"name": "c"}, {"name": "b"}, {"name": "a"})).same_tools_as(g, ordered=False))


def test_cli_policy_json_and_diff(tmp_path, capsys):
    tr = tmp_path / "t.json"
    tr.write_text(json.dumps({"schema": 1, "calls": [{"name": "http_get", "args": ["https://evil.net/x"]}]}))
    pol = tmp_path / "p.json"
    pol.write_text(json.dumps({"http_host_allow": ["api.example.com"]}))
    assert main(["audit", str(tr), "--policy", str(pol), "--json"]) == 1
    rep = json.loads(capsys.readouterr().out)
    assert rep["ok"] is False and len(rep["traces"][0]["sha256"]) == 64
    pol.write_text(json.dumps({"http_hosts_allow": ["x"]}))  # typo: must not silently pass
    assert main(["audit", str(tr), "--policy", str(pol)]) == 2
    g = tmp_path / "g.json"
    g.write_text(json.dumps({"schema": 1, "calls": [{"name": "http_get"}]}))
    assert main(["diff", str(g), str(tr)]) == 0
    g.write_text(json.dumps({"schema": 1, "calls": [{"name": "fs_read"}]}))
    assert main(["diff", str(g), str(tr)]) == 1


def test_always_preceded_by_linear_and_correct():
    calls = [{"name": "fs_read", "args": [f"/f{i}"]} for i in range(20000)]
    calls += [{"name": "fs_delete", "args": [f"/f{i}"]} for i in range(20000)] + [{"name": "fs_delete", "args": ["/zzz"]}]
    t0 = time.perf_counter()
    bads = bad(Expect(trace(*calls)).always_preceded_by("fs_delete", "fs_read"))
    assert time.perf_counter() - t0 < 2 and len(bads) == 1


def test_unknown_tool_attribute_has_hint():
    def agent(task, tools):
        tools.fs_wrte("/a", "x")

    t = run(agent, "t", World())
    assert "fs_write" in t.failure


def test_fs_thread_safety_and_utf8_quota():
    import threading

    w = World(fs_max_bytes=1000)
    with pytest.raises(OSError):
        w.fs.write("/a", "é" * 600)  # 1200 bytes
    w = World()

    def work(n):
        for i in range(200):
            w.fs.write(f"/t{n}/{i}", "x" * 10)
            w.fs.delete(f"/t{n}/{i}")

    ths = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    assert w.fs._used == 0


# -- pass B: attacks on the round-3 code ---------------------------------------
def test_long_secret_value_cut_by_truncation_is_still_masked():
    long_secret = "Z" * 900
    t = run(lambda task, tools: tools.kv_set("k", {"password": long_secret}) or tools.kv_get("k"), "t", World())
    shown = t.calls[-1].result_repr + t.transcript()
    assert "ZZZZZZZZ" not in shown
    assert "ZZZZZZZZ" not in redact("{'token': '" + "Z" * 50)  # unterminated quote


def test_no_secret_shapes_sees_secret_keys_in_kwargs():
    t = trace({"name": "http_post", "args": ["https://x/"], "kwargs": {"headers": {"Authorization": "Bearer abcdef123456"}}})
    assert bad(Expect(t).no_secret_shapes())
    t = trace({"name": "custom", "kwargs": {"api_key": "hunter2hunter2"}})
    assert bad(Expect(t).no_secret_shapes())


def test_golden_roundtrip_with_redaction_still_equal(tmp_path):
    def agent(task, tools):
        tools.http_get("https://x.io/?token=abcdef")
        return 1

    t = run(agent, "t", World(strict_http=False))
    p = tmp_path / "g.json"
    t.save(p)
    golden = Trajectory.load(p)
    assert not bad(Expect(run(agent, "t", World(strict_http=False))).same_tools_as(golden, args=True))


def test_assert_all_masks_known_secret_in_transcript():
    secret = "plainsecret" + "-12345"
    t = run(lambda task, tools: tools.kv_set("note", secret), "t", World())
    with pytest.raises(EvalFailure) as e:
        Expect(t).no_data_leak(secret, tools=("kv_*",)).assert_all()
    assert secret not in str(e.value)


def test_sample_accepts_timeout():
    from trajtest import sample

    ts = sample(lambda task, tools: tools.kv_set("a", 1) or "ok", "t", World, n=2, timeout=5)
    assert all(t.finished for t in ts)


def test_async_agent_with_timeout():
    async def agent(task, tools):
        tools.kv_set("a", 1)
        return "ok"

    t = run(agent, "t", World(), timeout=5)
    assert t.finished and t.final_answer == "ok"


def test_hosts_wildcard_does_not_match_bare_domain():
    t = trace({"name": "http_get", "args": ["https://example.com/"]})
    assert bad(Expect(t).http_hosts_confined_to("*.example.com"))
