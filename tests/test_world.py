"""Tests for the mock world: FakeFS, FakeHTTP, KVStore, Clock, World."""

import pytest

from trajtest.errors import UnmockedRequestError
from trajtest.world import Clock, FakeFS, FakeHTTP, KVStore, World


class TestFakeFS:
    def test_write_and_read(self):
        fs = FakeFS()
        fs.write("/docs/a.md", "hello")
        assert fs.read("/docs/a.md") == "hello"

    def test_read_missing_raises(self):
        fs = FakeFS()
        with pytest.raises(FileNotFoundError):
            fs.read("/nope.txt")

    def test_paths_must_be_absolute(self):
        fs = FakeFS()
        with pytest.raises(ValueError):
            fs.write("relative/path.txt", "x")

    def test_dotdot_cannot_escape_root(self):
        fs = FakeFS()
        fs.write("/a.txt", "x")
        assert fs.read("/../../a.txt") == "x"
        assert fs.read("/sub/../a.txt") == "x"

    def test_append(self):
        fs = FakeFS()
        fs.write("/a.txt", "x")
        fs.append("/a.txt", "y")
        assert fs.read("/a.txt") == "xy"

    def test_append_missing_raises(self):
        fs = FakeFS()
        with pytest.raises(FileNotFoundError):
            fs.append("/nope.txt", "y")

    def test_delete(self):
        fs = FakeFS()
        fs.write("/a.txt", "x")
        fs.delete("/a.txt")
        assert not fs.exists("/a.txt")

    def test_delete_missing_raises(self):
        fs = FakeFS()
        with pytest.raises(FileNotFoundError):
            fs.delete("/nope.txt")

    def test_exists_dirs_implicit(self):
        fs = FakeFS()
        fs.write("/docs/sub/a.md", "x")
        assert fs.exists("/docs")
        assert fs.exists("/docs/sub")
        assert fs.exists("/docs/sub/a.md")
        assert not fs.exists("/other")

    def test_listdir(self):
        fs = FakeFS()
        fs.write("/docs/a.md", "x")
        fs.write("/docs/sub/b.md", "y")
        assert fs.listdir("/docs") == ["a.md", "sub"]

    def test_listdir_missing_raises(self):
        fs = FakeFS()
        with pytest.raises(FileNotFoundError):
            fs.listdir("/nope")

    def test_listdir_on_file_raises(self):
        fs = FakeFS()
        fs.write("/a.txt", "x")
        with pytest.raises(NotADirectoryError):
            fs.listdir("/a.txt")

    def test_glob(self):
        fs = FakeFS()
        fs.write("/docs/a.md", "x")
        fs.write("/docs/b.txt", "y")
        fs.write("/docs/sub/c.md", "z")
        assert fs.glob("/docs/*.md") == ["/docs/a.md"]
        assert fs.glob("/docs/**/*.md") == ["/docs/a.md", "/docs/sub/c.md"]

    def test_mkdir(self):
        fs = FakeFS()
        fs.mkdir("/empty/dir")
        assert fs.exists("/empty/dir")
        assert fs.listdir("/empty/dir") == []

    def test_write_non_str_rejected(self):
        fs = FakeFS()
        with pytest.raises(TypeError):
            fs.write("/a.bin", 12345)

    def test_bytes_roundtrip(self):
        fs = FakeFS()
        fs.write("/a.bin", b"\x00\x01\x02")
        assert fs.read("/a.bin") == b"\x00\x01\x02"

    def test_append_bytes(self):
        fs = FakeFS()
        fs.write("/a.bin", b"\x00")
        fs.append("/a.bin", b"\x01")
        assert fs.read("/a.bin") == b"\x00\x01"

    def test_append_mixed_types_rejected(self):
        fs = FakeFS()
        fs.write("/a.txt", "x")
        with pytest.raises(TypeError):
            fs.append("/a.txt", b"y")

    def test_reset(self):
        fs = FakeFS()
        fs.write("/a.txt", "x")
        fs.reset()
        assert not fs.exists("/a.txt")


class TestFakeHTTP:
    def test_mock_and_get_json(self):
        http = FakeHTTP()
        http.mock("GET", "https://api.example.com/rows", json={"rows": [1]})
        resp = http.get("https://api.example.com/rows")
        assert resp.status_code == 200
        assert resp.json() == {"rows": [1]}
        assert resp.ok

    def test_method_case_insensitive(self):
        http = FakeHTTP()
        http.mock("get", "https://x.example/", text="hi")
        assert http.request("GET", "https://x.example/").text == "hi"

    def test_wildcard_url(self):
        http = FakeHTTP()
        http.mock("GET", "https://api.example.com/users/*", json={"ok": True})
        assert http.get("https://api.example.com/users/42").json() == {"ok": True}

    def test_exact_beats_wildcard(self):
        http = FakeHTTP()
        http.mock("GET", "https://api.example.com/users/*", text="wild")
        http.mock("GET", "https://api.example.com/users/1", text="exact")
        assert http.get("https://api.example.com/users/1").text == "exact"
        assert http.get("https://api.example.com/users/2").text == "wild"

    def test_unmocked_raises_strict(self):
        http = FakeHTTP()
        with pytest.raises(UnmockedRequestError):
            http.get("https://nope.example/")

    def test_unmocked_non_strict_returns_404(self):
        http = FakeHTTP(strict=False)
        resp = http.get("https://nope.example/")
        assert resp.status_code == 404

    def test_error_timeout(self):
        http = FakeHTTP()
        http.mock("GET", "https://slow.example/", error="timeout")
        with pytest.raises(TimeoutError):
            http.get("https://slow.example/")

    def test_error_connection(self):
        http = FakeHTTP()
        http.mock("POST", "https://down.example/", error="connection")
        with pytest.raises(ConnectionError):
            http.post("https://down.example/")

    def test_bad_error_mode_rejected(self):
        http = FakeHTTP()
        with pytest.raises(ValueError):
            http.mock("GET", "https://x.example/", error="explode")

    def test_status_and_raise_for_status(self):
        http = FakeHTTP()
        http.mock("GET", "https://x.example/", status=500, text="boom")
        resp = http.get("https://x.example/")
        assert not resp.ok
        with pytest.raises(IOError):
            resp.raise_for_status()

    def test_json_null_body(self):
        http = FakeHTTP()
        http.mock("GET", "https://x.example/", json=None)
        assert http.get("https://x.example/").json() is None

    def test_json_missing_raises(self):
        http = FakeHTTP()
        http.mock("GET", "https://x.example/", text="plain")
        with pytest.raises(ValueError):
            http.get("https://x.example/").json()

    def test_mock_queue_consumed_in_order(self):
        http = FakeHTTP()
        http.mock("GET", "https://x.example/", status=500, text="boom")
        http.mock("GET", "https://x.example/", status=200, text="ok")
        assert http.get("https://x.example/").status_code == 500
        assert http.get("https://x.example/").text == "ok"
        # queue exhausted: last mock repeats
        assert http.get("https://x.example/").text == "ok"

    def test_mock_times(self):
        http = FakeHTTP()
        http.mock("GET", "https://x.example/", status=500, text="boom", times=2)
        http.mock("GET", "https://x.example/", status=200, text="ok")
        assert http.get("https://x.example/").status_code == 500
        assert http.get("https://x.example/").status_code == 500
        assert http.get("https://x.example/").text == "ok"

    def test_mock_times_invalid(self):
        http = FakeHTTP()
        with pytest.raises(ValueError):
            http.mock("GET", "https://x.example/", times=0)

    def test_retry_scenario(self):
        """The motivating case: agent retries a flaky endpoint."""
        http = FakeHTTP()
        http.mock("GET", "https://flaky.example/", error="timeout")
        http.mock("GET", "https://flaky.example/", json={"ok": True})
        with pytest.raises(TimeoutError):
            http.get("https://flaky.example/")
        assert http.get("https://flaky.example/").json() == {"ok": True}

    def test_requests_logged(self):
        http = FakeHTTP()
        http.mock("GET", "https://x.example/a", text="1")
        http.get("https://x.example/a", headers={"X": "y"})
        assert len(http.requests) == 1
        assert http.requests[0].method == "GET"
        assert http.requests[0].url == "https://x.example/a"

    def test_convenience_verbs(self):
        http = FakeHTTP()
        for verb in ["get", "post", "put", "patch", "delete"]:
            http.mock(verb, f"https://x.example/{verb}", text=verb)
        assert http.get("https://x.example/get").text == "get"
        assert http.post("https://x.example/post").text == "post"
        assert http.put("https://x.example/put").text == "put"
        assert http.patch("https://x.example/patch").text == "patch"
        assert http.delete("https://x.example/delete").text == "delete"


class TestKVStore:
    def test_set_get(self):
        kv = KVStore()
        kv.set("k", {"v": 1})
        assert kv.get("k") == {"v": 1}

    def test_get_missing_raises(self):
        with pytest.raises(KeyError):
            KVStore().get("nope")

    def test_get_default(self):
        assert KVStore().get("nope", "dflt") == "dflt"

    def test_delete(self):
        kv = KVStore()
        kv.set("k", 1)
        kv.delete("k")
        assert not kv.exists("k")

    def test_delete_missing_raises(self):
        with pytest.raises(KeyError):
            KVStore().delete("nope")


class TestClock:
    def test_advance(self):
        c = Clock()
        before = c.now()
        c.advance(3600)
        assert (c.now() - before).total_seconds() == 3600


class TestWorld:
    def test_bundles_components(self):
        w = World()
        assert isinstance(w.fs, FakeFS)
        assert isinstance(w.http, FakeHTTP)
        assert isinstance(w.kv, KVStore)
        assert isinstance(w.clock, Clock)

    def test_reset(self):
        w = World()
        w.fs.write("/a.txt", "x")
        w.http.mock("GET", "https://x.example/", text="y")
        w.kv.set("k", 1)
        w.reset()
        assert not w.fs.exists("/a.txt")
        assert w.http.requests == []
        assert not w.kv.exists("k")

    def test_strict_http_default(self):
        assert World().http.strict is True

    def test_non_strict_option(self):
        assert World(strict_http=False).http.strict is False
