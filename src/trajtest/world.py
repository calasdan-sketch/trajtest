"""The mock world: in-memory FS, mock HTTP, KV store, clock.

Strict by design: missing files raise FileNotFoundError, unmocked HTTP
requests raise UnmockedRequestError. An eval world that forgives everything
produces agents that pass evals and fail in production.
"""

import fnmatch
import functools
import posixpath
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from trajtest.errors import FSQuotaExceeded, UnmockedRequestError


def _locked(fn):
    """Serialise a method on the instance's RLock (agents may use threads)."""

    @functools.wraps(fn)
    def wrapper(self, *a, **k):
        with self._lock:
            return fn(self, *a, **k)

    return wrapper


def _norm(path: str) -> str:
    """Normalize to an absolute posix path. `..` cannot escape root."""
    if not isinstance(path, str) or not path.startswith("/"):
        hint = ""
        if isinstance(path, str) and (path[1:3] in (":\\", ":/") or path.startswith("\\")):
            hint = " (Windows-style path: the mock FS is posix; use e.g. '/c/out/x')"
        raise ValueError(f"trajtest FakeFS paths must be absolute, got {path!r}{hint}")
    if "\x00" in path:
        raise ValueError("trajtest FakeFS paths must not contain NUL")
    parts = []
    for part in posixpath.normpath(path).split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/" + "/".join(parts)


def _size(content) -> int:
    """Quota size: bytes for bytes, UTF-8 bytes for str (len() undercounts
    non-ASCII up to 4x, so a 64 MiB quota could really hold 256 MiB)."""
    if isinstance(content, str):
        return len(content) if content.isascii() else len(content.encode("utf-8", "surrogatepass"))
    return len(content)


class FakeFS:
    """In-memory filesystem. Directories are implicit (derived from files)
    unless created with mkdir(). Content may be str or bytes; read()
    returns whatever was written."""

    #: default total content budget (characters for str, bytes for bytes)
    DEFAULT_MAX_BYTES = 64 * 1024 * 1024

    def __init__(self, max_bytes: int | None = DEFAULT_MAX_BYTES):
        self._files: dict[str, str | bytes] = {}
        self._dirs: set[str] = {"/"}
        self.max_bytes = max_bytes  # None disables the quota
        self._used = 0
        self._lock = threading.RLock()

    def _charge(self, delta: int) -> None:
        if self.max_bytes is not None and self._used + delta > self.max_bytes:
            raise FSQuotaExceeded(
                f"FakeFS quota exceeded: {self._used + delta} > {self.max_bytes} "
                "bytes. A runaway agent can exhaust CI memory; raise "
                "World(fs_max_bytes=...) if this is intended."
            )
        self._used += delta

    # -- mutation ------------------------------------------------------
    @_locked
    def write(self, path: str, content) -> None:
        path = _norm(path)
        if not isinstance(content, (str, bytes)):
            raise TypeError(
                f"FakeFS.write content must be str or bytes, "
                f"got {type(content).__name__}"
            )
        self._charge(_size(content) - _size(self._files.get(path, "")))
        self._files[path] = content
        self._ensure_parents(path)

    @_locked
    def append(self, path: str, content) -> None:
        path = _norm(path)
        if path not in self._files:
            raise FileNotFoundError(f"No such file: {path}")
        existing = self._files[path]
        if type(existing) is not type(content):
            raise TypeError(
                f"cannot append {type(content).__name__} to "
                f"{type(existing).__name__} file"
            )
        self._charge(_size(content))
        self._files[path] = existing + content

    @_locked
    def delete(self, path: str) -> None:
        path = _norm(path)
        if path not in self._files:
            raise FileNotFoundError(f"No such file: {path}")
        self._used -= _size(self._files[path])
        del self._files[path]

    @_locked
    def mkdir(self, path: str) -> None:
        path = _norm(path)
        self._dirs.add(path)
        self._ensure_parents(path)

    def _ensure_parents(self, path: str) -> None:
        parent = posixpath.dirname(path)
        while parent and parent not in self._dirs:
            self._dirs.add(parent)
            parent = posixpath.dirname(parent)

    # -- inspection ----------------------------------------------------
    @_locked
    def read(self, path: str):
        """Return file content as written (str or bytes)."""
        path = _norm(path)
        if path not in self._files:
            raise FileNotFoundError(f"No such file: {path}")
        return self._files[path]

    @_locked
    def exists(self, path: str) -> bool:
        path = _norm(path)
        return path in self._files or path in self._dirs

    @_locked
    def is_file(self, path: str) -> bool:
        return _norm(path) in self._files

    @_locked
    def is_dir(self, path: str) -> bool:
        path = _norm(path)
        return path in self._dirs and path not in self._files

    @_locked
    def listdir(self, path: str) -> list:
        """Immediate child names (files and dirs) of a directory."""
        path = _norm(path)
        if path in self._files:
            raise NotADirectoryError(f"Not a directory: {path}")
        if path not in self._dirs:
            raise FileNotFoundError(f"No such directory: {path}")
        prefix = path if path.endswith("/") else path + "/"
        children = set()
        for p in list(self._files) + list(self._dirs):
            if p.startswith(prefix):
                rest = p[len(prefix) :]
                if rest:
                    children.add(rest.split("/")[0])
        return sorted(children)

    @_locked
    def glob(self, pattern: str) -> list:
        """Glob over file paths. Pattern must be absolute, e.g. '/docs/*.md'.

        `*` matches within one path segment; `**` crosses segments
        (e.g. '/docs/**/*.md' matches '/docs/a.md' and '/docs/sub/b.md').
        """
        pattern = _norm(pattern)
        pat_segs = [s for s in pattern.split("/") if s]
        matches = []
        for path in self._files:
            if _segments_match([s for s in path.split("/") if s], pat_segs):
                matches.append(path)
        return sorted(matches)

    @_locked
    def reset(self) -> None:
        self._files.clear()
        self._dirs = {"/"}
        self._used = 0


def _segments_match(path_segs: list, pat_segs: list) -> bool:
    """Segment-wise glob where '**' matches zero or more segments.

    Dynamic programming, O(len(path) * len(pattern)). The v0.1 recursive
    version was exponential in the number of '**' segments, so an
    agent-supplied pattern could hang an eval run.
    """
    # collapse consecutive '**' (they are equivalent to one)
    pats: list = []
    for seg in pat_segs:
        if seg == "**" and pats and pats[-1] == "**":
            continue
        pats.append(seg)
    n, m = len(path_segs), len(pats)
    # reach[j] = pattern prefix pats[:j] can match path_segs[:i]
    reach = [False] * (m + 1)
    reach[0] = True
    for j in range(1, m + 1):
        reach[j] = reach[j - 1] and pats[j - 1] == "**"
    for i in range(1, n + 1):
        nxt = [False] * (m + 1)
        for j in range(1, m + 1):
            if pats[j - 1] == "**":
                nxt[j] = nxt[j - 1] or reach[j]
            else:
                nxt[j] = reach[j - 1] and fnmatch.fnmatchcase(
                    path_segs[i - 1], pats[j - 1]
                )
        reach = nxt
    return reach[m]


@dataclass
class FakeResponse:
    """Minimal requests-like response."""

    status_code: int
    text: str = ""
    headers: dict = field(default_factory=dict)
    _json: object = None
    _has_json: bool = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        if not self._has_json:
            raise ValueError("No JSON body on this mock response")
        return self._json

    def raise_for_status(self) -> None:
        if not self.ok:
            raise OSError(f"HTTP {self.status_code}")


_UNSET = object()


@dataclass
class _MockRoute:
    method: str
    url_pattern: str
    status: int
    json_body: object
    has_json: bool
    text: str
    headers: dict
    error: str | None  # "timeout" | "connection" | None
    remaining: int = 1  # queue uses; see FakeHTTP.mock


@dataclass
class LoggedRequest:
    method: str
    url: str
    kwargs: dict


class FakeHTTP:
    """Mock HTTP server. Strict: requests with no matching mock raise.

    URL matching: exact match first, then `*` wildcards via fnmatch
    (e.g. mock("GET", "https://api.example.com/users/*")). Query strings
    are part of the matched string — mock the full URL you expect.
    """

    def __init__(self, strict: bool = True):
        self.strict = strict
        self._routes: list[_MockRoute] = []
        self.requests: list[LoggedRequest] = []
        self._lock = threading.RLock()

    @_locked
    def mock(
        self,
        method: str,
        url: str,
        status: int = 200,
        json=_UNSET,
        text: str = "",
        headers: dict | None = None,
        error: str | None = None,
        times: int = 1,
    ) -> None:
        """Register a mock.

        error="timeout" raises TimeoutError, error="connection" raises
        ConnectionError — so evals can verify agents handle failures.

        Mocks for the same method+URL form a queue consumed in registration
        order: mock a 500 then a 200 to test retry logic. When the queue is
        exhausted, the last mock repeats. `times=n` registers the mock n
        times at once.
        """
        if error is not None and error not in ("timeout", "connection"):
            raise ValueError(
                f"unknown error mode {error!r}; use 'timeout' or 'connection'"
            )
        if times < 1:
            raise ValueError("times must be >= 1")
        self._routes.append(
            _MockRoute(
                method=method.upper(),
                url_pattern=url,
                status=status,
                json_body=None if json is _UNSET else json,
                has_json=json is not _UNSET,
                text=text,
                headers=dict(headers or {}),
                error=error,
                remaining=times,
            )
        )

    def _find(self, method: str, url: str) -> _MockRoute | None:
        method = method.upper()
        exact = [
            i
            for i, r in enumerate(self._routes)
            if r.method == method and r.url_pattern == url
        ]
        wild = [
            i
            for i, r in enumerate(self._routes)
            if r.method == method
            and r.url_pattern != url
            and fnmatch.fnmatchcase(url, r.url_pattern)
        ]
        candidates = exact + wild
        for i in candidates:
            if self._routes[i].remaining > 0:
                self._routes[i].remaining -= 1
                return self._routes[i]
        if candidates:
            return self._routes[candidates[-1]]  # queue exhausted: repeat last
        return None

    @_locked
    def request(self, method: str, url: str, **kwargs) -> FakeResponse:
        self.requests.append(LoggedRequest(method.upper(), url, kwargs))
        route = self._find(method, url)
        if route is None:
            if self.strict:
                known = [f"{r.method} {r.url_pattern}" for r in self._routes]
                raise UnmockedRequestError(
                    f"No mock for {method.upper()} {url}. "
                    f"Registered mocks: {known or 'none'}"
                )
            return FakeResponse(status_code=404, text="unmocked")
        if route.error == "timeout":
            raise TimeoutError(f"mocked timeout for {method.upper()} {url}")
        if route.error == "connection":
            raise ConnectionError(f"mocked connection error for {method.upper()} {url}")
        return FakeResponse(
            status_code=route.status,
            text=route.text,
            headers=dict(route.headers),
            _json=route.json_body,
            _has_json=route.has_json,
        )

    # convenience verbs
    def get(self, url: str, **kwargs) -> FakeResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> FakeResponse:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> FakeResponse:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs) -> FakeResponse:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs) -> FakeResponse:
        return self.request("DELETE", url, **kwargs)

    @_locked
    def reset(self) -> None:
        self._routes.clear()
        self.requests.clear()


class KVStore:
    """Tiny key-value scratch space (agent memory, caches, flags)."""

    def __init__(self):
        self._data: dict = {}
        self._lock = threading.RLock()

    @_locked
    def get(self, key, default=...):
        if key in self._data:
            return self._data[key]
        if default is ...:
            raise KeyError(key)
        return default

    @_locked
    def set(self, key, value) -> None:
        self._data[key] = value

    @_locked
    def delete(self, key) -> None:
        if key not in self._data:
            raise KeyError(key)
        del self._data[key]

    @_locked
    def exists(self, key) -> bool:
        return key in self._data

    @_locked
    def clear(self) -> None:
        self._data.clear()


class Clock:
    """Controllable clock for time-dependent agent behavior."""

    def __init__(self, start: datetime | None = None):
        self._now = start or datetime(2026, 1, 1, 12, 0, 0)

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)

    def reset(self) -> None:
        self._now = datetime(2026, 1, 1, 12, 0, 0)


class World:
    """The mock world an agent runs against: FS + HTTP + KV + clock."""

    def __init__(
        self, strict_http: bool = True, fs_max_bytes: int | None = FakeFS.DEFAULT_MAX_BYTES
    ):
        self.fs = FakeFS(max_bytes=fs_max_bytes)
        self.http = FakeHTTP(strict=strict_http)
        self.kv = KVStore()
        self.clock = Clock()

    def reset(self) -> None:
        self.fs.reset()
        self.http.reset()
        self.kv.clear()
        self.clock.reset()
