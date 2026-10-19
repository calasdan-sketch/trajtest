"""The mock world: in-memory FS, mock HTTP, KV store, clock.

Strict by design: missing files raise FileNotFoundError, unmocked HTTP
requests raise UnmockedRequestError. An eval world that forgives everything
produces agents that pass evals and fail in production.
"""

import fnmatch
import posixpath
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from trajtest.errors import UnmockedRequestError


def _norm(path: str) -> str:
    """Normalize to an absolute posix path. `..` cannot escape root."""
    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(f"trajtest FakeFS paths must be absolute, got {path!r}")
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


class FakeFS:
    """In-memory filesystem. Directories are implicit (derived from files)
    unless created with mkdir(). Content may be str or bytes; read()
    returns whatever was written."""

    def __init__(self):
        self._files: dict[str, str | bytes] = {}
        self._dirs: set[str] = {"/"}

    # -- mutation ------------------------------------------------------
    def write(self, path: str, content) -> None:
        path = _norm(path)
        if not isinstance(content, (str, bytes)):
            raise TypeError(
                f"FakeFS.write content must be str or bytes, "
                f"got {type(content).__name__}"
            )
        self._files[path] = content
        self._ensure_parents(path)

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
        self._files[path] = existing + content

    def delete(self, path: str) -> None:
        path = _norm(path)
        if path not in self._files:
            raise FileNotFoundError(f"No such file: {path}")
        del self._files[path]

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
    def read(self, path: str):
        """Return file content as written (str or bytes)."""
        path = _norm(path)
        if path not in self._files:
            raise FileNotFoundError(f"No such file: {path}")
        return self._files[path]

    def exists(self, path: str) -> bool:
        path = _norm(path)
        return path in self._files or path in self._dirs

    def is_file(self, path: str) -> bool:
        return _norm(path) in self._files

    def is_dir(self, path: str) -> bool:
        path = _norm(path)
        return path in self._dirs and path not in self._files

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

    def reset(self) -> None:
        self._files.clear()
        self._dirs = {"/"}


def _segments_match(path_segs: list, pat_segs: list) -> bool:
    """Segment-wise glob where '**' matches zero or more segments."""
    if not pat_segs:
        return not path_segs
    if pat_segs[0] == "**":
        return any(
            _segments_match(path_segs[i:], pat_segs[1:])
            for i in range(len(path_segs) + 1)
        )
    if not path_segs:
        return False
    if not fnmatch.fnmatchcase(path_segs[0], pat_segs[0]):
        return False
    return _segments_match(path_segs[1:], pat_segs[1:])


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

    def reset(self) -> None:
        self._routes.clear()
        self.requests.clear()


class KVStore:
    """Tiny key-value scratch space (agent memory, caches, flags)."""

    def __init__(self):
        self._data: dict = {}

    def get(self, key, default=...):
        if key in self._data:
            return self._data[key]
        if default is ...:
            raise KeyError(key)
        return default

    def set(self, key, value) -> None:
        self._data[key] = value

    def delete(self, key) -> None:
        if key not in self._data:
            raise KeyError(key)
        del self._data[key]

    def exists(self, key) -> bool:
        return key in self._data

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

    def __init__(self, strict_http: bool = True):
        self.fs = FakeFS()
        self.http = FakeHTTP(strict=strict_http)
        self.kv = KVStore()
        self.clock = Clock()

    def reset(self) -> None:
        self.fs.reset()
        self.http.reset()
        self.kv.clear()
        self.clock.reset()
