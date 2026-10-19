"""Argument matchers for trajectory assertions.

Matchers let evals assert on tool-call arguments without pinning down every
value exactly::

    exp.called("http_get", url__contains="api.example.com")
    exp.called("fs_write", "/out/report.md", Contains("total"))
"""

import re


class Matcher:
    """Base class. Subclass and implement matches()/describe()."""

    def matches(self, value) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return self.describe()


class _Any(Matcher):
    def matches(self, value) -> bool:
        return True

    def describe(self) -> str:
        return "ANY"


ANY = _Any()


class _Eq(Matcher):
    def __init__(self, expected):
        self.expected = expected

    def matches(self, value) -> bool:
        return value == self.expected

    def describe(self) -> str:
        return repr(self.expected)


class Contains(Matcher):
    """Substring (str) or member (list/tuple/set/dict-keys) check."""

    def __init__(self, sub):
        self.sub = sub

    def matches(self, value) -> bool:
        try:
            return self.sub in value
        except TypeError:
            return False

    def describe(self) -> str:
        return f"contains({self.sub!r})"


class Regex(Matcher):
    def __init__(self, pattern):
        self.pattern = pattern
        self._rx = re.compile(pattern)

    def matches(self, value) -> bool:
        return isinstance(value, str) and self._rx.search(value) is not None

    def describe(self) -> str:
        return f"regex({self.pattern!r})"


class StartsWith(Matcher):
    def __init__(self, prefix):
        self.prefix = prefix

    def matches(self, value) -> bool:
        return isinstance(value, str) and value.startswith(self.prefix)

    def describe(self) -> str:
        return f"startswith({self.prefix!r})"


class EndsWith(Matcher):
    def __init__(self, suffix):
        self.suffix = suffix

    def matches(self, value) -> bool:
        return isinstance(value, str) and value.endswith(self.suffix)

    def describe(self) -> str:
        return f"endswith({self.suffix!r})"


class Gt(Matcher):
    def __init__(self, bound):
        self.bound = bound

    def matches(self, value) -> bool:
        try:
            return value > self.bound
        except TypeError:
            return False

    def describe(self) -> str:
        return f"> {self.bound!r}"


class Lt(Matcher):
    def __init__(self, bound):
        self.bound = bound

    def matches(self, value) -> bool:
        try:
            return value < self.bound
        except TypeError:
            return False

    def describe(self) -> str:
        return f"< {self.bound!r}"


class Ge(Gt):
    def matches(self, value) -> bool:
        try:
            return value >= self.bound
        except TypeError:
            return False

    def describe(self) -> str:
        return f">= {self.bound!r}"


class Le(Lt):
    def matches(self, value) -> bool:
        try:
            return value <= self.bound
        except TypeError:
            return False

    def describe(self) -> str:
        return f"<= {self.bound!r}"


class InstanceOf(Matcher):
    def __init__(self, cls):
        self.cls = cls

    def matches(self, value) -> bool:
        return isinstance(value, self.cls)

    def describe(self) -> str:
        return f"instanceof({self.cls.__name__})"


class Predicate(Matcher):
    """Wrap a plain callable. fn(value) -> bool."""

    def __init__(self, fn, description="predicate(...)"):
        self.fn = fn
        self.description = description

    def matches(self, value) -> bool:
        try:
            return bool(self.fn(value))
        except Exception:  # noqa: BLE001 - a raising predicate is a non-match
            return False

    def describe(self) -> str:
        return self.description


class DictSubset(Matcher):
    """Matches dicts containing at least the given key/value pairs."""

    def __init__(self, subset: dict):
        self.subset = subset

    def matches(self, value) -> bool:
        if not isinstance(value, dict):
            return False
        return all(k in value and value[k] == v for k, v in self.subset.items())

    def describe(self) -> str:
        return f"dict_subset({self.subset!r})"


def to_matcher(value) -> Matcher:
    """Coerce a literal into an equality matcher; pass Matchers through."""
    if isinstance(value, Matcher):
        return value
    return _Eq(value)


# `kwarg__suffix` sugar used by Expect.called()/never_called()/etc.
_SUFFIXES = {
    "contains": Contains,
    "regex": Regex,
    "startswith": StartsWith,
    "endswith": EndsWith,
    "gt": Gt,
    "lt": Lt,
    "ge": Ge,
    "le": Le,
}


def parse_kwarg(key: str, value):
    """Split `name__suffix` into (name, Matcher). Plain keys become equality.

    A literal Matcher value is used as-is (no suffix needed).
    """
    if "__" in key:
        name, suffix = key.rsplit("__", 1)
        if suffix in _SUFFIXES:
            if isinstance(value, Matcher):
                raise ValueError(
                    f"kwarg {key!r}: pass a Matcher OR use __{suffix} sugar, not both"
                )
            return name, _SUFFIXES[suffix](value)
        # Unknown suffix: treat the whole key literally so callers get a
        # clear mismatch instead of silent magic.
        return key, to_matcher(value)
    return key, to_matcher(value)
