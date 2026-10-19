"""Tests for matchers."""

import pytest

from trajtest.matchers import (
    ANY,
    Contains,
    DictSubset,
    EndsWith,
    Ge,
    Gt,
    InstanceOf,
    Le,
    Lt,
    Predicate,
    Regex,
    StartsWith,
    parse_kwarg,
    to_matcher,
)


def test_any_matches_everything():
    assert ANY.matches(None)
    assert ANY.matches(0)


def test_to_matcher_passthrough():
    m = Contains("x")
    assert to_matcher(m) is m
    eq = to_matcher(42)
    assert eq.matches(42)
    assert not eq.matches(43)


def test_contains_str_and_collection():
    assert Contains("ell").matches("hello")
    assert not Contains("z").matches("hello")
    assert Contains(2).matches([1, 2, 3])
    assert not Contains(9).matches([1, 2, 3])
    assert not Contains("x").matches(123)  # non-container -> False, not TypeError


def test_regex():
    assert Regex(r"^a+$").matches("aaa")
    assert not Regex(r"^a+$").matches("aab")
    assert not Regex(r"x").matches(None)


def test_starts_ends_with():
    assert StartsWith("/out").matches("/out/report.md")
    assert not StartsWith("/out").matches("/in/report.md")
    assert EndsWith(".md").matches("/out/report.md")
    assert not EndsWith(".md").matches(None)


def test_comparisons():
    assert Gt(5).matches(6) and not Gt(5).matches(5)
    assert Lt(5).matches(4) and not Lt(5).matches(5)
    assert Ge(5).matches(5)
    assert Le(5).matches(5)
    assert not Gt(5).matches("str")  # incomparable -> False


def test_instanceof():
    assert InstanceOf(dict).matches({})
    assert not InstanceOf(dict).matches([])


def test_predicate():
    assert Predicate(lambda v: v % 2 == 0).matches(4)
    assert not Predicate(lambda v: v % 2 == 0).matches(3)
    # a raising predicate is a non-match, not a crash
    assert not Predicate(lambda v: 1 / 0).matches(1)


def test_dict_subset():
    assert DictSubset({"a": 1}).matches({"a": 1, "b": 2})
    assert not DictSubset({"a": 1}).matches({"a": 2})
    assert not DictSubset({"a": 1}).matches([("a", 1)])


def test_parse_kwarg_plain():
    name, matcher = parse_kwarg("url", "https://x")
    assert name == "url"
    assert matcher.matches("https://x")
    assert not matcher.matches("https://y")


def test_parse_kwarg_suffixes():
    cases = [
        ("url__contains", "api", "https://api.example.com", True),
        ("url__regex", r"\d+", "abc123", True),
        ("p__startswith", "/out", "/out/a", True),
        ("p__endswith", ".md", "/out/a.md", True),
        ("n__gt", 3, 4, True),
        ("n__lt", 3, 2, True),
        ("n__ge", 3, 3, True),
        ("n__le", 3, 3, True),
    ]
    for key, val, sample, expected in cases:
        name, matcher = parse_kwarg(key, val)
        assert matcher.matches(sample) is expected, key
        assert "__" not in name, key


def test_parse_kwarg_matcher_value_no_suffix():
    m = Contains("x")
    name, matcher = parse_kwarg("url", m)
    assert name == "url" and matcher is m


def test_parse_kwarg_matcher_with_suffix_rejected():
    with pytest.raises(ValueError):
        parse_kwarg("url__contains", Contains("x"))


def test_parse_kwarg_unknown_suffix_literal():
    # unknown suffixes are treated as literal kwarg names (explicit > magic)
    name, matcher = parse_kwarg("weird__thing", 1)
    assert name == "weird__thing"
    assert matcher.matches(1)


def test_describe_roundtrip():
    for m_ in [
        ANY,
        Contains("x"),
        Regex("y"),
        StartsWith("a"),
        EndsWith("b"),
        Gt(1),
        Lt(2),
        Ge(3),
        Le(4),
        InstanceOf(int),
        Predicate(bool),
        DictSubset({"a": 1}),
    ]:
        assert isinstance(m_.describe(), str)
        assert isinstance(repr(m_), str)
