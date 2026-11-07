"""trajtest — pytest for agent trajectories.

Assert on what an agent *did* (tool calls, order, args, world side effects),
not just what it *said*. Local-first, deterministic, no LLM-as-judge.
"""

from trajtest.errors import (
    EvalFailure,
    FSQuotaExceeded,
    MaxStepsExceeded,
    UnmockedRequestError,
)
from trajtest.expect import Expect
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
    Matcher,
    Predicate,
    Regex,
    StartsWith,
)
from trajtest.recorder import TOOLS, Note, ToolCall, Trajectory, flaky, run, sample
from trajtest.stats import PassRate, assert_pass_rate, pass_rate
from trajtest.world import Clock, FakeFS, FakeHTTP, FakeResponse, KVStore, World

__version__ = "0.2.0"

__all__ = [
    "ANY",
    "TOOLS",
    "Clock",
    "Contains",
    "DictSubset",
    "EndsWith",
    "EvalFailure",
    "Expect",
    "FSQuotaExceeded",
    "FakeFS",
    "FakeHTTP",
    "FakeResponse",
    "Ge",
    "Gt",
    "InstanceOf",
    "KVStore",
    "Le",
    "Lt",
    "Matcher",
    "MaxStepsExceeded",
    "Note",
    "PassRate",
    "Predicate",
    "Regex",
    "StartsWith",
    "ToolCall",
    "Trajectory",
    "UnmockedRequestError",
    "World",
    "assert_pass_rate",
    "flaky",
    "pass_rate",
    "run",
    "sample",
]
