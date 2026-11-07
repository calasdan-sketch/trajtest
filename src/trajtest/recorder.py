"""Trajectory recorder: runs an agent against a mock world, records everything.

The agent under test is a plain function::

    def my_agent(task: str, tools) -> str:
        data = tools.http_get("https://api.example.com/rows").json()
        tools.fs_write("/out/report.md", render(data))
        return "done"

trajtest instruments the `tools` namespace — every call is recorded with its
name, arguments, result, duration and any error. The mock world supplies the
results. No framework SDK required.
"""

import asyncio
import inspect
import re
import threading
import time
from dataclasses import dataclass, field

from trajtest.errors import AgentTimeout, MaxStepsExceeded
from trajtest.sanitize import clean, clip
from trajtest.world import World

_RESULT_REPR_LIMIT = 500
_LINE_LIMIT = 1000


@dataclass
class ToolCall:
    step: int
    name: str
    args: tuple
    kwargs: dict
    result_repr: str
    duration_ms: float
    error: str | None = None
    bound: dict = field(default_factory=dict)
    """Positional args bound to parameter names (VAR_KEYWORD flattened).

    Lets evals write called("http_get", url__contains="...") even when the
    agent passed the URL positionally — matching is by parameter name.
    """

    def short(self) -> str:
        arg_strs = [clip(repr(a), 400) for a in self.args]
        arg_strs += [f"{k}={clip(repr(v), 400)}" for k, v in self.kwargs.items()]
        err = f" !! {self.error}" if self.error else ""
        # Untrusted text: secrets masked, control/bidi/newline chars escaped.
        line = clean(f"#{self.step} {self.name}({', '.join(arg_strs)}){err}")
        if len(line) > _LINE_LIMIT:
            line = line[:_LINE_LIMIT] + f"…(+{len(line) - _LINE_LIMIT} chars)"
        return line


@dataclass
class Note:
    """Agent narration captured via tools.note(). Notes are NOT steps and
    never appear in tool-call sequences — they exist so evals can compare
    what the agent *said* against what it *did*."""

    step: int  # tool-call count when the note was written
    text: str


@dataclass
class Trajectory:
    calls: list[ToolCall] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    final_answer: object = None
    finished: bool = False
    failure: str | None = None
    max_steps: int = 25
    world: "World | None" = field(default=None, repr=False)
    registered: "list[str] | None" = None
    """Every tool name that existed for this run (built-ins + custom).

    None means unknown (e.g. an imported trace without a tool list): name
    typo checks are skipped rather than guessed.
    """

    @property
    def steps(self) -> int:
        return len(self.calls)

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.calls]

    def to_dict(self) -> dict:
        """JSON-safe snapshot (lossy: non-JSON values become their repr)."""
        from trajtest import traces

        return traces.to_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Trajectory":
        from trajtest import traces

        return traces.from_dict(data)

    def save(self, path, redact: bool = True) -> None:
        """Write the trajectory as JSON (e.g. a golden/baseline trace).

        Secret-looking values are masked by default; redact=False keeps
        raw values. The write is atomic.
        """
        from trajtest import traces

        traces.save(self, path, redact=redact)

    @classmethod
    def load(cls, path) -> "Trajectory":
        from trajtest import traces

        return traces.load(path)

    @classmethod
    def from_openai_messages(cls, messages, tools=None) -> "Trajectory":
        from trajtest import traces

        return traces.from_openai_messages(messages, tools)

    @classmethod
    def from_anthropic_messages(cls, messages, tools=None) -> "Trajectory":
        from trajtest import traces

        return traces.from_anthropic_messages(messages, tools)

    def calls_to(self, name: str) -> list[ToolCall]:
        return [c for c in self.calls if c.name == name]

    def unused_tools(self, registered: "list[str] | None" = None) -> list[str]:
        """Tools that exist but the agent never touched.

        Counters meaningless coverage: a green eval suite where the
        dangerous tools were never exercised is not evidence of safety.
        Defaults to this run's own tool list (built-ins + custom tools).
        """
        if registered is None:
            registered = self.registered or []
        used = set(self.tool_names)
        return [t for t in registered if t not in used]

    def transcript(self) -> str:
        lines = []
        for call in self.calls:
            lines.append(call.short())
        for note in self.notes:
            lines.append(clean(f"(note @ step {note.step}) {note.text}"))
        lines.append(
            clean(
                f"-> finished={self.finished} steps={self.steps} "
                f"answer={clip(repr(self.final_answer), 400)}"
                + (f" failure={clip(repr(self.failure), 400)}" if self.failure else "")
            )
        )
        return "\n".join(lines)


def _truncate_repr(value) -> str:
    from trajtest.sanitize import redact

    # Bound first: repr+redact of a 20 MB value cost seconds. Slightly more
    # than the limit is kept so a secret straddling it is still matched.
    raw = repr(value)
    cut = len(raw) > _RESULT_REPR_LIMIT
    text = redact(raw[: _RESULT_REPR_LIMIT + 200])
    return text[:_RESULT_REPR_LIMIT] + "…" if cut or len(text) > _RESULT_REPR_LIMIT else text


class _Tools:
    """Instrumented tool namespace handed to the agent."""

    #: every tool trajtest exposes, for coverage checks
    REGISTERED = (
        "fs_read",
        "fs_write",
        "fs_append",
        "fs_delete",
        "fs_exists",
        "fs_list",
        "fs_glob",
        "http_get",
        "http_post",
        "http_put",
        "http_patch",
        "http_delete",
        "http_request",
        "kv_get",
        "kv_set",
        "kv_delete",
        "note",
    )

    def __init__(
        self,
        world: World,
        trajectory: Trajectory,
        max_steps: int,
        extra: "dict | None" = None,
    ):
        self._world = world
        self._traj = trajectory
        self._max_steps = max_steps
        self._lock = threading.RLock()
        self._wrapped: dict = {}
        self._cancelled = False
        self._impls = {
            "fs_read": world.fs.read,
            "fs_write": world.fs.write,
            "fs_append": world.fs.append,
            "fs_delete": world.fs.delete,
            "fs_exists": world.fs.exists,
            "fs_list": world.fs.listdir,
            "fs_glob": world.fs.glob,
            "http_get": world.http.get,
            "http_post": world.http.post,
            "http_put": world.http.put,
            "http_patch": world.http.patch,
            "http_delete": world.http.delete,
            "http_request": world.http.request,
            "kv_get": world.kv.get,
            "kv_set": world.kv.set,
            "kv_delete": world.kv.delete,
        }
        for name, fn in (extra or {}).items():
            _check_tool_name(name)
            if name in self._impls or name == "note":
                raise ValueError(
                    f"custom tool {name!r} would shadow a built-in trajtest tool"
                )
            if not callable(fn):
                raise TypeError(f"custom tool {name!r} must be callable")
            self._impls[name] = fn

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "note":
            return self._note
        impls = self.__dict__.get("_impls", {})
        if name in impls:
            # cache: one wrapper (and one signature inspection) per tool
            cache = self.__dict__["_wrapped"]
            if name not in cache:
                cache[name] = self._wrap(name, impls[name])
            return cache[name]
        import difflib

        close = difflib.get_close_matches(name, list(impls) + ["note"], n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        raise AttributeError(f"trajtest has no tool {name!r}{hint}")

    def _note(self, text: str) -> None:
        with self._lock:
            self._traj.notes.append(Note(step=len(self._traj.calls), text=str(text)))

    @staticmethod
    def _signature(fn):
        try:
            return inspect.signature(fn)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _bind_args(sig, args, kwargs) -> dict:
        """Bind positional args to parameter names; flatten **kwargs.

        Returns {} when the signature can't be inspected or the call does
        not fit it — callers fall back to raw args/kwargs matching.
        """
        if sig is None:
            return {}
        try:
            bound = sig.bind_partial(*args, **kwargs)
        except TypeError:
            return {}
        flat: dict = {}
        for name, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                extra = bound.arguments.get(name, {})
                if isinstance(extra, dict):
                    flat.update(extra)
            elif name in bound.arguments:
                flat[name] = bound.arguments[name]
        return flat

    def _wrap(self, name, fn):
        sig = self._signature(fn)

        def record(args, kwargs, bound, result, start, error=None):
            # Step numbering and the budget check are atomic so threads
            # cannot produce duplicate step ids.
            with self._lock:
                if self._cancelled:  # timed-out agent: freeze the trajectory
                    return
                self._traj.calls.append(
                    ToolCall(
                        step=len(self._traj.calls) + 1,
                        name=name,
                        args=args,
                        kwargs=kwargs,
                        result_repr="" if error else _truncate_repr(result),
                        duration_ms=(time.perf_counter() - start) * 1000,
                        error=error,
                        bound=bound,
                    )
                )

        def recorded(*args, **kwargs):
            with self._lock:
                if self._cancelled:
                    raise AgentTimeout("agent was cancelled after run(timeout=...)")
                if len(self._traj.calls) >= self._max_steps:
                    raise MaxStepsExceeded(
                        f"agent exceeded max_steps={self._max_steps} "
                        f"(last tool: {name})"
                    )
            start = time.perf_counter()
            bound = self._bind_args(sig, args, kwargs)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                # Record the failed call, then let the agent see the error —
                # agents that never face errors in evals fall over in prod.
                record(args, kwargs, bound, None, start, f"{type(exc).__name__}: {exc}")
                raise
            record(args, kwargs, bound, result, start)
            return result

        recorded.__name__ = name
        return recorded


_TOOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _check_tool_name(name) -> None:
    if not isinstance(name, str) or not _TOOL_NAME.match(name) or name.startswith("_"):
        raise ValueError(
            f"invalid custom tool name {name!r}: use letters, digits, underscore"
        )


def flaky(fn, *, fail_times: int = 1, error: Exception | None = None):
    """Wrap a custom tool so its first `fail_times` calls raise.

    Lets evals prove an agent retries or degrades gracefully::

        tools = {"send_email": flaky(send, fail_times=2, error=TimeoutError("smtp"))}
    """
    if fail_times < 0:
        raise ValueError("fail_times must be >= 0")
    state = {"left": fail_times}
    lock = threading.Lock()

    def wrapper(*args, **kwargs):
        with lock:
            fail = state["left"] > 0
            if fail:
                state["left"] -= 1
        if fail:
            raise error if error is not None else RuntimeError("injected failure")
        return fn(*args, **kwargs)

    try:
        wrapper.__signature__ = inspect.signature(fn)  # keep param-name matching
    except (TypeError, ValueError):
        pass
    wrapper.__name__ = getattr(fn, "__name__", "flaky_tool")
    return wrapper


#: Every tool name trajtest exposes. Use with Trajectory.unused_tools()
#: to check an eval actually exercised the risky surface:
#:     assert traj.unused_tools(TOOLS) == []  # or review the leftovers
TOOLS = list(_Tools.REGISTERED)


def _accepts_seed(agent) -> bool:
    """Does this agent callable accept a `seed` keyword?"""
    try:
        params = inspect.signature(agent).parameters
    except (TypeError, ValueError):
        raise TypeError("trajtest run(seed=...) requires an inspectable agent callable")
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        or (
            p.kind
            in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            and name == "seed"
        )
        for name, p in params.items()
    )


def _call_agent(agent, task: str, tools, seed):
    """Invoke the agent, injecting seed only if it accepts one.

    Async agents (coroutine functions, or anything returning an awaitable)
    are driven to completion on a private event loop in a worker thread, so
    this also works inside an already-running loop (pytest-asyncio). v0.1
    silently "finished" with an un-awaited coroutine and zero steps, a false
    pass. Tools stay synchronous: call them directly inside the coroutine.
    """
    result = agent(task, tools) if seed is None else agent(task, tools, seed=seed)
    if inspect.isawaitable(result):
        return _drive_awaitable(result)
    return result


def _drive_awaitable(awaitable):
    box: dict = {}

    async def main():
        return await awaitable

    def target():
        try:
            box["value"] = asyncio.run(main())
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller
            box["error"] = exc

    th = threading.Thread(target=target, name="trajtest-async-agent", daemon=True)
    th.start()
    th.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _call_with_timeout(agent, task, handle, seed, timeout):
    box: dict = {}

    def target():
        try:
            box["value"] = _call_agent(agent, task, handle, seed)
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller
            box["error"] = exc

    th = threading.Thread(target=target, name="trajtest-agent", daemon=True)
    th.start()
    th.join(timeout)
    with handle._lock:
        if th.is_alive():
            handle._cancelled = True
            raise AgentTimeout(f"agent still running after {timeout}s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def run(
    agent,
    task: str,
    world: World,
    *,
    max_steps: int = 25,
    seed=None,
    tools: "dict | None" = None,
    timeout: "float | None" = None,
) -> Trajectory:
    """Run an agent against a mock world; record its trajectory.

    agent: callable (task, tools) -> str, sync or async. If seed is given
           the agent must accept a `seed` keyword (used to tame stochastic
           agents).
    tools: optional {name: callable} of custom tools (send_email, shell,
           search, ...). They are recorded exactly like built-ins, appear
           in traj.registered, and cannot shadow a built-in.
    timeout: optional wall-clock seconds. The agent then runs in a daemon
           thread; on expiry the trajectory is frozen with
           failure="AgentTimeout..." and the agent's next tool call raises.
           A Python thread cannot be killed, so an agent stuck in a tool-free
           loop keeps burning CPU in the background until the process exits.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    if seed is not None and not _accepts_seed(agent):
        # Programmer error, not agent behavior: fail loudly instead of
        # recording it as a trajectory failure.
        raise TypeError(
            "run(seed=...) was given but this agent does not accept a "
            "'seed' keyword argument"
        )
    traj = Trajectory(max_steps=max_steps)
    traj.world = world
    handle = _Tools(world, traj, max_steps, tools)  # may raise: setup error
    traj.registered = [n for n in _Tools.REGISTERED] + [
        n for n in (tools or {}) if n not in _Tools.REGISTERED
    ]
    if timeout is not None and not (timeout > 0):
        raise ValueError("timeout must be > 0 seconds")
    try:
        if timeout is None:
            traj.final_answer = _call_agent(agent, task, handle, seed)
        else:
            traj.final_answer = _call_with_timeout(agent, task, handle, seed, timeout)
        traj.finished = True
    except AgentTimeout as exc:
        traj.finished = False
        traj.failure = f"AgentTimeout: {exc}"
    except MaxStepsExceeded as exc:
        traj.finished = False
        traj.failure = str(exc)
    except Exception as exc:  # noqa: BLE001 - the agent's crash is eval data
        traj.finished = False
        traj.failure = f"{type(exc).__name__}: {exc}"
    return traj


def sample(
    agent,
    task: str,
    world_factory,
    *,
    n: int = 5,
    max_steps: int = 25,
    seed_start: int = 0,
    timeout: "float | None" = None,
) -> list[Trajectory]:
    """Run the agent n times against fresh worlds (for stochastic agents).

    world_factory: zero-arg callable returning a new World each run.
    Seeds seed_start..seed_start+n-1 are passed to the agent when it
    accepts a `seed` keyword; otherwise the agent runs unseeded. Use the
    result to assert consistency, e.g. every trajectory satisfies the
    same Expect chain.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    use_seed = _accepts_seed(agent)
    trajs = []
    for i in range(n):
        world = world_factory()
        seed = seed_start + i if use_seed else None
        trajs.append(run(agent, task, world, max_steps=max_steps, seed=seed, timeout=timeout))
    return trajs
