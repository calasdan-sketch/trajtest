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

import inspect
import time
from dataclasses import dataclass, field

from trajtest.errors import MaxStepsExceeded
from trajtest.world import World

_RESULT_REPR_LIMIT = 500


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
        arg_strs = [repr(a) for a in self.args]
        arg_strs += [f"{k}={v!r}" for k, v in self.kwargs.items()]
        err = f" !! {self.error}" if self.error else ""
        return f"#{self.step} {self.name}({', '.join(arg_strs)}){err}"


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

    @property
    def steps(self) -> int:
        return len(self.calls)

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.calls]

    def calls_to(self, name: str) -> list[ToolCall]:
        return [c for c in self.calls if c.name == name]

    def unused_tools(self, registered: list[str]) -> list[str]:
        """Tools that exist but the agent never touched.

        Counters meaningless coverage: a green eval suite where the
        dangerous tools were never exercised is not evidence of safety.
        """
        used = set(self.tool_names)
        return [t for t in registered if t not in used]

    def transcript(self) -> str:
        lines = []
        for call in self.calls:
            lines.append(call.short())
        for note in self.notes:
            lines.append(f"(note @ step {note.step}) {note.text}")
        lines.append(
            f"-> finished={self.finished} steps={self.steps} "
            f"answer={self.final_answer!r}"
            + (f" failure={self.failure!r}" if self.failure else "")
        )
        return "\n".join(lines)


def _truncate_repr(value) -> str:
    text = repr(value)
    if len(text) > _RESULT_REPR_LIMIT:
        return text[:_RESULT_REPR_LIMIT] + "…"
    return text


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

    def __init__(self, world: World, trajectory: Trajectory, max_steps: int):
        self._world = world
        self._traj = trajectory
        self._max_steps = max_steps
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

    def __getattr__(self, name: str):
        if name == "note":
            return self._note
        if name in self._impls:
            return self._wrap(name, self._impls[name])
        raise AttributeError(f"trajtest has no tool {name!r}")

    def _note(self, text: str) -> None:
        self._traj.notes.append(Note(step=len(self._traj.calls), text=str(text)))

    @staticmethod
    def _bind_args(fn, args, kwargs) -> dict:
        """Bind positional args to parameter names; flatten **kwargs.

        Returns {} when the signature can't be inspected — callers fall
        back to raw args/kwargs matching.
        """
        try:
            sig = inspect.signature(fn)
            bound = sig.bind_partial(*args, **kwargs)
        except (TypeError, ValueError):
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
        def recorded(*args, **kwargs):
            if len(self._traj.calls) >= self._max_steps:
                raise MaxStepsExceeded(
                    f"agent exceeded max_steps={self._max_steps} (last tool: {name})"
                )
            start = time.perf_counter()
            bound = self._bind_args(fn, args, kwargs)
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                # Record the failed call, then let the agent see the error —
                # agents that never face errors in evals fall over in prod.
                self._traj.calls.append(
                    ToolCall(
                        step=len(self._traj.calls) + 1,
                        name=name,
                        args=args,
                        kwargs=kwargs,
                        result_repr="",
                        duration_ms=(time.perf_counter() - start) * 1000,
                        error=f"{type(exc).__name__}: {exc}",
                        bound=bound,
                    )
                )
                raise
            self._traj.calls.append(
                ToolCall(
                    step=len(self._traj.calls) + 1,
                    name=name,
                    args=args,
                    kwargs=kwargs,
                    result_repr=_truncate_repr(result),
                    duration_ms=(time.perf_counter() - start) * 1000,
                    bound=bound,
                )
            )
            return result

        recorded.__name__ = name
        return recorded


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
    """Invoke the agent, injecting seed only if it accepts one."""
    if seed is None:
        return agent(task, tools)
    return agent(task, tools, seed=seed)


def run(
    agent, task: str, world: World, *, max_steps: int = 25, seed=None
) -> Trajectory:
    """Run an agent against a mock world; record its trajectory.

    agent: callable (task, tools) -> str. If seed is given the agent must
           accept a `seed` keyword (used to tame stochastic agents).
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
    tools = _Tools(world, traj, max_steps)
    try:
        traj.final_answer = _call_agent(agent, task, tools, seed)
        traj.finished = True
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
        trajs.append(run(agent, task, world, max_steps=max_steps, seed=seed))
    return trajs
