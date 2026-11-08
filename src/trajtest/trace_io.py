"""Read redbg JSONL traces into a Trajectory (stdlib only, no redbg import).

Lets the same Expect assertions run on a *real* recorded agent run, not just
a mock-world run. A loaded trajectory has no mock World, so world assertions
fail with a clear message instead of passing silently.
"""

import gzip
import json

from trajtest.recorder import Note, ToolCall, Trajectory


def _open(path, mode="rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def load_trace(path) -> Trajectory:
    """Load a redbg trace (``.jsonl`` or ``.jsonl.gz``) as a Trajectory.

    tool_call/tool_result pairs become calls (args as keyword args), the last
    ``response`` becomes the final answer, ``note`` events become notes and
    ``session_end`` status ``ok`` means finished. A call with no result gets
    ``error="no result recorded"``.
    """
    traj = Trajectory(loaded_from=str(path))
    by_id: dict = {}  # call event id -> (ToolCall, call mono)
    status = None
    with _open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: malformed JSON: {exc}") from exc
            etype = event.get("type")
            if etype == "tool_call":
                args = event.get("args")
                args = args if isinstance(args, dict) else {"input": args}
                call = ToolCall(
                    step=len(traj.calls) + 1,
                    name=str(event.get("tool", "unknown")),
                    args=(),
                    kwargs=args,
                    result_repr="",
                    duration_ms=0.0,
                    error="no result recorded",
                    bound=dict(args),
                )
                traj.calls.append(call)
                by_id[event.get("id")] = (call, event.get("mono"))
            elif etype == "tool_result" and event.get("call_id") in by_id:
                call, start = by_id[event["call_id"]]
                if event.get("ok"):
                    call.error = None
                    call.result_repr = str(event.get("output") or "")
                else:
                    call.error = str(event.get("error") or "failed")
                try:
                    call.duration_ms = (event["mono"] - start) * 1000.0
                except (KeyError, TypeError):
                    pass
            elif etype == "response":
                traj.final_answer = event.get("text")
            elif etype == "note":
                traj.notes.append(Note(step=len(traj.calls), text=str(event.get("text", ""))))
            elif etype == "session_end":
                status = event.get("status")
    traj.finished = status == "ok"
    if status is None:
        traj.failure = "no session_end in recording (crashed or still running)"
    elif status != "ok":
        traj.failure = f"session ended with status {status!r}"
    return traj
