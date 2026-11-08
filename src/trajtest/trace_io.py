"""Read and write redbg JSONL traces (stdlib only, no redbg import).

Lets the same Expect assertions run on a *real* recorded agent run, not just
a mock-world run. A loaded trajectory has no mock World, so world assertions
fail with a clear message instead of passing silently.
"""

import gzip
import json
import uuid
from datetime import datetime, timezone

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


def save_trace(traj: Trajectory, path) -> str:
    """Write a Trajectory as a redbg trace (``.gz`` suffix -> gzip).

    Args are written by parameter name; outputs are the recorded result
    reprs (truncated like the transcript). Returns the path as a string.
    """
    session_id = uuid.uuid4().hex
    clock = [0.0]

    def event(etype, **fields):
        return {
            "v": 1, "id": uuid.uuid4().hex,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "mono": clock[0], "type": etype, "session_id": session_id, **fields,
        }

    events = [event("session_start", agent="trajtest", source="trajtest")]
    notes = list(traj.notes)
    for call in traj.calls:
        events += [event("note", text=n.text) for n in notes if n.step == call.step - 1]
        args = call.bound or {
            **{f"arg{i}": a for i, a in enumerate(call.args)}, **call.kwargs
        }
        start = event("tool_call", step=call.step, tool=call.name, args=args)
        clock[0] += call.duration_ms / 1000.0
        result = {"output": call.result_repr} if call.error is None else {"error": call.error}
        events += [start, event("tool_result", step=call.step, call_id=start["id"],
                                tool=call.name, ok=call.error is None, **result)]
    events += [event("note", text=n.text) for n in notes if n.step >= len(traj.calls)]
    if traj.final_answer is not None:
        events.append(event("response", text=str(traj.final_answer)))
    ok = traj.finished and not traj.failure
    end = {"status": "ok"} if ok else {"status": "error", "error": traj.failure}
    events.append(event("session_end", steps=len(traj.calls), **end))
    with _open(path, "wt") as handle:
        for e in events:
            handle.write(json.dumps(e, default=repr, ensure_ascii=False) + "\n")
    return str(path)
