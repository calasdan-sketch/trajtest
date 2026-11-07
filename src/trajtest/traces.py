"""Load recorded agent traces so the SAME assertions run on real runs.

Formats: trajtest's own JSON, OpenAI-style chat messages (assistant
``tool_calls`` + ``tool`` role results) and Anthropic-style messages
(``tool_use`` / ``tool_result`` content blocks). Imported trajectories have
no mock world, so world_* assertions refuse to run on them; every
call-based assertion (order, allowlists, egress, loops) works.

Input is untrusted: size and shape are validated and malformed data raises
ValueError instead of being guessed at. JSON only, never pickle.
"""

import json
import os

from trajtest.recorder import Note, ToolCall, Trajectory

SCHEMA_VERSION = 1
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_CALLS = 100_000
_MAX_DEPTH = 20


def _jsonable(v, depth=0):
    if depth > _MAX_DEPTH:
        return "<max depth>"
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x, depth + 1) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x, depth + 1) for k, x in v.items()}
    return repr(v)


def to_dict(traj: Trajectory) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "finished": traj.finished,
        "failure": traj.failure,
        "final_answer": _jsonable(traj.final_answer),
        "max_steps": traj.max_steps,
        "registered": traj.registered,
        "notes": [{"step": n.step, "text": n.text} for n in traj.notes],
        "calls": [
            {
                "name": c.name,
                "args": _jsonable(c.args),
                "kwargs": _jsonable(c.kwargs),
                "bound": _jsonable(c.bound),
                "result_repr": c.result_repr,
                "error": c.error,
                "duration_ms": c.duration_ms,
            }
            for c in traj.calls
        ],
    }


def _req(cond, msg):
    if not cond:
        raise ValueError(f"invalid trajtest trace: {msg}")


def from_dict(data: dict) -> Trajectory:
    _req(isinstance(data, dict), "top level must be an object")
    _req(data.get("schema") == SCHEMA_VERSION, f"unsupported schema {data.get('schema')!r}")
    raw_calls = data.get("calls")
    _req(isinstance(raw_calls, list), "'calls' must be a list")
    _req(len(raw_calls) <= MAX_CALLS, f"more than {MAX_CALLS} calls")
    traj = Trajectory(
        final_answer=data.get("final_answer"),
        finished=bool(data.get("finished", False)),
        failure=data.get("failure"),
        max_steps=int(data.get("max_steps", 25)),
    )
    reg = data.get("registered")
    _req(reg is None or (isinstance(reg, list) and all(isinstance(x, str) for x in reg)),
         "'registered' must be a list of strings or null")
    traj.registered = reg
    for i, c in enumerate(raw_calls, 1):
        _req(isinstance(c, dict) and isinstance(c.get("name"), str), f"call {i} needs a string name")
        args, kwargs = c.get("args", []), c.get("kwargs", {})
        _req(isinstance(args, list) and isinstance(kwargs, dict), f"call {i} args/kwargs malformed")
        bound = c.get("bound", kwargs)
        _req(isinstance(bound, dict), f"call {i} bound must be an object")
        err = c.get("error")
        _req(err is None or isinstance(err, str), f"call {i} error must be a string or null")
        traj.calls.append(
            ToolCall(
                step=i, name=c["name"], args=tuple(args), kwargs=kwargs,
                result_repr=str(c.get("result_repr", "")),
                duration_ms=float(c.get("duration_ms", 0.0)),
                error=err, bound=dict(bound),
            )
        )
    for n in data.get("notes", []):
        _req(isinstance(n, dict) and isinstance(n.get("text"), str), "malformed note")
        traj.notes.append(Note(step=int(n.get("step", 0)), text=n["text"]))
    return traj


def save(traj: Trajectory, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_dict(traj), f, indent=2)


def load_json_file(path):
    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES:
        raise ValueError(f"trace file is {size} bytes (limit {MAX_FILE_BYTES})")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load(path) -> Trajectory:
    return from_dict(load_json_file(path))


# -- chat-message formats --------------------------------------------------


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p, dict) and p.get("type") == "tool_result":
                parts.append(_text(p.get("content")))
        return "\n".join(parts)
    return repr(content)


def _new_call(traj, name, kwargs, error=None):
    _req(isinstance(name, str) and name, "tool call without a name")
    _req(len(traj.calls) < MAX_CALLS, f"more than {MAX_CALLS} calls")
    call = ToolCall(
        step=len(traj.calls) + 1, name=name, args=(), kwargs=kwargs,
        result_repr="", duration_ms=0.0, error=error, bound=dict(kwargs),
    )
    traj.calls.append(call)
    return call


def _finish(traj, final_text, ended_on_tool_use, tools):
    traj.registered = list(tools) if tools is not None else None
    if ended_on_tool_use or final_text is None:
        traj.finished = False
        traj.failure = "trace ends without a final assistant answer"
    else:
        traj.finished = True
        traj.final_answer = final_text
    return traj


def from_openai_messages(messages, tools=None) -> Trajectory:
    """OpenAI chat-completions style messages -> Trajectory.

    Assistant text that accompanies tool calls becomes a *note* (narration),
    so notes_contain / narration-vs-action checks work on real runs.
    `tools`: optional list of every tool name the agent had; enables
    typo-detection in never_called(). Omit if unknown.
    """
    _req(isinstance(messages, list), "messages must be a list")
    traj = Trajectory()
    by_id: dict = {}
    final_text, last_was_tool_use = None, False
    for msg in messages:
        _req(isinstance(msg, dict), "each message must be an object")
        role = msg.get("role")
        if role == "assistant":
            text = _text(msg.get("content"))
            tcs = list(msg.get("tool_calls") or [])
            if msg.get("function_call"):  # legacy
                tcs.append({"id": None, "function": msg["function_call"]})
            if tcs:
                if text:
                    traj.notes.append(Note(step=len(traj.calls), text=text))
                for tc in tcs:
                    fn = (tc or {}).get("function") or {}
                    raw = fn.get("arguments", "{}")
                    err, kwargs = None, {}
                    try:
                        parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
                        _req(isinstance(parsed, dict), "arguments must be a JSON object")
                        kwargs = parsed
                    except (ValueError, TypeError):
                        err = "MalformedArguments: arguments were not a valid JSON object"
                        kwargs = {"_raw_arguments": str(raw)[:2000]}
                    call = _new_call(traj, fn.get("name"), kwargs, err)
                    if tc.get("id"):
                        by_id[tc["id"]] = call
                last_was_tool_use, final_text = True, None
            else:
                final_text, last_was_tool_use = text, False
        elif role in ("tool", "function"):
            call = by_id.get(msg.get("tool_call_id"))
            if call is not None:
                call.result_repr = repr(_text(msg.get("content")))[:500]
    return _finish(traj, final_text, last_was_tool_use, tools)


def from_anthropic_messages(messages, tools=None) -> Trajectory:
    """Anthropic Messages style (tool_use / tool_result blocks) -> Trajectory."""
    _req(isinstance(messages, list), "messages must be a list")
    traj = Trajectory()
    by_id: dict = {}
    final_text, last_was_tool_use = None, False
    for msg in messages:
        _req(isinstance(msg, dict), "each message must be an object")
        role, content = msg.get("role"), msg.get("content")
        if isinstance(content, str):
            blocks = [{"type": "text", "text": content}]
        else:
            blocks = [b for b in (content or []) if isinstance(b, dict)]
        if role == "assistant":
            text = "\n".join(b["text"] for b in blocks if b.get("type") == "text" and isinstance(b.get("text"), str))
            uses = [b for b in blocks if b.get("type") == "tool_use"]
            if uses:
                if text:
                    traj.notes.append(Note(step=len(traj.calls), text=text))
                for b in uses:
                    inp = b.get("input", {})
                    _req(isinstance(inp, dict), "tool_use input must be an object")
                    call = _new_call(traj, b.get("name"), inp)
                    if b.get("id"):
                        by_id[b["id"]] = call
                last_was_tool_use, final_text = True, None
            else:
                final_text, last_was_tool_use = text, False
        elif role == "user":
            for b in blocks:
                if b.get("type") == "tool_result":
                    call = by_id.get(b.get("tool_use_id"))
                    if call is not None:
                        call.result_repr = repr(_text(b.get("content")))[:500]
                        if b.get("is_error"):
                            call.error = "ToolError: " + _text(b.get("content"))[:200]
    return _finish(traj, final_text, last_was_tool_use, tools)


def detect_and_load(data, tools=None) -> Trajectory:
    """Auto-detect the format of already-parsed JSON."""
    if isinstance(data, dict) and "calls" in data:
        return from_dict(data)
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        data = data["messages"]
    if isinstance(data, list):
        for msg in data:
            if not isinstance(msg, dict):
                continue
            if msg.get("tool_calls") or msg.get("role") in ("tool", "function") or msg.get("function_call"):
                return from_openai_messages(data, tools)
            c = msg.get("content")
            if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result") for b in c):
                return from_anthropic_messages(data, tools)
        # no tool use at all: valid, empty trajectory (openai-shaped)
        return from_openai_messages(data, tools)
    raise ValueError("unrecognised trace format")
