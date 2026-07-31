#!/usr/bin/env python3
"""Anthropic Messages API in front of llama-diffusion-gemma-visual-server.

DiffusionGemma is a block-diffusion model, so llama-server cannot serve it: llama-server's
decode loop is causal and incremental, and this model denoises a whole canvas at a time.
What it *can* do is speak the Anthropic Messages API. This adapter borrows that shape and
puts it in front of the diffusion binary, which already accepts OpenAI-format messages,
applies the GGUF's own chat template, and streams committed text per block.

    client --HTTP /v1/messages--> adapter --stdin/stdout--> llama-diffusion-gemma-visual-server

Usage:
    python3 anthropic_adapter.py --model models/.../diffusiongemma-...-Q4_K_M.gguf

Then point any Anthropic client at http://127.0.0.1:8080 .

The backend is strictly one request at a time, so requests are serialized behind a lock.
See README-anthropic-adapter.md for what is and is not supported.
"""

import argparse
import json
import os
import queue
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The visual server detokenizes with special=true so the client can split out reasoning.
# These come from the GGUF's own chat template, which wraps reasoning as
#   <|channel>thought\n ... \n<channel|>
# Note the asymmetry: the closing marker is <channel|>, not <|channel|>.
# --think-open/--think-close override them if a build spells them differently.
DEFAULT_THINK_OPEN = "<|channel>thought"
DEFAULT_THINK_CLOSE = "<channel|>"

# Tool calls are emitted as
#   <|tool_call>call:NAME{key:value,...}<tool_call|>
# where values follow the template's format_argument macro: JSON-like, but strings are
# delimited by <|"|> instead of quotes and keys are bare. Note the same open/close
# asymmetry as the reasoning markers.
TOOL_OPEN = "<|tool_call>"
TOOL_CLOSE = "<tool_call|>"
STR_DELIM = '<|"|>'

# Markers that special=true detokenization keeps but that are not part of the answer.
# <|turn>role\n opens each turn, so the role name goes with it.
STRIP_PATTERNS = [
    re.compile(r"<\|turn>\w*\n?"),
    re.compile(r"<\|think\|>\n?"),
    re.compile(r"<(?:eos|bos|pad|end_of_turn|start_of_turn)>"),
]


class BackendError(RuntimeError):
    """The diffusion subprocess rejected or failed a request."""

    def __init__(self, message, kind="error"):
        super().__init__(message)
        self.kind = kind


# ERR codes the server emits *without* a trailing DONE (it `continue`s the read loop).
# Every other ERR breaks the block loop and still falls through to STATS/DONE.
_ERR_NO_DONE = {"badreq", "parse", "emptyprompt"}


class DiffusionBackend:
    """Owns the llama-diffusion-gemma-visual-server subprocess and its line protocol.

    Protocol recap (see diffusion-gemma-visual-server.cpp):
      stdin  : one line per request, containing a path to a JSON request file
      stdout : F <block> <step> <total> <json-str>   live canvas for this denoising step
               C <block> <json-str>                  cumulative committed answer text
               STATS <key=value ...>                 one summary line
               DONE | ERR <msg>
    """

    def __init__(self, binary, model, ngl=99, maxtok=None, flash_attn=False, verbose=False):
        env = dict(os.environ, NGL=str(ngl), FA="1" if flash_attn else "0")
        if maxtok is not None:
            env["MAXTOK"] = str(maxtok)

        self.verbose = verbose
        self.lock = threading.Lock()
        self.canvas_length = None
        self.maxtok = None
        self.n_vocab = None

        self.proc = subprocess.Popen(
            [binary, model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )

        # stderr carries the model-load chatter plus the one line that reports canvas length.
        # Drain it on a thread so the pipe never fills and blocks the subprocess.
        self._stderr_lines = []
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        ready = self.proc.stdout.readline()
        if not ready.startswith("READY"):
            time.sleep(0.5)  # let the stderr drain catch up so the error is legible
            raise RuntimeError(
                "backend did not come up.\n--- stderr ---\n" + "".join(self._stderr_lines[-40:])
            )
        _, n_vocab, maxtok_resolved = ready.split()
        self.n_vocab = int(n_vocab)
        self.maxtok = int(maxtok_resolved)

        # The ready line on stderr reports the canvas length, which sets the block size and so
        # how max_tokens maps onto n_blocks. Wait briefly for it rather than guessing.
        for _ in range(50):
            if self.canvas_length is not None:
                break
            time.sleep(0.1)
        if self.canvas_length is None:
            raise RuntimeError("backend never reported its canvas length on stderr")

    def _drain_stderr(self):
        for line in self.proc.stderr:
            self._stderr_lines.append(line)
            if self.canvas_length is None:
                m = re.search(r"canvas=(\d+)", line)
                if m:
                    self.canvas_length = int(m.group(1))
            if self.verbose:
                sys.stderr.write("[backend] " + line)

    def blocks_for(self, max_tokens):
        """How many canvas blocks to ask for to cover max_tokens."""
        return max(1, -(-int(max_tokens) // self.canvas_length))

    def generate(self, messages, tools=None, n_blocks=1, seed=None):
        """Run one request. Yields ("frame", ...) | ("commit", text) | ("stats", dict).

        `commit` carries the *cumulative* answer text, not a delta. Frames are the live
        canvas mid-denoise: they rewrite in place, so they are heartbeats, not output.
        """
        req = {
            "seed": random.randint(0, 2**31 - 1) if seed is None else int(seed),
            "n_blocks": int(n_blocks),
            "messages": messages,
        }
        if tools:
            req["tools"] = tools

        with self.lock:
            if self.proc.poll() is not None:
                raise BackendError("backend process has exited", kind="dead")

            fd, path = tempfile.mkstemp(suffix=".json", prefix="dg-req-")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(req, f)

                self.proc.stdin.write(path + "\n")
                self.proc.stdin.flush()

                pending_error = None
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        raise BackendError("backend closed its output stream", kind="dead")
                    line = line.rstrip("\n")

                    if line == "DONE":
                        if pending_error:
                            raise pending_error
                        return
                    if line.startswith("ERR "):
                        parts = line[4:].split()
                        code = parts[0] if parts else "unknown"
                        if code == "toolong":
                            needed, budget = (parts[1], parts[2]) if len(parts) >= 3 else ("?", "?")
                            msg = (
                                f"conversation too long: needs {needed} tokens, context budget is "
                                f"{budget}. Shorten the conversation or raise MAXTOK."
                            )
                        else:
                            msg = f"backend error: {line[4:]}"
                        err = BackendError(msg, kind=code)
                        if code in _ERR_NO_DONE:
                            raise err
                        # toolong / gen: a partial answer may already have been committed,
                        # and STATS + DONE still follow. Keep reading, report at the end.
                        pending_error = err
                        continue
                    if line.startswith("C "):
                        _, _block, payload = line.split(" ", 2)
                        text = json.loads(payload)
                        if self.verbose:
                            sys.stderr.write("[raw commit] %r\n" % text)
                        yield ("commit", text)
                    elif line.startswith("F "):
                        parts = line.split(" ", 4)
                        yield ("frame", int(parts[1]), int(parts[2]), int(parts[3]))
                    elif line.startswith("STATS "):
                        stats = {}
                        for kv in line[6:].split():
                            k, _, v = kv.partition("=")
                            stats[k] = float(v) if "." in v else int(v)
                        yield ("stats", stats)
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


# ---------------------------------------------------------------------------
# Anthropic <-> OpenAI message shape
# ---------------------------------------------------------------------------


def _blocks_to_text(content):
    """Flatten an Anthropic content value into plain text (text-bearing blocks only)."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        btype = block.get("type")
        if btype == "text":
            parts.append(block.get("text", ""))
        elif btype == "thinking":
            continue  # prior turns' reasoning is not replayed into the prompt
        elif btype == "image":
            parts.append("[image omitted: this model is text-only]")
    return "\n".join(p for p in parts if p)


def _convert_message(msg, out):
    """Append the OpenAI messages equivalent to one Anthropic message.

    One Anthropic message can become several: the API carries tool results as blocks inside
    a user message, while the chat template wants them as separate role="tool" messages.
    """
    role = msg.get("role", "user")
    content = msg.get("content")
    blocks = content if isinstance(content, list) else None

    if blocks is None:
        out.append({"role": role, "content": content or ""})
        return

    # Tool results come first: they answer the preceding assistant turn.
    for block in blocks:
        if block.get("type") == "tool_result":
            inner = block.get("content")
            rendered = inner if isinstance(inner, str) else _blocks_to_text(inner)
            if block.get("is_error"):
                rendered = f"Error: {rendered}"
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": rendered,
                }
            )

    text = _blocks_to_text(blocks)
    tool_calls = [
        {
            "id": b.get("id", ""),
            "type": "function",
            "function": {
                "name": b.get("name", ""),
                "arguments": json.dumps(b.get("input", {})),
            },
        }
        for b in blocks
        if b.get("type") == "tool_use"
    ]

    if tool_calls:
        out.append({"role": role, "content": text, "tool_calls": tool_calls})
    elif text or not any(b.get("type") == "tool_result" for b in blocks):
        # Skip a user message that carried nothing but tool results.
        out.append({"role": role, "content": text})


def anthropic_to_oai(body):
    """Convert an Anthropic Messages request into the OpenAI messages the backend wants."""
    messages = []

    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _blocks_to_text(system)})

    for msg in body.get("messages", []):
        _convert_message(msg, messages)

    tools = None
    if body.get("tools"):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name"),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in body["tools"]
            if t.get("name")
        ]

    return messages, tools


class _ArgParser:
    """Recursive-descent parser for one tool call's argument body.

    The grammar is JSON with two substitutions: strings are wrapped in <|"|> rather than
    double quotes, and object keys are bare identifiers. A regex cannot do this -- a string
    value is free to contain commas, colons and braces.
    """

    def __init__(self, src, pos=0):
        self.src = src
        self.pos = pos

    def _skip_ws(self):
        while self.pos < len(self.src) and self.src[self.pos].isspace():
            self.pos += 1

    def _at(self, token):
        return self.src.startswith(token, self.pos)

    def parse_value(self):
        self._skip_ws()
        if self._at(STR_DELIM):
            return self.parse_string()
        if self._at("{"):
            return self.parse_object()
        if self._at("["):
            return self.parse_array()
        return self.parse_scalar()

    def parse_string(self):
        self.pos += len(STR_DELIM)
        end = self.src.find(STR_DELIM, self.pos)
        if end < 0:
            raise ValueError("unterminated string")
        value = self.src[self.pos:end]
        self.pos = end + len(STR_DELIM)
        return value

    def parse_object(self):
        self.pos += 1  # '{'
        obj = {}
        self._skip_ws()
        if self._at("}"):
            self.pos += 1
            return obj
        while True:
            self._skip_ws()
            key = self.parse_string() if self._at(STR_DELIM) else self.parse_bare_key()
            self._skip_ws()
            if not self._at(":"):
                raise ValueError(f"expected ':' after key {key!r}")
            self.pos += 1
            obj[key] = self.parse_value()
            self._skip_ws()
            if self._at(","):
                self.pos += 1
                continue
            if self._at("}"):
                self.pos += 1
                return obj
            raise ValueError("expected ',' or '}' in object")

    def parse_array(self):
        self.pos += 1  # '['
        items = []
        self._skip_ws()
        if self._at("]"):
            self.pos += 1
            return items
        while True:
            items.append(self.parse_value())
            self._skip_ws()
            if self._at(","):
                self.pos += 1
                continue
            if self._at("]"):
                self.pos += 1
                return items
            raise ValueError("expected ',' or ']' in array")

    def parse_bare_key(self):
        start = self.pos
        while self.pos < len(self.src) and self.src[self.pos] not in ":,}":
            self.pos += 1
        key = self.src[start:self.pos].strip()
        if not key:
            raise ValueError("empty object key")
        return key

    def parse_scalar(self):
        """An unquoted token: number, true/false/null, or bare text as a last resort."""
        start = self.pos
        depth = 0
        while self.pos < len(self.src):
            ch = self.src[self.pos]
            if ch in "[{":
                depth += 1
            elif ch in "]}":
                if depth == 0:
                    break
                depth -= 1
            elif ch == "," and depth == 0:
                break
            self.pos += 1
        raw = self.src[start:self.pos].strip()
        if raw in ("true", "false"):
            return raw == "true"
        if raw in ("null", "None"):
            return None
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        return raw


def parse_tool_call(body):
    """Parse "call:NAME{args}" into (name, input_dict). Raises ValueError if malformed."""
    if not body.startswith("call:"):
        raise ValueError("tool call does not start with 'call:'")
    brace = body.find("{", 5)
    if brace < 0:
        raise ValueError("tool call has no argument block")
    name = body[5:brace].strip()
    if not name:
        raise ValueError("tool call has no name")
    parser = _ArgParser(body, brace)
    args = parser.parse_object()
    return name, args


def split_segments(text, think_open, think_close):
    """Split raw model output into ordered segments.

    Returns a list of (kind, payload):
      ("thinking", str) | ("text", str) | ("tool_use", {"name": str, "input": dict})

    A tool call whose closing marker has not arrived yet is withheld rather than leaked as
    text: during streaming the answer arrives one 256-token block at a time, so a call can
    straddle a block boundary and be genuinely incomplete.
    """
    for pattern in STRIP_PATTERNS:
        text = pattern.sub("", text)

    segments = []
    pos = 0
    while pos < len(text):
        think_at = text.find(think_open, pos)
        tool_at = text.find(TOOL_OPEN, pos)
        if think_at < 0 and tool_at < 0:
            break
        # whichever marker comes first
        if tool_at < 0 or (0 <= think_at < tool_at):
            if think_at > pos:
                segments.append(("text", text[pos:think_at]))
            body_start = think_at + len(think_open)
            end = text.find(think_close, body_start)
            if end < 0:
                # Still open. Whether this is reasoning or the answer itself cannot be known
                # until generation ends -- see finalize_segments.
                segments.append(("thinking_open", text[body_start:]))
                pos = len(text)
                break
            segments.append(("thinking", text[body_start:end]))
            pos = end + len(think_close)
        else:
            if tool_at > pos:
                segments.append(("text", text[pos:tool_at]))
            body_start = tool_at + len(TOOL_OPEN)
            end = text.find(TOOL_CLOSE, body_start)
            if end < 0:
                pos = len(text)  # incomplete: withhold it
                break
            try:
                name, args = parse_tool_call(text[body_start:end])
                segments.append(("tool_use", {"name": name, "input": args}))
            except ValueError:
                # Unparseable: surface the raw text rather than silently dropping output.
                segments.append(("text", text[tool_at:end + len(TOOL_CLOSE)]))
            pos = end + len(TOOL_CLOSE)

    if pos < len(text):
        segments.append(("text", text[pos:]))
    return [(k, v) for k, v in segments if v]


def finalize_segments(segments, truncated):
    """Resolve reasoning that was never closed, now that generation has ended.

    This model frequently opens <|channel>thought, writes its final answer there, and stops
    without ever emitting the closing <channel|>. In a sample of six identical tool-followup
    requests, four ended that way, and the withheld content was the answer verbatim. Reporting
    it as reasoning would hand the client an empty response.

    A run cut short by the token budget is the opposite case: reasoning really was in progress
    and really was truncated, so it stays reasoning.
    """
    return [
        ("thinking" if truncated else "text", payload) if kind == "thinking_open" else (kind, payload)
        for kind, payload in segments
    ]


# ---------------------------------------------------------------------------
# Response assembly
# ---------------------------------------------------------------------------


def _truncated(stats, n_blocks, canvas_length, errored):
    """Did generation run out of budget rather than finish?

    The backend stops a block loop either because the canvas hit an end-of-generation token
    or a repetition loop (answer complete), or because it ran out of blocks. A run that used
    every block and filled every canvas was cut off by the budget.
    """
    if errored:
        return True
    if not stats:
        return False
    return (
        stats.get("blocks", 0) >= n_blocks
        and stats.get("predicted_n", 0) >= n_blocks * canvas_length
    )


def _stop_reason(truncated, has_tool_use=False):
    if truncated:
        return "max_tokens"
    if has_tool_use:
        return "tool_use"
    return "end_turn"


def _content_block(kind, payload):
    """Render one parsed segment as an Anthropic content block."""
    if kind == "thinking":
        return {"type": "thinking", "thinking": payload}
    if kind == "tool_use":
        # Stamp the id on first use so streaming start/delta/stop all agree on it.
        payload.setdefault("id", "toolu_" + uuid.uuid4().hex[:24])
        return {
            "type": "tool_use",
            "id": payload["id"],
            "name": payload["name"],
            "input": payload["input"],
        }
    return {"type": "text", "text": payload}


def _usage(stats):
    return {
        "input_tokens": int(stats.get("prompt_n", 0)) if stats else 0,
        "output_tokens": int(stats.get("predicted_n", 0)) if stats else 0,
    }


class MessagesHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "diffusiongemma-anthropic-adapter"

    # -- plumbing ----------------------------------------------------------

    def log_message(self, fmt, *args):
        if self.server.adapter_verbose:
            sys.stderr.write("[http] %s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status, payload):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_error(self, status, err_type, message):
        self._send_json(
            status, {"type": "error", "error": {"type": err_type, "message": message}}
        )

    def _authorized(self):
        expected = self.server.api_key
        if not expected:
            return True
        given = self.headers.get("x-api-key") or ""
        if not given:
            auth = self.headers.get("authorization") or ""
            if auth.lower().startswith("bearer "):
                given = auth[7:]
        return given == expected

    def _route(self):
        """Path with the query string and any trailing slash removed.

        Clients append query parameters -- Claude Code posts to /v1/messages?beta=true --
        so matching on self.path directly would 404 a perfectly valid request.
        """
        return urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"

    def do_HEAD(self):
        # Claude Code preflights connectivity with HEAD /api/hello before it will talk to
        # an endpoint. BaseHTTPRequestHandler answers 501 to any HEAD it is not told about.
        if self._route() in ("/api/hello", "/health"):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_GET(self):
        if self._route() == "/api/hello":
            self._send_json(200, {"status": "ok"})
            return
        if self._route() == "/health":
            backend = self.server.backend
            self._send_json(
                200,
                {
                    "status": "ok",
                    "canvas_length": backend.canvas_length,
                    "max_context": backend.maxtok,
                    "n_vocab": backend.n_vocab,
                },
            )
        else:
            self._send_error(404, "not_found", f"unknown route {self._route()}")

    def do_POST(self):
        if not self._authorized():
            self._send_error(401, "authentication_error", "invalid x-api-key")
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            self._send_error(400, "invalid_request_error", f"malformed JSON body: {e}")
            return

        route = self._route()
        if route == "/v1/messages/count_tokens":
            self._handle_count_tokens(body)
        elif route == "/v1/messages":
            self._handle_messages(body)
        else:
            self._send_error(404, "not_found", f"unknown route {route}")

    # -- endpoints ---------------------------------------------------------

    def _handle_count_tokens(self, body):
        """Rough character-based estimate.

        The backend tokenizes internally and exposes no tokenize command, so the adapter has
        no access to the real vocabulary. This is an approximation and will not match the
        model's own count. Real input_tokens are reported in the usage of an actual response.
        """
        messages, _ = anthropic_to_oai(body)
        chars = sum(len(m["content"]) for m in messages)
        self._send_json(200, {"input_tokens": max(1, chars // 4), "approximate": True})

    def _handle_messages(self, body):
        if "max_tokens" not in body:
            self._send_error(400, "invalid_request_error", "max_tokens is required")
            return
        if not body.get("messages"):
            self._send_error(400, "invalid_request_error", "messages must be non-empty")
            return

        backend = self.server.backend
        messages, tools = anthropic_to_oai(body)
        n_blocks = backend.blocks_for(body["max_tokens"])
        model_name = body.get("model", self.server.model_name)

        unsupported = [
            k
            for k in ("temperature", "top_p", "top_k", "stop_sequences")
            if body.get(k) not in (None, [], "")
        ]
        if unsupported and self.server.adapter_verbose:
            sys.stderr.write(
                "[adapter] ignoring unsupported sampling parameters: %s\n" % ", ".join(unsupported)
            )

        try:
            if body.get("stream"):
                self._stream(backend, messages, tools, n_blocks, model_name)
            else:
                self._complete(backend, messages, tools, n_blocks, model_name)
        except BackendError as e:
            if e.kind == "toolong":
                self._send_error(400, "invalid_request_error", str(e))
            else:
                self._send_error(500, "api_error", str(e))
        except BrokenPipeError:
            pass  # client hung up mid-stream

    def _complete(self, backend, messages, tools, n_blocks, model_name):
        text, stats, errored = "", None, False
        try:
            for event in backend.generate(messages, tools, n_blocks):
                if event[0] == "commit":
                    text = event[1]
                elif event[0] == "stats":
                    stats = event[1]
        except BackendError as e:
            if e.kind != "toolong" or not text:
                raise
            errored = True  # partial answer: return it and say it was cut off

        truncated = _truncated(stats, n_blocks, backend.canvas_length, errored)
        segments = finalize_segments(
            split_segments(text, self.server.think_open, self.server.think_close),
            truncated,
        )
        content = [_content_block(kind, payload) for kind, payload in segments]
        if not content:
            content = [{"type": "text", "text": ""}]
        has_tool_use = any(k == "tool_use" for k, _ in segments)

        self._send_json(
            200,
            {
                "id": "msg_" + uuid.uuid4().hex[:24],
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": content,
                "stop_reason": _stop_reason(truncated, has_tool_use),
                "stop_sequence": None,
                "usage": _usage(stats),
            },
        )

    def _stream(self, backend, messages, tools, n_blocks, model_name):
        # An SSE body has no Content-Length, so on HTTP/1.1 the client can only know the
        # response ended when the connection closes. Say so up front and close at the end,
        # otherwise clients hang after message_stop waiting for more bytes.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(event, data):
            chunk = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
            self.wfile.write(chunk)
            self.wfile.flush()

        msg_id = "msg_" + uuid.uuid4().hex[:24]
        emit(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model_name,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

        # Emitted segments, so cumulative commits can be turned into deltas.
        sent = []       # list of [kind, payload_already_sent]
        open_index = None
        stats, errored = None, False
        last_text = ""
        last_ping = time.time()

        def open_block(index, kind, payload=None):
            if kind == "tool_use":
                block = dict(_content_block(kind, payload), input={})
            elif kind == "thinking":
                block = {"type": "thinking", "thinking": ""}
            else:
                block = {"type": "text", "text": ""}
            emit(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": block},
            )

        def emit_delta(index, kind, payload):
            if kind == "tool_use":
                # Only complete tool calls are ever emitted, so the whole argument object
                # goes out as a single partial_json chunk.
                delta = {"type": "input_json_delta", "partial_json": json.dumps(payload["input"])}
            elif kind == "thinking":
                delta = {"type": "thinking_delta", "thinking": payload}
            else:
                delta = {"type": "text_delta", "text": payload}
            emit(
                "content_block_delta",
                {"type": "content_block_delta", "index": index, "delta": delta},
            )

        def close_block(index):
            emit("content_block_stop", {"type": "content_block_stop", "index": index})

        try:
            for event in backend.generate(messages, tools, n_blocks):
                if event[0] == "frame":
                    # A block can denoise for many seconds with no committed text. Frames are
                    # the only sign of life, so use them to keep the connection warm.
                    if time.time() - last_ping > 5:
                        emit("ping", {"type": "ping"})
                        last_ping = time.time()
                    continue
                if event[0] == "stats":
                    stats = event[1]
                    continue

                last_text = event[1]
                segments = split_segments(
                    last_text, self.server.think_open, self.server.think_close
                )
                # A trailing unclosed reasoning block is withheld: it may turn out to be the
                # answer rather than reasoning, and a delta already sent cannot be retracted.
                # finalize_segments settles it once generation ends.
                if segments and segments[-1][0] == "thinking_open":
                    segments = segments[:-1]
                for i, (kind, full) in enumerate(segments):
                    if i < len(sent):
                        prev_kind, prev_payload = sent[i]
                        if prev_kind != kind:
                            continue  # a settled segment should not change kind
                        if kind == "tool_use":
                            continue  # atomic: emitted whole, never revised
                        if len(full) > len(prev_payload) and full.startswith(prev_payload):
                            delta = full[len(prev_payload):]
                        else:
                            # Committed text is append-only; if it ever is not, resync rather
                            # than emit a delta that would corrupt the client's copy.
                            delta = ""
                        if delta:
                            emit_delta(i, kind, delta)
                        sent[i] = [kind, full]
                    else:
                        if open_index is not None and open_index < i:
                            close_block(open_index)
                        open_block(i, kind, full)
                        open_index = i
                        if full:
                            emit_delta(i, kind, full)
                        sent.append([kind, full])
                last_ping = time.time()
        except BackendError as e:
            if e.kind != "toolong" or not sent:
                emit(
                    "error",
                    {"type": "error", "error": {"type": "api_error", "message": str(e)}},
                )
                return
            errored = True

        # Flush whatever was withheld, now that the outcome settles what it was.
        truncated = _truncated(stats, n_blocks, backend.canvas_length, errored)
        final = finalize_segments(
            split_segments(last_text, self.server.think_open, self.server.think_close),
            truncated,
        )
        for i in range(len(sent), len(final)):
            kind, payload = final[i]
            if open_index is not None and open_index < i:
                close_block(open_index)
            open_block(i, kind, payload)
            open_index = i
            if payload:
                emit_delta(i, kind, payload)
            sent.append([kind, payload])

        if open_index is None:
            open_block(0, "text")
            open_index = 0
        close_block(open_index)

        emit(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _stop_reason(
                        truncated, any(k == "tool_use" for k, _ in sent)
                    ),
                    "stop_sequence": None,
                },
                "usage": _usage(stats),
            },
        )
        emit("message_stop", {"type": "message_stop"})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="path to the DiffusionGemma GGUF")
    ap.add_argument(
        "--binary",
        default="build/bin/llama-diffusion-gemma-visual-server",
        help="path to llama-diffusion-gemma-visual-server",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ngl", type=int, default=99, help="layers to offload to the GPU")
    ap.add_argument(
        "--maxtok",
        type=int,
        default=None,
        help="context budget in tokens (default: let the backend auto-size)",
    )
    ap.add_argument("--flash-attn", action="store_true")
    ap.add_argument(
        "--api-key",
        default=os.environ.get("ADAPTER_API_KEY"),
        help="require this key in x-api-key (default: no auth)",
    )
    ap.add_argument("--think-open", default=DEFAULT_THINK_OPEN)
    ap.add_argument("--think-close", default=DEFAULT_THINK_CLOSE)
    ap.add_argument("-v", "--verbose", action="store_true", help="echo backend stderr")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit(f"backend binary not found: {args.binary}\nBuild it with:\n"
                 f"  cmake --build build -j --target llama-diffusion-gemma-visual-server")
    if not os.path.exists(args.model):
        sys.exit(f"model not found: {args.model}")

    print(f"loading {args.model} ...", file=sys.stderr)
    backend = DiffusionBackend(
        args.binary, args.model, ngl=args.ngl, maxtok=args.maxtok,
        flash_attn=args.flash_attn, verbose=args.verbose,
    )
    print(
        f"backend ready: canvas={backend.canvas_length} max_context={backend.maxtok}",
        file=sys.stderr,
    )

    httpd = ThreadingHTTPServer((args.host, args.port), MessagesHandler)
    httpd.backend = backend
    httpd.api_key = args.api_key
    httpd.model_name = os.path.basename(args.model)
    httpd.think_open = args.think_open
    httpd.think_close = args.think_close
    httpd.adapter_verbose = args.verbose

    print(
        f"Anthropic Messages API on http://{args.host}:{args.port}/v1/messages",
        file=sys.stderr,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down ...", file=sys.stderr)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
