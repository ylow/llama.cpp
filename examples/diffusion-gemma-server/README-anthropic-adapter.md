# Anthropic Messages API for DiffusionGemma

This is a small HTTP adapter that puts the Anthropic Messages API in front of DiffusionGemma.

```
client --HTTP /v1/messages--> anthropic_adapter.py --stdin/stdout--> llama-diffusion-gemma-visual-server
                                                                              |
                                                                     diffusiongemma Q4_K_M
```

## Why this exists instead of just using llama-server

`llama-server` already speaks the Anthropic Messages API — it has `/v1/messages`, Anthropic
SSE streaming, `/v1/messages/count_tokens`, the whole thing. So the obvious move is to point
it at the GGUF and be done.

That does not work. `llama-server` has no diffusion support at all — grep the directory, there
isn't a single mention. Its decode loop is causal and incremental: one token, append to the KV
cache, repeat. DiffusionGemma does not generate that way. It denoises an entire 256-token canvas
in place over many steps, non-causally, then commits the block and starts the next one. There is
no "next token" to append.

Meanwhile `llama-diffusion-gemma-visual-server` *does* run the model correctly, and — this is the
lucky part — it already accepts OpenAI-format `messages`, applies the GGUF's own chat template,
and streams committed text per block. It just isn't an HTTP server. Its name is misleading: it
speaks a line protocol over stdin/stdout, one request-file path per line.

So the adapter is glue. It does no inference and no tokenization. It translates request and
response shapes, and it owns the subprocess.

## Running it

There is a wrapper script, `dg.sh`, with three subcommands:

```bash
./dg.sh init     # configure, build the binaries, download the weights, run the tests
./dg.sh start    # serve on http://127.0.0.1:8080 in the foreground (Ctrl-C stops it)
./dg.sh code     # point Claude Code at the running server (needs 'start' in another terminal)
```

`init` is safe to re-run — it skips the download if the weights are already there. Everything
is overridable from the environment: `DG_PORT`, `DG_NGL`, `DG_QUANT`, `DG_MODEL`, `DG_MAXTOK`,
`DG_API_KEY`, `DG_FLASH_ATTN`, `DG_VERBOSE`. Run `./dg.sh` with no arguments for the full list.

Extra arguments pass through, so `./dg.sh start -v` turns on verbose logging and
`./dg.sh code --help` reaches `claude`.

### Doing it by hand

Build the backend and download the weights:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-diffusion-gemma-visual-server
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF --include '*Q4_K_M*' --local-dir models/diffusiongemma
```

Then start the adapter:

```bash
python3 examples/diffusion-gemma-server/anthropic_adapter.py \
  --model models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  --ngl 99 --port 8080
```

No dependencies beyond the Python standard library. It prints the resolved canvas length and
context budget once the model is up, which takes a while — the Q4_K_M weights are about 17 GB.

Point any Anthropic client at it:

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"diffusiongemma","max_tokens":512,
       "messages":[{"role":"user","content":"Why is the sky blue?"}]}'
```

Or with the SDK:

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://127.0.0.1:8080", api_key="unused")
msg = client.messages.create(model="diffusiongemma", max_tokens=512,
                             messages=[{"role": "user", "content": "Hello"}])
```

Useful flags: `--maxtok N` pins the context budget instead of letting the backend auto-size it,
`--api-key KEY` requires that key in `x-api-key`, `-v` echoes the backend's stderr, and
`--flash-attn` turns on flash attention.

## What works

- `POST /v1/messages`, streaming and non-streaming
- `system`, multi-turn `messages`, and content-block or plain-string content
- Reasoning, surfaced as Anthropic `thinking` blocks. The model wraps reasoning in
  `<|channel>thought ... <channel|>` and the adapter splits it out.
- Tool calling, both directions — `tools` go in, `tool_use` blocks come out, and `tool_result`
  goes back for the next turn. See below.
- `stop_reason` of `end_turn`, `max_tokens` or `tool_use`, and real `usage` token counts
- `GET /health`, which reports canvas length and context budget

## What does not work, and why

Be aware of these before you wire an agent up to it.

**Sampling parameters are ignored.** `temperature`, `top_p`, `top_k` and `stop_sequences` are
accepted and silently dropped — run with `-v` and the adapter will say so. The backend protocol
takes exactly three knobs: `seed`, `n_blocks`, and `messages`. Sampling happens on the GPU inside
the entropy-bound decoder and is not exposed. This is a backend limitation, not an oversight
here; wiring it up means adding fields to the C++ line protocol.

**`count_tokens` is an estimate.** It returns `characters / 4` and marks the response
`"approximate": true`. The adapter has no tokenizer; the backend tokenizes internally and offers
no tokenize command. The `usage` on a real response is exact — only the pre-flight count is not.

**One request at a time.** The backend protocol is synchronous, so requests serialize behind a
lock. No slots, no continuous batching. Fine for one person at a terminal, not for a shared endpoint.

**Streaming is per block, not per token.** Deltas are emitted when a 256-token block commits, so
output arrives in chunks with pauses between them. This is inherent to block diffusion — there is
no token-by-token stream to forward. The adapter sends `ping` events during a block so clients
do not time out.

**Text only.** Image blocks in a request are replaced with a placeholder note.

## Tool calling

This works, but the model's wire format is its own, so it is worth knowing how it is decoded.

The C++ side only ever handled half of this. The `tools` field is passed through to the chat
template, so the model sees the definitions — but nothing parsed the response back, so a tool
call used to arrive as a blob of text. The adapter does that parsing.

The model emits calls like this:

```
<|tool_call>call:send_email{to:<|"|>alice@example.com<|"|>,subject:<|"|>Hi<|"|>}<tool_call|>
```

It is JSON with two substitutions: strings are wrapped in `<|"|>` instead of double quotes, and
object keys are bare. Note the open/close asymmetry — `<|tool_call>` but `<tool_call|>` — the same
quirk the reasoning markers have.

This gets a real recursive-descent parser rather than a regex, because a string value is free to
contain commas, colons and braces. `body:<|"|>Hello, how are you?<|"|>` is an ordinary thing for
the model to emit and a regex would split it at the comma.

Two cases are handled deliberately:

- **An incomplete call is withheld, not leaked.** Output arrives one 256-token block at a time,
  so a call can straddle a block boundary. Until `<tool_call|>` arrives, nothing is emitted.
- **An unparseable call is surfaced as text**, not dropped. If the model emits something the
  grammar does not cover, you see it rather than getting a silently empty response.

Input conversion goes the other way: Anthropic carries tool results as blocks inside a *user*
message, while the chat template wants separate `role: "tool"` messages, so one Anthropic message
can become several OpenAI ones.

Streaming emits `tool_use` blocks whole — one `input_json_delta` with the complete arguments —
since only finished calls are ever emitted.

## Claude Code against this endpoint

`./dg.sh code` sets `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`/`ANTHROPIC_API_KEY`,
`ANTHROPIC_MODEL`, and the small-model overrides (`ANTHROPIC_DEFAULT_HAIKU_MODEL` and the older
`ANTHROPIC_SMALL_FAST_MODEL`, so background calls also land here rather than on the real API).

It works for short prompts — `./dg.sh code -p "Reply with exactly: OK"` returns `OK` from the
local model. **It does not work for real agentic work.** Claude Code's system prompt plus its
tool definitions measured **23476 tokens**, against a context budget of **20480**, so anything
using the full tool set comes back as `conversation too long`.

Raising `DG_MAXTOK` does not currently help. The backend auto-sizes the context by estimating an
fp32 `[n_head, N, N]` attention scores buffer, and it applies that estimate *even when flash
attention is enabled* — though with FA the buffer is never materialized. So `DG_FLASH_ATTN=1`
plus `DG_MAXTOK=65536` still resolves to 20480. Making that sizing heuristic FA-aware in
`diffusion-gemma-visual-server.cpp` is the fix, and it is not done here.

Two smaller things worth knowing, both of which broke Claude Code until they were fixed:

- Claude Code posts to `/v1/messages?beta=true`. Route matching has to ignore the query string.
- It preflights connectivity with `HEAD /api/hello`, and `BaseHTTPRequestHandler` answers `501`
  to any HEAD it is not explicitly taught, which Claude Code reports as
  "There's an issue with the selected model" — a misleading error that has nothing to do with
  the model name.

## The model often forgets to close its reasoning channel

Worth knowing about, because it looks like an adapter bug and is not.

The model regularly opens `<|channel>thought`, writes its actual answer inside, and stops without
ever emitting the closing `<channel|>`. Across six identical tool-followup requests, four ended
that way — and the withheld text was the final answer, word for word. Taken literally, those
responses have an empty `text` block, which is useless to a client.

So the adapter uses how generation *ended* to settle it. If the model stopped on its own, an
unclosed channel means it simply never closed it, and the content is the answer. If generation was
cut off by the token budget, the reasoning really was mid-flight and stays reasoning.

One consequence for streaming: a trailing unclosed reasoning block is held back until generation
ends, because a delta already sent cannot be retracted. Reasoning that the model *does* close
streams normally as it arrives.

## Set max_tokens generously

This model reasons a *lot*, and the reasoning comes out of the same budget as the answer. Asking
for a 400-word explanation with `max_tokens: 1024` produced 1024 tokens of pure reasoning, got
cut off mid-thought, and returned an empty `text` block with `stop_reason: "max_tokens"`. The
same prompt at `max_tokens: 3072` spent about 1000 tokens thinking and then wrote the full
answer, finishing in 34 s with `stop_reason: "end_turn"`.

So if you get an empty answer, that is the first thing to check. `max_tokens` is rounded up to a
whole number of 256-token canvas blocks, and unused blocks cost nothing — generation stops as
soon as the model emits an end-of-generation token. There is no reason to be stingy.

## Notes on the wiring

A couple of details worth knowing if you go modify this.

The `C` records from the backend carry the *cumulative* answer, not a delta, so the adapter
diffs against what it has already sent. The `F` records are the live canvas mid-denoise — they
rewrite in place rather than append, so they are useless as output and are used only as
keep-alive heartbeats.

Error handling has a wrinkle. Most `ERR` codes are followed by `STATS` and `DONE`, but
`badreq`, `parse` and `emptyprompt` are not — the C++ side `continue`s its read loop without
emitting `DONE`. Reading until `DONE` after one of those will hang, so those three are handled
separately.

`ERR toolong` is recoverable: it means the conversation outgrew the context budget partway
through, and any blocks already committed are still valid. The adapter returns the partial answer
with `stop_reason: "max_tokens"` rather than throwing it away. Raise `--maxtok` if you hit it early.

One correction worth recording: `tools/ui/src/lib/utils/chat-template-thinking-detector.ts` lists
the reasoning tag pair as `<|channel>thought` / `<|channel|>`. The closing tag in this model's
template is actually `<channel|>`. The adapter uses the template's spelling, and
`--think-open`/`--think-close` override it if a future build differs.
