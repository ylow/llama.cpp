import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anthropic_adapter import (
    split_segments, finalize_segments, anthropic_to_oai, _blocks_to_text,
    _stop_reason, _truncated, _usage,
    DEFAULT_THINK_OPEN as TO, DEFAULT_THINK_CLOSE as TC,
)

fails = []
def check(name, got, want):
    if got != want:
        fails.append(f"{name}\n   got:  {got!r}\n   want: {want!r}")

# --- split_segments ---
check("plain text", split_segments("hello world", TO, TC), [("text", "hello world")])

check("thinking then text",
      split_segments(f"{TO}let me think{TC}The answer is 4.", TO, TC),
      [("thinking", "let me think"), ("text", "The answer is 4.")])

check("open thinking (mid-stream)",
      split_segments(f"{TO}still reasoning", TO, TC),
      [("thinking_open", "still reasoning")])

check("text before thinking",
      split_segments(f"prelude{TO}hmm{TC}done", TO, TC),
      [("text", "prelude"), ("thinking", "hmm"), ("text", "done")])

check("strips turn markers",
      split_segments("<|turn>model\nanswer<eos>", TO, TC),
      [("text", "answer")])

check("real template shape",
      split_segments("<|turn>model\n<|channel>thought\nreasoning\n<channel|>Final.<eos>", TO, TC),
      [("thinking", "\nreasoning\n"), ("text", "Final.")])

check("empty", split_segments("", TO, TC), [])

# --- anthropic_to_oai ---
msgs, tools = anthropic_to_oai({
    "system": "You are terse.",
    "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ],
})
check("system + blocks", msgs, [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
])
check("no tools", tools, None)

msgs, tools = anthropic_to_oai({
    "messages": [{"role": "user", "content": "x"}],
    "tools": [{"name": "get_weather", "description": "d", "input_schema": {"type": "object"}}],
})
check("tools converted", tools, [{
    "type": "function",
    "function": {"name": "get_weather", "description": "d", "parameters": {"type": "object"}},
}])

# system as content blocks
msgs, _ = anthropic_to_oai({"system": [{"type": "text", "text": "S"}],
                            "messages": [{"role": "user", "content": "u"}]})
check("system blocks", msgs[0], {"role": "system", "content": "S"})

# tool blocks are no longer flattened into text: they become proper OAI messages
# (covered in test_tools.py). _blocks_to_text handles text-bearing blocks only.
check("tool_result not flattened into text",
      _blocks_to_text([{"type": "tool_result", "tool_use_id": "t1", "content": "72F"}]), "")

check("thinking dropped from input", _blocks_to_text([
    {"type": "thinking", "thinking": "old"}, {"type": "text", "text": "keep"}]), "keep")

# --- truncation + stop_reason ---
check("not truncated (short answer)",
      _truncated({"blocks": 1, "predicted_n": 100}, 2, 256, False), False)
check("truncated (budget exhausted)",
      _truncated({"blocks": 2, "predicted_n": 512}, 2, 256, False), True)
check("truncated (errored)", _truncated(None, 1, 256, True), True)
check("end_turn", _stop_reason(False), "end_turn")
check("max_tokens", _stop_reason(True), "max_tokens")

# --- unclosed reasoning channel ---
# The model often writes its answer inside <|channel>thought and never closes it.
open_think = split_segments(f"{TO}The answer is 42.", TO, TC)
check("open reasoning marked", open_think, [("thinking_open", "The answer is 42.")])
check("finished run -> that was the answer",
      finalize_segments(open_think, truncated=False), [("text", "The answer is 42.")])
check("truncated run -> genuinely cut-off reasoning",
      finalize_segments(open_think, truncated=True), [("thinking", "The answer is 42.")])
check("closed reasoning unaffected",
      finalize_segments(split_segments(f"{TO}r{TC}a", TO, TC), truncated=False),
      [("thinking", "r"), ("text", "a")])
check("usage", _usage({"prompt_n": 12, "predicted_n": 34}),
      {"input_tokens": 12, "output_tokens": 34})

if fails:
    print("FAIL\n" + "\n".join(fails)); sys.exit(1)
print("all pure-logic tests passed")
