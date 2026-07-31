import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anthropic_adapter import (
    split_segments, parse_tool_call, anthropic_to_oai, _content_block, _stop_reason,
    DEFAULT_THINK_OPEN as TO, DEFAULT_THINK_CLOSE as TC,
)

Q = '<|"|>'
fails = []
def check(name, got, want):
    if got != want:
        fails.append(f"{name}\n   got:  {got!r}\n   want: {want!r}")

# --- argument grammar ---
check("string arg", parse_tool_call(f"call:get_weather{{location:{Q}Paris{Q}}}"),
      ("get_weather", {"location": "Paris"}))
check("int + bool + null",
      parse_tool_call("call:f{count:5,flag:true,off:false,nothing:null}"),
      ("f", {"count": 5, "flag": True, "off": False, "nothing": None}))
check("float", parse_tool_call("call:f{x:1.5,y:-2.25}"), ("f", {"x": 1.5, "y": -2.25}))
check("no args", parse_tool_call("call:ping{}"), ("ping", {}))
check("array of strings",
      parse_tool_call(f"call:f{{tags:[{Q}a{Q},{Q}b{Q}]}}"), ("f", {"tags": ["a", "b"]}))
check("array of ints", parse_tool_call("call:f{n:[1,2,3]}"), ("f", {"n": [1, 2, 3]}))
check("nested object",
      parse_tool_call(f"call:f{{opts:{{deep:{{k:{Q}v{Q}}},n:2}}}}"),
      ("f", {"opts": {"deep": {"k": "v"}, "n": 2}}))
# the whole reason for a real parser: delimiters inside string values
check("comma in string", parse_tool_call(f"call:f{{q:{Q}a,b{Q},n:1}}"),
      ("f", {"q": "a,b", "n": 1}))
check("colon+braces in string", parse_tool_call(f"call:f{{q:{Q}{{a:b}}{Q}}}"),
      ("f", {"q": "{a:b}"}))
check("newline in string", parse_tool_call(f"call:f{{q:{Q}line1\nline2{Q}}}"),
      ("f", {"q": "line1\nline2"}))
check("empty string", parse_tool_call(f"call:f{{q:{Q}{Q}}}"), ("f", {"q": ""}))
check("empty array", parse_tool_call("call:f{a:[]}"), ("f", {"a": []}))

for bad, why in [("get_weather{}", "no call: prefix"), ("call:f", "no braces"),
                 ("call:{}", "no name"), (f"call:f{{q:{Q}unterminated}}", "bad string")]:
    try:
        parse_tool_call(bad); fails.append(f"expected ValueError for {why}: {bad!r}")
    except ValueError:
        pass

# --- segmentation ---
call = f"<|tool_call>call:get_weather{{location:{Q}Paris{Q}}}<tool_call|>"
check("tool call only", split_segments(call, TO, TC),
      [("tool_use", {"name": "get_weather", "input": {"location": "Paris"}})])

check("thinking + text + tool",
      split_segments(f"{TO}reasoning{TC}Let me check.{call}", TO, TC),
      [("thinking", "reasoning"), ("text", "Let me check."),
       ("tool_use", {"name": "get_weather", "input": {"location": "Paris"}})])

check("two tool calls",
      split_segments(call + call, TO, TC),
      [("tool_use", {"name": "get_weather", "input": {"location": "Paris"}}),
       ("tool_use", {"name": "get_weather", "input": {"location": "Paris"}})])

# incomplete call straddling a block boundary must be withheld, not leaked as text
check("incomplete tool call withheld",
      split_segments("Checking.<|tool_call>call:get_weather{loc", TO, TC),
      [("text", "Checking.")])

# unparseable but complete: surfaced as text rather than dropped
seg = split_segments("<|tool_call>garbage<tool_call|>", TO, TC)
check("unparseable surfaced as text", seg, [("text", "<|tool_call>garbage<tool_call|>")])

check("realistic full output",
      split_segments(f"<|turn>model\n{TO}\nNeed weather.\n{TC}{call}<eos>", TO, TC),
      [("thinking", "\nNeed weather.\n"),
       ("tool_use", {"name": "get_weather", "input": {"location": "Paris"}})])

# --- content blocks ---
b = _content_block("tool_use", {"name": "f", "input": {"a": 1}})
check("tool_use block shape",
      {k: b[k] for k in ("type", "name", "input")},
      {"type": "tool_use", "name": "f", "input": {"a": 1}})
if not b["id"].startswith("toolu_"): fails.append(f"bad tool id {b['id']!r}")
payload = {"name": "f", "input": {}}
check("id stable across calls",
      _content_block("tool_use", payload)["id"], _content_block("tool_use", payload)["id"])

check("stop_reason tool_use", _stop_reason(False, True), "tool_use")
check("truncation beats tool_use", _stop_reason(True, True), "max_tokens")

# --- request conversion: tool_use / tool_result round trip ---
msgs, tools = anthropic_to_oai({
    "messages": [
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
             "input": {"location": "Paris"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"}]},
    ],
    "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
})
check("assistant tool_calls", msgs[1], {
    "role": "assistant", "content": "Checking.",
    "tool_calls": [{"id": "toolu_1", "type": "function",
                    "function": {"name": "get_weather",
                                 "arguments": json.dumps({"location": "Paris"})}}]})
check("tool result becomes role=tool",
      msgs[2], {"role": "tool", "tool_call_id": "toolu_1", "content": "18C"})
check("no empty trailing user message", len(msgs), 3)

# error result
msgs2, _ = anthropic_to_oai({"messages": [{"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "t", "content": "boom", "is_error": True}]}]})
check("is_error marked", msgs2[0]["content"], "Error: boom")

if fails:
    print("FAIL (%d)\n" % len(fails) + "\n".join(fails)); sys.exit(1)
print("all tool-calling tests passed")
