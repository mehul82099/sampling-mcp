#!/usr/bin/env python3
"""sampling-mcp - a zero-dependency MCP server (stdio) that uses the
client's own language model through MCP sampling, carried by the
SEP-2322 InputRequiredResult envelope on the 2026-07-28 stateless core.

Demo tool polish_text: send rough text, and the server asks the client
to run two sampling rounds - a rewrite in the requested tone, then a
tightening pass - before returning the final text with deterministic
post-processing (word count, reading time, hype-word warnings,
truncation detection).

  client -> tools/call(id=1, polish_text{text, tone})
  server -> {resultType: "input_required",
             inputRequests: {"draft": sampling/createMessage...},
             requestState: <signed blob>}        (request ends)
  client -> tools/call(id=2, inputResponses={"draft": CreateMessageResult})
  server -> {resultType: "input_required",
             inputRequests: {"tighten": sampling/createMessage...}, ...}
  client -> tools/call(id=3, inputResponses={"tighten": ...})
  server -> final CallToolResult

The server is fully stateless: round state travels in requestState,
base64url JSON signed with HMAC-SHA256 (per-process random key).
Tampered or foreign state fails with -32602.

Spec rules honored (2026-07-28 sampling spec, SEP-2322, SEP-1577):
- sampling only requested when the client declares the `sampling`
  capability in _meta.io.modelcontextprotocol/clientCapabilities;
  otherwise a plain-text fallback result, never an elicitation-style ask.
- includeContext is omitted entirely (defaults to "none"); the
  deprecated "thisServer"/"allServers" values are never sent.
- maxTokens is always set (required); modelPreferences carry priorities
  only, no model-name hints (the client's choice stays the client's).
- A stopReason of "toolUse" is answered with an error: this server
  offers no sampling tools (SEP-1577 tool loop not implemented - see
  README honest limits).
- Errors: bad arguments are tool execution errors (SEP-1303); malformed
  protocol input gets JSON-RPC errors; tool names follow SEP-986.

Run: python3 sampling_mcp.py
"""
import base64
import hashlib
import hmac
import json
import re
import secrets
import sys

PROTOCOL_VERSION = "2026-07-28"
LEGACY_VERSIONS = {"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"}
SERVER_INFO = {"name": "sampling-mcp", "version": "1.0.0"}

ERR_INVALID_PARAMS = -32602
ERR_METHOD_NOT_FOUND = -32601
ERR_PARSE = -32700

CAP_KEY = "io.modelcontextprotocol/clientCapabilities"
MAX_TEXT = 20_000
DRAFT_MAX_TOKENS = 800
TIGHTEN_MAX_TOKENS = 400
TONES = ("professional", "casual", "concise")

# Inflated words the post-processor flags in model output.
HYPE_WORDS = ("delve", "crucial", "pivotal", "leverage", "streamline",
              "holistic", "seamless", "foster", "empower", "elevate",
              "showcase", "groundbreaking", "renowned", "profound",
              "boasts", "garner")

_STATE_KEY = secrets.token_bytes(32)


def _b64e(raw):
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def mint_state(payload):
    body = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(_STATE_KEY, body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def read_state(blob):
    try:
        body, sig = blob.split(".", 1)
    except ValueError:
        raise ValueError("malformed")
    expect = hmac.new(_STATE_KEY, body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expect):
        raise ValueError("signature mismatch")
    payload = json.loads(_b64d(body))
    if not isinstance(payload, dict):
        raise ValueError("bad payload")
    return payload


def count_words(text):
    return len(re.findall(r"[A-Za-z0-9']+", text))


def reading_time_seconds(words):
    return round(words / 200 * 60)


def hype_hits(text):
    low = text.lower()
    return [w for w in HYPE_WORDS if re.search(rf"\b{re.escape(w)}\b", low)]


def _input_required(key, sampling_params, state):
    return {"resultType": "input_required",
            "inputRequests": {key: {"method": "sampling/createMessage",
                                    "params": sampling_params}},
            "requestState": state}


def _complete(text, structured):
    return {"resultType": "complete",
            "content": [{"type": "text", "text": text}],
            "structuredContent": structured}


def _tool_error(text):
    return {"resultType": "complete",
            "content": [{"type": "text", "text": text}], "isError": True}


def _sampling_params(instruction, tone):
    return {
        "messages": [{"role": "user",
                      "content": {"type": "text", "text": instruction}}],
        "systemPrompt": f"You rewrite text in a {tone} tone. Return only "
                        "the rewritten text, no preamble or commentary.",
        "modelPreferences": {"intelligencePriority": 0.7,
                             "speedPriority": 0.3, "costPriority": 0.3},
        "temperature": 0.2,
        "maxTokens": DRAFT_MAX_TOKENS,
        # includeContext omitted on purpose (defaults to "none"; the
        # deprecated thisServer/allServers values are never sent).
    }


def _client_supports_sampling(meta):
    caps = (meta or {}).get(CAP_KEY, {})
    return "sampling" in caps


def _extract_text(result):
    """Validate a CreateMessageResult, return (text, model, stopReason)
    or raise ValueError with a client-facing message."""
    if not isinstance(result, dict):
        raise ValueError("sampling response must be an object")
    if result.get("role") != "assistant":
        raise ValueError("sampling response role must be 'assistant'")
    content = result.get("content")
    if not isinstance(content, dict) or content.get("type") != "text" \
            or not isinstance(content.get("text"), str):
        raise ValueError("sampling response content must be a text block")
    model = result.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("sampling response must name the model used")
    return content["text"], model, result.get("stopReason")


def polish_text(arguments, meta):
    arguments = arguments or {}
    input_responses = arguments.get("inputResponses") or {}
    blob = arguments.get("requestState")

    state = {}
    if blob is not None:
        if not isinstance(blob, str):
            return ("error", "requestState must be a string.")
        try:
            state = read_state(blob)
        except (ValueError, json.JSONDecodeError):
            return ("error", "requestState is invalid or was not minted by "
                             "this server process. Start over without "
                             "requestState.")

    if not input_responses:
        text = arguments.get("text")
        tone = arguments.get("tone", "professional")
        if not isinstance(text, str) or not 1 <= len(text) <= MAX_TEXT:
            return ("complete", _tool_error(
                f"text is required (1-{MAX_TEXT} characters)."))
        if tone not in TONES:
            return ("complete", _tool_error(
                f"tone must be one of {', '.join(TONES)}."))
        if not _client_supports_sampling(meta):
            return ("complete", _complete(
                "This client does not declare MCP sampling support, so "
                "polish_text cannot borrow its model. Reconnect with a "
                "sampling-capable client (capability: sampling) to use "
                "the interactive flow.",
                {"final": None, "sampling": "unsupported"}))
        instruction = (f"Rewrite the following text in a {tone} tone. "
                       f"Keep the meaning, fix the flow.\n\n{text}")
        return ("complete", _input_required(
            "draft", _sampling_params(instruction, tone),
            mint_state({"round": 1, "text": text, "tone": tone})))

    key = "draft" if "draft" in input_responses else "tighten"
    if key not in input_responses or len(input_responses) != 1:
        return ("error", "inputResponses must answer exactly the key from "
                         "inputRequests.")
    resp = input_responses[key]
    if isinstance(resp, dict) and "error" in resp:
        return ("complete", _tool_error(
            "The client could not complete the sampling request"
            + (f": {resp['error']}" if isinstance(resp["error"], str)
               else ".") + " Call polish_text again to retry."))
    try:
        text_out, model, stop_reason = _extract_text(resp)
    except ValueError as exc:
        return ("error", str(exc))
    if stop_reason == "toolUse":
        # SEP-1577 tool loop: this server offers no sampling tools.
        return ("complete", _tool_error(
            "The model asked to use tools during sampling, but "
            "polish_text offers none (stopReason toolUse). Start over; "
            "consider a lower intelligencePriority."))

    if key == "draft":
        if state.get("round") != 1:
            return ("error", "draft response arrived without round-1 "
                             "requestState; start over.")
        warnings = []
        if stop_reason == "maxTokens":
            warnings.append("draft hit maxTokens and may be truncated")
        hits = hype_hits(text_out)
        if hits:
            warnings.append("hype words in draft: " + ", ".join(hits))
        words = count_words(text_out)
        target = max(1, round(words * 0.7))
        instruction = (f"Tighten this draft to about {target} words "
                       f"without losing meaning. Keep the same tone.\n\n"
                       f"{text_out}")
        params = _sampling_params(instruction, state["tone"])
        params["maxTokens"] = TIGHTEN_MAX_TOKENS
        return ("complete", _input_required(
            "tighten", params,
            mint_state({"round": 2, "draft": text_out,
                        "draft_model": model, "tone": state["tone"],
                        "warnings": warnings})))

    # tighten round answered: finalize.
    if state.get("round") != 2 or "draft" not in state:
        return ("error", "tighten response arrived without round-2 "
                         "requestState; start over.")
    warnings = list(state.get("warnings", []))
    if stop_reason == "maxTokens":
        warnings.append("final text hit maxTokens and may be truncated")
    hits = hype_hits(text_out)
    if hits:
        warnings.append("hype words in final: " + ", ".join(hits))
    draft_words = count_words(state["draft"])
    final_words = count_words(text_out)
    summary = {
        "draft": state["draft"],
        "final": text_out,
        "draft_model": state.get("draft_model"),
        "final_model": model,
        "tone": state.get("tone"),
        "draft_words": draft_words,
        "final_words": final_words,
        "reading_time_seconds": reading_time_seconds(final_words),
        "warnings": warnings,
    }
    note = f" ({'; '.join(warnings)})" if warnings else ""
    return ("complete", _complete(
        f"Polished: {draft_words} -> {final_words} words via "
        f"{model}.{note}\n\n{text_out}", summary))


TOOLS = [{
    "name": "polish_text",
    "description": "Polish rough text using the client's own model via "
                   "MCP sampling (SEP-2322 round-trips): one rewrite in "
                   "the requested tone, one tightening pass, then "
                   "deterministic post-processing (word counts, reading "
                   "time, hype-word warnings). Stateless: round state "
                   "travels in the signed requestState blob. Requires a "
                   "client that declares the sampling capability.",
    "inputSchema": {"type": "object", "properties": {
        "text": {"type": "string", "maxLength": MAX_TEXT},
        "tone": {"type": "string", "enum": list(TONES)},
        "inputResponses": {"type": "object"},
        "requestState": {"type": "string"}},
        "additionalProperties": True},
}]


def capabilities():
    return {"tools": {"listChanged": False}}


def handle_message(msg):
    if not isinstance(msg, dict) or "id" not in msg:
        return None
    rid = msg["id"]
    method = msg.get("method")
    params = msg.get("params") or {}

    if method == "server/discover":
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": SERVER_INFO, "capabilities": capabilities()}}
    if method == "initialize":
        requested = params.get("protocolVersion", "")
        version = requested if requested in LEGACY_VERSIONS \
            or requested == PROTOCOL_VERSION else PROTOCOL_VERSION
        return {"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": version,
            "serverInfo": SERVER_INFO, "capabilities": capabilities()}}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": rid, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": rid, "result": {"tools": TOOLS}}
    if method == "tools/call":
        if params.get("name") != "polish_text":
            return {"jsonrpc": "2.0", "id": rid, "error": {
                "code": ERR_INVALID_PARAMS,
                "message": f"Unknown tool '{params.get('name')}'."}}
        kind, payload = polish_text(params.get("arguments"),
                                    msg.get("_meta"))
        if kind == "error":
            return {"jsonrpc": "2.0", "id": rid, "error": {
                "code": ERR_INVALID_PARAMS, "message": payload}}
        return {"jsonrpc": "2.0", "id": rid, "result": payload}
    return {"jsonrpc": "2.0", "id": rid, "error": {
        "code": ERR_METHOD_NOT_FOUND, "message": f"Unknown method '{method}'."}}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({
                "jsonrpc": "2.0", "id": None,
                "error": {"code": ERR_PARSE, "message": "Parse error"}})
                + "\n")
            sys.stdout.flush()
            continue
        response = handle_message(msg)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
