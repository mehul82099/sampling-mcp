# sampling-mcp

A zero-dependency, single-file Python MCP server (stdio) that borrows
the client's own language model through **MCP sampling**, carried by the
[SEP-2322](https://modelcontextprotocol.io/seps/2322-multi-round-trip-requests)
`InputRequiredResult` envelope on the 2026-07-28 stateless core.

The demo tool, `polish_text`, runs two sampling rounds against the
client's model - a rewrite in your chosen tone, then a tightening pass -
and finishes with deterministic post-processing in plain Python: word
counts, reading time, hype-word warnings ("delve", "leverage",
"seamless"...), and truncation detection when the model hits
`maxTokens`. Fully stateless: round state travels in an HMAC-signed
`requestState` blob, so any round can land on any process holding the
key.

## The flow

```
client -> tools/call(id=1, polish_text{text, tone})
server -> {resultType: "input_required",
           inputRequests: {"draft": sampling/createMessage...},
           requestState: "..."}                  # request ends
client -> tools/call(id=2, inputResponses={"draft": CreateMessageResult})
server -> {resultType: "input_required",
           inputRequests: {"tighten": sampling/createMessage...}, ...}
client -> tools/call(id=3, inputResponses={"tighten": CreateMessageResult})
server -> {resultType: "complete",
           structuredContent: {draft, final, models, word counts, warnings}}
```

## What it demonstrates, mapped to the specs

- **Sampling through the round-trip envelope (2026-07-28 sampling spec +
  SEP-2322):** `sampling/createMessage` travels inside
  `InputRequiredResult.inputRequests`; the client's
  `CreateMessageResult` comes back in `inputResponses` under the same
  key, on a brand-new request. No SSE stream, no server-side session.
- **Capability gating:** the server reads
  `_meta.io.modelcontextprotocol/clientCapabilities.sampling` on every
  call and never asks a client that did not declare sampling - it
  answers with a plain-text fallback instead.
- **Deprecation-clean requests:** `includeContext` is omitted entirely
  (it defaults to `"none"`; the `"thisServer"`/`"allServers"` values are
  deprecated under SEP-2596 and are never sent). `modelPreferences`
  carries priorities only, no model-name hints - the client's model
  choice stays the client's.
- **`maxTokens` always set** (required by the spec), and a `stopReason`
  of `"maxTokens"` surfaces as an explicit truncation warning instead of
  silently shipping a cut-off draft.
- **SEP-1577 boundary stated:** if the model answers with
  `stopReason: "toolUse"`, the server fails cleanly - it offers no
  sampling tools, so the tool-use loop is out of scope (see limits).
- **Stateless correlation:** `requestState` is base64url JSON signed
  with HMAC-SHA256 (per-process random key). Tampered, forged, or
  cross-process state fails with `-32602`; the client treats the blob
  as opaque per spec.
- **Result validation:** role, text-only content, and the `model` field
  are checked before any post-processing; malformed sampling results get
  protocol errors, not crashes. A client-side sampling failure (error
  object in the response slot) becomes a clean tool error.
- **Conventions:** bad arguments are tool-level `isError` results
  (SEP-1303); unknown tools/methods get `-32602`/`-32601`; tool names
  follow SEP-986; legacy `initialize` answered alongside stateless
  `server/discover` (SEP-2575).

## Run it

```sh
python3 sampling_mcp.py
```

```json
{
  "mcpServers": {
    "sampling": {
      "command": "python3",
      "args": ["/path/to/sampling_mcp.py"]
    }
  }
}
```

The interactive flow needs a client that declares the `sampling`
capability; without it the server says so in plain text instead of
asking.

## Tests

17 stdlib `unittest` tests drive the real server over stdio: both
handshakes, envelope shape (messages, systemPrompt, maxTokens,
modelPreferences priorities, no includeContext, no hints), tone
propagation, the full two-round happy path, hype-word and truncation
detection, toolUse rejection, client sampling failures, malformed
result validation, wrong/duplicate response keys, requestState tamper,
argument validation, reading time, capability gating (missing /
elicitation-only capabilities), list-method scope, and SEP-986 names.
CI runs them on Ubuntu 24.04 and Windows, Python 3.9 and 3.13.

```sh
python3 -m unittest -v
```

Honest limits: the SEP-1577 sampling-with-tools loop is not implemented
(the server offers no tools to the sampled model and rejects `toolUse`
stop reasons); `requestState` is signed, not encrypted - never put
secrets in tool arguments; the model's output is trusted only after
shape validation, but this is still a demo, not a review gate for
production text. Landscape note: existing Python sampling material
(e.g. bwlab/mcp-sampling) demonstrates sampling *security risks*; this
repo is a protocol-shape reference for the SEP-2322 envelope flow
(GitHub search, 2026-11-07).

## Sources

- [MCP specification 2026-07-28 - Sampling](https://modelcontextprotocol.io/specification/2026-07-28/client/sampling) (capability shape, createMessage params, deprecation note)
- [SEP-2322: Multi Round-Trip Requests](https://modelcontextprotocol.io/seps/2322-multi-round-trip-requests) (InputRequiredResult/inputResponses/requestState)
- [SEP-1577: Sampling With Tools](https://modelcontextprotocol.io/seps/1577-sampling-with-tools), [SEP-2596: Feature Lifecycle](https://modelcontextprotocol.io/seps/2596-specification-feature-lifecycle-and-deprecation-policy), [SEP-2575: Stateless MCP](https://modelcontextprotocol.io/seps/2575-stateless-mcp), [SEP-1303](https://modelcontextprotocol.io/seps/1303-input-validation-errors-as-tool-execution-errors), [SEP-986](https://modelcontextprotocol.io/seps/986-specify-format-for-tool-names)
- All fetched 2026-11-07.

## License

MIT
