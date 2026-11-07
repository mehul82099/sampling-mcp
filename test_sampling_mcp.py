import base64
import json
import os
import re
import subprocess
import sys
import unittest

SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "sampling_mcp.py")
CAPS = {"io.modelcontextprotocol/clientCapabilities": {"sampling": {}}}


class Server:
    def __init__(self):
        self.proc = subprocess.Popen(
            [sys.executable, SERVER], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.next_id = 0

    def request(self, method, params=None, meta=None):
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method,
               "params": params or {}}
        if meta is not None:
            msg["_meta"] = meta
        self.proc.stdin.write(json.dumps(msg).encode() + b"\n")
        self.proc.stdin.flush()
        reply = json.loads(self.proc.stdout.readline().decode())
        assert reply.get("id") == self.next_id, f"bad id: {reply}"
        return reply

    def result(self, method, params=None, meta=None):
        msg = self.request(method, params, meta)
        assert "result" in msg, f"error from {method}: {msg}"
        return msg["result"]

    def call(self, arguments=None, meta=CAPS):
        return self.result("tools/call",
                           {"name": "polish_text",
                            "arguments": arguments or {}}, meta)

    def close(self):
        self.proc.kill()
        self.proc.wait(timeout=5)


def fake_result(text, model="test-model-1", stop="endTurn"):
    return {"role": "assistant",
            "content": {"type": "text", "text": text},
            "model": model, "stopReason": stop}


def b64d(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


class HandshakeTests(unittest.TestCase):
    def test_discover_and_legacy(self):
        s = Server()
        try:
            d = s.result("server/discover")
            self.assertEqual(d["protocolVersion"], "2026-07-28")
            self.assertEqual(d["serverInfo"]["name"], "sampling-mcp")
            r = s.result("initialize", {"protocolVersion": "2025-03-26"})
            self.assertEqual(r["protocolVersion"], "2025-03-26")
        finally:
            s.close()

    def test_errors(self):
        s = Server()
        try:
            self.assertEqual(s.request("nope")["error"]["code"], -32601)
            self.assertEqual(s.request("tools/call", {"name": "nope"})
                             ["error"]["code"], -32602)
            s.proc.stdin.write(b"{oops\n")
            s.proc.stdin.flush()
            self.assertEqual(json.loads(s.proc.stdout.readline())
                             ["error"]["code"], -32700)
        finally:
            s.close()


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.s = Server()

    def tearDown(self):
        self.s.close()

    def test_first_call_envelope(self):
        r = self.s.call({"text": "ship the thing friday ok"})
        self.assertEqual(r["resultType"], "input_required")
        self.assertEqual(list(r["inputRequests"]), ["draft"])
        req = r["inputRequests"]["draft"]
        self.assertEqual(req["method"], "sampling/createMessage")
        p = req["params"]
        self.assertEqual(p["messages"][0]["role"], "user")
        self.assertIn("ship the thing friday ok",
                      p["messages"][0]["content"]["text"])
        self.assertIsInstance(p["maxTokens"], int)
        self.assertNotIn("includeContext", p)  # deprecated values never sent
        self.assertIn("intelligencePriority", p["modelPreferences"])
        self.assertNotIn("hints", p["modelPreferences"])
        self.assertIn("professional", p["systemPrompt"])
        self.assertTrue(r["requestState"])

    def test_tone_reaches_system_prompt(self):
        r = self.s.call({"text": "hey team", "tone": "casual"})
        self.assertIn("casual", r["inputRequests"]["draft"]["params"]
                      ["systemPrompt"])

    def test_full_two_round_flow(self):
        r1 = self.s.call({"text": "we should probably maybe ship friday"})
        r2 = self.s.call({
            "inputResponses": {"draft": fake_result(
                "We will ship on Friday.", model="model-a")},
            "requestState": r1["requestState"]})
        self.assertEqual(r2["resultType"], "input_required")
        self.assertEqual(list(r2["inputRequests"]), ["tighten"])
        instr = r2["inputRequests"]["tighten"]["params"]["messages"][0][
            "content"]["text"]
        self.assertIn("We will ship on Friday.", instr)
        self.assertEqual(r2["inputRequests"]["tighten"]["params"]
                         ["maxTokens"], 400)
        r3 = self.s.call({
            "inputResponses": {"tighten": fake_result(
                "We ship Friday.", model="model-b")},
            "requestState": r2["requestState"]})
        self.assertEqual(r3["resultType"], "complete")
        sc = r3["structuredContent"]
        self.assertEqual(sc["draft"], "We will ship on Friday.")
        self.assertEqual(sc["final"], "We ship Friday.")
        self.assertEqual(sc["draft_model"], "model-a")
        self.assertEqual(sc["final_model"], "model-b")
        self.assertEqual(sc["draft_words"], 5)
        self.assertEqual(sc["final_words"], 3)
        self.assertEqual(sc["warnings"], [])
        self.assertIn("We ship Friday.", r3["content"][0]["text"])

    def test_hype_word_detection(self):
        r1 = self.s.call({"text": "our platform helps teams"})
        r2 = self.s.call({
            "inputResponses": {"draft": fake_result(
                "We leverage a seamless platform to empower teams.")},
            "requestState": r1["requestState"]})
        body = r2["requestState"].split(".", 1)[0]
        payload = json.loads(b64d(body))
        self.assertEqual(payload["round"], 2)
        self.assertTrue(any("leverage" in w for w in payload["warnings"]))
        r3 = self.s.call({
            "inputResponses": {"tighten": fake_result(
                "Our platform helps teams delve deeper.")},
            "requestState": r2["requestState"]})
        self.assertTrue(any("delve" in w
                            for w in r3["structuredContent"]["warnings"]))

    def test_maxtokens_truncation_warning(self):
        r1 = self.s.call({"text": "long story here"})
        r2 = self.s.call({
            "inputResponses": {"draft": fake_result(
                "A " * 50 + "story.", stop="maxTokens")},
            "requestState": r1["requestState"]})
        r3 = self.s.call({
            "inputResponses": {"tighten": fake_result("A story.")},
            "requestState": r2["requestState"]})
        self.assertTrue(any("truncated" in w
                            for w in r3["structuredContent"]["warnings"]))

    def test_tooluse_stopreason_rejected(self):
        r1 = self.s.call({"text": "hello world"})
        r2 = self.s.call({
            "inputResponses": {"draft": fake_result("", stop="toolUse")},
            "requestState": r1["requestState"]})
        self.assertTrue(r2["isError"])
        self.assertIn("toolUse", r2["content"][0]["text"])

    def test_sampling_failure_object(self):
        r1 = self.s.call({"text": "hello world"})
        r2 = self.s.call({
            "inputResponses": {"draft": {"error": "user declined"}},
            "requestState": r1["requestState"]})
        self.assertTrue(r2["isError"])
        self.assertIn("user declined", r2["content"][0]["text"])

    def test_malformed_sampling_results(self):
        r1 = self.s.call({"text": "hello world"})
        bad_results = [
            {"role": "user", "content": {"type": "text", "text": "x"},
             "model": "m"},
            {"role": "assistant",
             "content": {"type": "image", "data": "..."}, "model": "m"},
            {"role": "assistant",
             "content": {"type": "text", "text": "x"}},
            "just a string",
        ]
        for bad in bad_results:
            msg = self.s.request("tools/call", {
                "name": "polish_text",
                "arguments": {"inputResponses": {"draft": bad},
                              "requestState": r1["requestState"]}}, CAPS)
            self.assertIn("error", msg, f"accepted {bad!r}")

    def test_wrong_key_and_double_key(self):
        r1 = self.s.call({"text": "hello world"})
        for responses in ({"nope": fake_result("x")},
                          {"draft": fake_result("x"),
                           "tighten": fake_result("y")}):
            msg = self.s.request("tools/call", {
                "name": "polish_text",
                "arguments": {"inputResponses": responses,
                              "requestState": r1["requestState"]}}, CAPS)
            self.assertIn("error", msg)

    def test_tampered_state(self):
        r1 = self.s.call({"text": "hello world"})
        body = r1["requestState"].split(".", 1)[0]
        forged = body + "." + "0" * 64
        msg = self.s.request("tools/call", {
            "name": "polish_text",
            "arguments": {"inputResponses": {"draft": fake_result("x")},
                          "requestState": forged}}, CAPS)
        self.assertEqual(msg["error"]["code"], -32602)

    def test_argument_validation(self):
        for args in ({}, {"text": ""}, {"text": "x" * 20001},
                     {"text": "ok", "tone": "fancy"},
                     {"text": 42}):
            r = self.s.call(args)
            self.assertTrue(r.get("isError"), f"accepted {args!r}")

    def test_reading_time(self):
        r1 = self.s.call({"text": "count me please"})
        r2 = self.s.call({
            "inputResponses": {"draft": fake_result("one two three four")},
            "requestState": r1["requestState"]})
        r3 = self.s.call({
            "inputResponses": {"tighten": fake_result("one two")},
            "requestState": r2["requestState"]})
        self.assertEqual(r3["structuredContent"]["reading_time_seconds"], 1)


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        self.s = Server()

    def tearDown(self):
        self.s.close()

    def test_no_sampling_capability_no_ask(self):
        for meta in (None, {}, {"io.modelcontextprotocol/"
                                "clientCapabilities": {}},
                     {"io.modelcontextprotocol/clientCapabilities":
                      {"elicitation": {"form": {}}}}):
            r = self.s.call({"text": "hello"}, meta=meta)
            self.assertEqual(r["resultType"], "complete")
            self.assertNotIn("inputRequests", r)
            self.assertEqual(r["structuredContent"]["sampling"],
                             "unsupported")

    def test_lists_never_input_required(self):
        r = self.s.result("tools/list")
        self.assertNotIn("inputRequests", r)
        self.assertNotEqual(r.get("resultType"), "input_required")

    def test_tool_name_sep986(self):
        for t in self.s.result("tools/list")["tools"]:
            self.assertRegex(t["name"], re.compile(r"^[A-Za-z0-9_./-]{1,64}$"))


if __name__ == "__main__":
    unittest.main()
