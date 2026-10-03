"""
routes/realtime.py — Speech-to-speech over WebSocket (/v1/realtime)

A cascaded voice agent that speaks a subset of the OpenAI Realtime protocol,
so OpenAI SDKs, pipecat or LiveKit clients connect unchanged:

    mic PCM ─▶ Silero VAD ─▶ ASR ─▶ LLM (streamed, tools) ─▶ phrase chunker ─▶ TTS (streamed) ─▶ PCM

Latency comes from overlapping the stages rather than from any one of them:

  * Speculative turns. When the user has been silent for `speculative_ms`
    (200 ms), ASR and the LLM already start on the utterance, but nothing is
    sent until the silence reaches `silence_duration_ms` (500 ms) and the turn
    commits. If the user speaks again first, the turn is thrown away. ASR and
    most of the LLM's time to first token hide inside the endpointing wait.
  * The first phrase goes to the TTS at the first comma (or ~8 words), then
    sentence by sentence; pocket-tts streams PCM frames as it decodes them, so
    audio starts 25-60 ms after a phrase is ready.
  * The system prompt and tools are prefilled once per session.update so the
    llama-server prompt cache is warm before the first turn.
  * Models are held (not evictable) for the life of the connection.

Tools run on the client, as in the OpenAI protocol: the response ends with a
`function_call` item, the client sends `conversation.item.create`
(`function_call_output`) then `response.create`.

Barge-in: speech that lasts `barge_in_ms` while the assistant is talking
cancels the response; the assistant's message is cut in the history to what
was (estimated to have been) played. Clients must cancel echo (browsers:
getUserMedia `echoCancellation: true`), or the assistant interrupts itself.

Event names follow the GA Realtime API; `?events=beta` renames the handful
that changed (response.audio.delta, conversation.item.created, ...).
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import re
import secrets
import time
import unicodedata
import uuid
import wave
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from ..dependencies import get_pm, get_registry
from ..model_registry import ModelRegistry
from ..realtime_vad import FRAME_MS, SileroVad
from ..tts_lang import detect_language
from .openai import _resolve_instance

router = APIRouter()
logger = logging.getLogger("zallama.realtime")

OUT_RATE = 24000        # output PCM rate (pocket-tts and kokoro both speak 24 kHz)

# Prepended to the client's instructions: every reply is read aloud.
STYLE_PROMPT = (
    # Kept short on purpose: a longer speech-style prompt (no lists, natural
    # speech, "call the tool right away"...) made gemma-4-E2B narrate actions
    # instead of calling the tools (7/30 tool calls vs 30/30 with this one).
    "You are a voice assistant; your replies are spoken aloud. Answer briefly in "
    "the user's language, without Markdown or emoji. Use the tools to act or to get facts."
)

_BETA_NAMES = {
    "conversation.item.added": "conversation.item.created",
    "conversation.item.done": None,
    "response.output_audio.delta": "response.audio.delta",
    "response.output_audio.done": "response.audio.done",
    "response.output_audio_transcript.delta": "response.audio_transcript.delta",
    "response.output_audio_transcript.done": "response.audio_transcript.done",
    "response.output_text.delta": "response.text.delta",
    "response.output_text.done": "response.text.done",
}


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def _wav_bytes(pcm: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


def _wav_to_pcm(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1).astype("<i2")
    if rate != OUT_RATE and len(pcm):
        n = int(len(pcm) * OUT_RATE / rate)
        pcm = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm).astype("<i2")
    return pcm


# ---------------------------------------------------------------------------
# Text: what the LLM streams -> phrases the TTS can say
# ---------------------------------------------------------------------------
_HARD = re.compile(r"[.!?…:;]+[\"'»”)\]]*(?=\s)|\n+")
_SOFT = re.compile(r",(?=\s)")
_MD = re.compile(r"[*#`~|]+|^\s*>\s*|^\s*[-•]\s+", re.MULTILINE)


def _speakable(text: str) -> str:
    """Drop what a TTS would read as noise: emoji and Markdown markers."""
    text = _MD.sub(" ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("So", "Cs", "Co", "Cn"))
    return re.sub(r"\s+", " ", text).strip()


def _next_phrase(buf: str, first: bool) -> tuple[str, str] | None:
    """Cut the next phrase off `buf`, or None to wait for more text.

    The first phrase is cut early (first comma after 3 words, or 8 words) so
    audio starts quickly; later ones prefer whole sentences, which the TTS
    phrases better, since the TTS runs far ahead of real time anyway.
    """
    min_hard, min_soft, max_words = (1, 3, 8) if first else (3, 10, 25)
    for m in _HARD.finditer(buf):
        if len(buf[:m.end()].split()) >= min_hard:
            return buf[:m.end()], buf[m.end():]
    for m in _SOFT.finditer(buf):
        if len(buf[:m.end()].split()) >= min_soft:
            return buf[:m.end()], buf[m.end():]
    words = buf.split()
    if len(words) > max_words and buf[-1:].isspace():
        cut = buf.rstrip().rfind(" ")
        if cut > 0:
            return buf[:cut], buf[cut:]
    return None


def _lenient_args(body: str) -> str:
    """Arguments of a tool call written as text, as a JSON string. Gemma
    writes `{city:<|"|>Lyon<|"|>,n:5}` (bare keys, special string quotes);
    without the special tokens that is `{city:"Lyon"}` or `{city:Lyon}`."""
    t = body.replace('<|"|>', '"')
    attempts = [t, re.sub(r'([{,]\s*)([A-Za-z_][\w-]*)\s*:', r'\1"\2":', t)]

    def quote_bare(m: re.Match) -> str:
        v = m.group(1)
        return ": " + (v if v in ("true", "false", "null") else json.dumps(v))
    attempts.append(re.sub(r':\s*([^"\d\[{\s-][^,}]*?)\s*(?=[,}])', quote_bare, attempts[1]))
    for a in attempts:
        try:
            return json.dumps(json.loads(a), ensure_ascii=False)
        except ValueError:
            continue
    return t


class TextToolCalls:
    """Pull tool calls the LLM wrote as text out of its content stream.

    When llama.cpp's parser misses a call (small models, a broken special
    token), the content carries the model's raw syntax: Gemma's
    `call:name{...}` (inside `<|tool_call>...<tool_call|>`) or Hermes/Qwen's
    `<tool_call>{"name":..., "arguments":...}</tool_call>`. Spoken, that is
    noise read before the tool even runs. `feed()` holds back anything that
    may be the start of such a call (even split across chunks), turns a
    complete one naming a declared tool into a call, and gives back the rest
    as text. A `call:` that names no declared tool stays text. After a call,
    the model's further prose is usually an invented tool result, so it is
    dropped, not spoken.
    """
    _CALL = "call:"
    _OPEN = ("<|tool_call>", "<tool_call>")
    _DROP = ("<tool_call|>", "</tool_call>", '<|"|>')
    _NAME = re.compile(r"[A-Za-z_][\w.-]*")

    def __init__(self, names):
        self.names = set(names)
        self.buf = ""
        self.seen = False

    def feed(self, text: str) -> tuple[str, list[dict]]:
        self.buf += text
        return self._scan(final=False)

    def finish(self) -> tuple[str, list[dict]]:
        return self._scan(final=True)

    def _markers(self):
        return (self._CALL, *self._OPEN, *self._DROP)

    def _find(self, buf: str) -> tuple[int | None, str | None]:
        best, which = None, None
        for m in self._markers():
            start = 0
            while (i := buf.find(m, start)) != -1:
                # "call:" only as a word: not "recall:"
                if m == self._CALL and i > 0 and (buf[i - 1].isalnum() or buf[i - 1] == "_"):
                    start = i + 1
                    continue
                if best is None or i < best:
                    best, which = i, m
                break
        return best, which

    def _tail(self, buf: str) -> int:
        """Length of the end of `buf` that may be the start of a marker."""
        keep = 0
        for m in self._markers():
            for k in range(min(len(m) - 1, len(buf)), 0, -1):
                if buf.endswith(m[:k]):
                    keep = max(keep, k)
                    break
        return keep

    def _scan(self, final: bool) -> tuple[str, list[dict]]:
        out: list[str] = []
        calls: list[dict] = []

        def say(t: str):
            if t and not self.seen:
                out.append(t)
        while self.buf:
            i, marker = self._find(self.buf)
            if i is None:
                keep = 0 if final else self._tail(self.buf)
                say(self.buf[:len(self.buf) - keep])
                self.buf = self.buf[len(self.buf) - keep:]
                break
            say(self.buf[:i])
            self.buf = self.buf[i:]
            if marker in self._DROP:
                self.buf = self.buf[len(marker):]
                continue
            res = self._parse(marker, final)
            if res is None:             # incomplete: wait for more text
                if final:
                    logger.info("realtime: dropped an unfinished text tool call: %r", self.buf[:200])
                    self.buf = ""
                break
            used, call, as_text = res
            if call:
                calls.append(call)
                self.seen = True
            elif as_text:
                say(self.buf[:used])
            self.buf = self.buf[used:]
        return "".join(out), calls

    def _parse(self, marker: str, final: bool):
        """(chars used, call or None, speak the used text?) or None if the
        buffer ends before the call does."""
        buf, pos = self.buf, len(marker)
        if marker in self._OPEN:
            rest = buf[pos:].lstrip()
            pos = len(buf) - len(rest)
            if not rest or (len(rest) < len(self._CALL) and self._CALL.startswith(rest)):
                return None if not final else (len(buf), None, False)
            if rest.startswith(self._CALL):
                pos += len(self._CALL)
            elif rest.startswith("{"):          # Hermes JSON
                end = self._match_brace(buf, pos)
                if end is None:
                    return None
                try:
                    obj = json.loads(buf[pos:end])
                except ValueError:
                    obj = {}
                name = obj.get("name") if isinstance(obj, dict) else None
                if name in self.names:
                    args = obj.get("arguments", {})
                    args = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                    return end, {"id": "", "name": name, "arguments": args}, False
                return end, None, False
            else:
                return pos, None, False          # stray special token: drop it
        m = self._NAME.match(buf, pos)
        if not m or m.end() == len(buf):
            return None if not final else (len(buf), None, marker == self._CALL)
        name = m.group(0)
        brace = m.end()
        while brace < len(buf) and buf[brace] == " ":
            brace += 1
        if brace == len(buf):
            return None if not final else (len(buf), None, marker == self._CALL)
        if name not in self.names or buf[brace] != "{":
            return len(self._CALL) if marker == self._CALL else pos, None, marker == self._CALL
        end = self._match_brace(buf, brace)
        if end is None:
            return None
        return end, {"id": "", "name": name, "arguments": _lenient_args(buf[brace:end])}, False

    @staticmethod
    def _match_brace(buf: str, start: int) -> int | None:
        """Index after the brace closing the one at `start`, or None."""
        depth, i, in_str = 0, start, False
        while i < len(buf):
            if buf.startswith('<|"|>', i):
                in_str = not in_str
                i += 5
                continue
            ch = buf[i]
            if in_str:
                if ch == "\\":
                    i += 2
                    continue
                if ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return None


def _chat_tools(tools: list) -> list:
    """Realtime tools are flat ({type, name, description, parameters}); chat
    completions nest them under `function`. Accept either."""
    out = []
    for t in tools or []:
        if not isinstance(t, dict) or t.get("type", "function") != "function":
            continue
        if "function" in t:
            out.append({"type": "function", "function": t["function"]})
        elif t.get("name"):
            fn = {"name": t["name"], "description": t.get("description", ""),
                  "parameters": t.get("parameters") or {"type": "object", "properties": {}}}
            out.append({"type": "function", "function": fn})
    return out


# ---------------------------------------------------------------------------
# One response: (ASR) -> LLM -> TTS, sent only once its gate opens
# ---------------------------------------------------------------------------
class Turn:
    def __init__(self, session: "RealtimeSession", audio: np.ndarray | None, gated: bool,
                 speech_end: float | None = None):
        self.s = session
        self.audio = audio
        self.gate = asyncio.Event()
        if not gated:
            self.gate.set()
        self.out: asyncio.Queue = asyncio.Queue()
        self.transcript: asyncio.Future = asyncio.get_running_loop().create_future()
        self.user_item_id = _id("item")
        self.response_id = _id("resp")
        self.msg_item_id: str | None = None
        self.text = ""                      # everything the LLM said
        self.spoken: list[tuple[str, float, float]] = []   # (phrase, start_ms, end_ms) of audio
        self.audio_ms = 0.0                 # audio generated so far
        self.tool_calls: list[dict] = []    # chat-format tool calls
        self.tools_sent = False
        self.sent_ms = 0.0                  # audio sent to the client
        self.first_sent: float | None = None
        self.speech_end = speech_end or time.monotonic()
        self.lat: dict[str, float] = {}
        self.finished = False               # history updated
        self.aborted = False
        self.client_played: float | None = None   # from conversation.item.truncate
        self.task = asyncio.create_task(self._run())
        self.sender = asyncio.create_task(self._send_loop())

    # -- control -----------------------------------------------------------
    def cancel(self, keep_transcript: bool = False) -> None:
        """Stop the turn. With `keep_transcript`, a running ASR may finish (the
        user's words still belong in the history) but nothing else runs."""
        self.aborted = True
        if keep_transcript and self.audio is not None and not self.transcript.done():
            if not self.sender.done():
                self.sender.cancel()
            return
        for t in (self.task, self.sender):
            if not t.done():
                t.cancel()
        if not self.transcript.done():
            self.transcript.set_result("")

    def played_ms(self) -> float:
        if self.client_played is not None:
            return self.client_played
        if self.first_sent is None:
            return 0.0
        return min(self.sent_ms, (time.monotonic() - self.first_sent) * 1000.0)

    def finished_playing(self) -> bool:
        return self.first_sent is None or self.played_ms() >= self.sent_ms - 1

    def heard_text(self, played_ms: float | None) -> str:
        """What the user heard: whole phrases played, plus the played share of
        the phrase cut off."""
        if played_ms is None:
            return self.text
        parts = []
        for phrase, start, end in self.spoken:
            if start >= played_ms:
                break
            if end <= played_ms:
                parts.append(phrase)
            else:
                words = phrase.split()
                keep = int(len(words) * (played_ms - start) / max(end - start, 1.0))
                if keep:
                    parts.append(" ".join(words[:keep]) + "…")
                break
        return " ".join(p.strip() for p in parts).strip()

    def push(self, event: dict) -> None:
        self.out.put_nowait(event)

    # -- pipeline ----------------------------------------------------------
    async def _run(self) -> None:
        s = self.s
        try:
            messages = s.messages()
            if self.audio is not None:
                t = time.monotonic()
                text = await s.transcribe(self.audio)
                self.lat["asr_ms"] = (time.monotonic() - t) * 1000
                self.transcript.set_result(text)
                if not text or self.aborted:
                    return
                messages = messages + [{"role": "user", "content": text}]
            await self._respond(messages)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("realtime turn failed")
            if not self.transcript.done():
                self.transcript.set_result("")
            self.push({"type": "error", "error": {"type": "server_error", "message": str(e)}})
            self.push(self._done_event("failed"))
        finally:
            self.out.put_nowait(None)

    async def _respond(self, messages: list) -> None:
        s = self.s
        self.push({"type": "response.created", "response": self._response_obj("in_progress")})
        audio_out = "audio" in s.conf["modalities"]
        phrases: asyncio.Queue = asyncio.Queue()
        tts = asyncio.create_task(self._tts_loop(phrases)) if audio_out else None
        try:
            pending, first = "", True
            calls: dict[int, dict] = {}
            text_calls = TextToolCalls(t["function"]["name"] for t in _chat_tools(s.conf["tools"]))

            def say(content: str) -> None:
                nonlocal pending, first
                self._open_message()
                self.text += content
                if not audio_out:
                    self.push({"type": "response.output_text.delta", "response_id": self.response_id,
                               "item_id": self.msg_item_id, "output_index": 0, "content_index": 0,
                               "delta": content})
                    return
                pending += content
                while (cut := _next_phrase(pending, first)) is not None:
                    phrase, pending = cut
                    if _speakable(phrase):
                        phrases.put_nowait(phrase)
                        first = False

            def add_text_calls(found: list[dict]) -> None:
                for c in found:
                    logger.info("realtime: tool call written as text, run as a call: %s%s",
                                c["name"], c["arguments"][:200])
                    calls[10_000 + len(calls)] = c   # after the parsed ones

            t0 = time.monotonic()
            async for delta in s.chat_stream(messages):
                if "llm_ttft_ms" not in self.lat:
                    self.lat["llm_ttft_ms"] = (time.monotonic() - t0) * 1000
                for tc in delta.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    c["id"] = tc.get("id") or c["id"]
                    fn = tc.get("function") or {}
                    c["name"] += fn.get("name") or ""
                    c["arguments"] += fn.get("arguments") or ""
                content = delta.get("content")
                if not content:
                    continue
                text, found = text_calls.feed(content)
                add_text_calls(found)
                if text:
                    say(text)
            text, found = text_calls.finish()
            add_text_calls(found)
            if text:
                say(text)
            if audio_out and _speakable(pending):
                phrases.put_nowait(pending)
            phrases.put_nowait(None)
            if tts:
                await tts
            # The whole reply says more about its language than its first phrase.
            if (lang := detect_language(self.text)):
                s.language = lang
            self._close_message(audio_out)
            self._emit_tool_calls(calls)
            self.push(self._done_event("completed"))
        finally:
            if tts and not tts.done():
                tts.cancel()

    async def _tts_loop(self, phrases: asyncio.Queue) -> None:
        s = self.s
        lang = None     # one TTS language per reply: no model switch mid-sentence
        while (phrase := await phrases.get()) is not None:
            if lang is None:
                lang = s.tts_language(phrase)
            text = _speakable(phrase)
            start = self.audio_ms
            self.push({"type": "response.output_audio_transcript.delta", "response_id": self.response_id,
                       "item_id": self.msg_item_id, "output_index": 0, "content_index": 0,
                       "delta": phrase})
            t = time.monotonic()
            async for pcm in s.synthesize(text, lang):
                if "tts_first_ms" not in self.lat:
                    self.lat["tts_first_ms"] = (time.monotonic() - t) * 1000
                self.audio_ms += len(pcm) / 2 * 1000 / OUT_RATE
                self.push({"type": "response.output_audio.delta", "response_id": self.response_id,
                           "item_id": self.msg_item_id, "output_index": 0, "content_index": 0,
                           "delta": base64.b64encode(pcm).decode()})
            self.spoken.append((phrase, start, self.audio_ms))

    def _open_message(self) -> None:
        if self.msg_item_id:
            return
        self.msg_item_id = _id("item")
        audio = "audio" in self.s.conf["modalities"]
        self.push({"type": "response.output_item.added", "response_id": self.response_id, "output_index": 0,
                   "item": {"id": self.msg_item_id, "object": "realtime.item", "type": "message",
                            "role": "assistant", "status": "in_progress", "content": []}})
        self.push({"type": "response.content_part.added", "response_id": self.response_id,
                   "item_id": self.msg_item_id, "output_index": 0, "content_index": 0,
                   "part": {"type": "audio", "transcript": ""} if audio else {"type": "text", "text": ""}})

    def _message_item(self) -> dict:
        audio = "audio" in self.s.conf["modalities"]
        part = ({"type": "output_audio", "transcript": self.text} if audio
                else {"type": "output_text", "text": self.text})
        return {"id": self.msg_item_id, "object": "realtime.item", "type": "message",
                "role": "assistant", "status": "completed", "content": [part]}

    def _close_message(self, audio: bool) -> None:
        if not self.msg_item_id:
            return
        base = {"response_id": self.response_id, "item_id": self.msg_item_id,
                "output_index": 0, "content_index": 0}
        if audio:
            self.push({"type": "response.output_audio.done", **base})
            self.push({"type": "response.output_audio_transcript.done", **base, "transcript": self.text})
        else:
            self.push({"type": "response.output_text.done", **base, "text": self.text})
        part = {"type": "audio", "transcript": self.text} if audio else {"type": "text", "text": self.text}
        self.push({"type": "response.content_part.done", **base, "part": part})
        self.push({"type": "response.output_item.done", "response_id": self.response_id,
                   "output_index": 0, "item": self._message_item()})

    def _emit_tool_calls(self, calls: dict[int, dict]) -> None:
        idx = 1 if self.msg_item_id else 0
        for _, c in sorted(calls.items()):
            if not c["name"]:
                continue
            call_id = c["id"] or _id("call")
            item_id = _id("item")
            args = c["arguments"] or "{}"
            item = {"id": item_id, "object": "realtime.item", "type": "function_call",
                    "status": "completed", "call_id": call_id, "name": c["name"], "arguments": args}
            self.tool_calls.append({"id": call_id, "type": "function",
                                    "function": {"name": c["name"], "arguments": args}})
            self.push({"type": "response.output_item.added", "response_id": self.response_id,
                       "output_index": idx, "item": {**item, "status": "in_progress", "arguments": ""}})
            self.push({"type": "response.function_call_arguments.delta", "response_id": self.response_id,
                       "item_id": item_id, "output_index": idx, "call_id": call_id, "delta": args})
            self.push({"type": "response.function_call_arguments.done", "response_id": self.response_id,
                       "item_id": item_id, "output_index": idx, "call_id": call_id,
                       "name": c["name"], "arguments": args})
            self.push({"type": "response.output_item.done", "response_id": self.response_id,
                       "output_index": idx, "item": item})
            idx += 1
        self.tools_sent = bool(self.tool_calls)

    def _response_obj(self, status: str) -> dict:
        return {"id": self.response_id, "object": "realtime.response", "status": status,
                "output_modalities": list(self.s.conf["modalities"]), "output": []}

    def _done_event(self, status: str) -> dict:
        resp = self._response_obj(status)
        if self.msg_item_id:
            resp["output"].append(self._message_item())
        for tc in self.tool_calls:
            resp["output"].append({"type": "function_call", "object": "realtime.item", "status": "completed",
                                   "call_id": tc["id"], "name": tc["function"]["name"],
                                   "arguments": tc["function"]["arguments"]})
        resp["zallama"] = {"latency_ms": {k: round(v) for k, v in self.lat.items()}}
        return {"type": "response.done", "response": resp}

    async def _send_loop(self) -> None:
        await self.gate.wait()
        while (event := await self.out.get()) is not None:
            if event["type"] == "response.output_audio.delta":
                if self.first_sent is None:
                    self.first_sent = time.monotonic()
                    self.lat["first_audio_ms"] = (self.first_sent - self.speech_end) * 1000
                self.sent_ms += len(event["delta"]) * 3 / 4 / 2 * 1000 / OUT_RATE
            elif event["type"] == "response.done":
                event["response"]["zallama"]["latency_ms"] = {k: round(v) for k, v in self.lat.items()}
                logger.info("realtime turn %s latency %s", self.response_id,
                            event["response"]["zallama"]["latency_ms"])
            await self.s.send(event)
        self.s.turn_finished(self)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------
class RealtimeSession:
    def __init__(self, ws: WebSocket, cfg: dict, beta: bool, llm_model: str | None):
        rt = cfg.get("realtime") or {}
        self.ws = ws
        self.cfg = cfg
        self.rt = rt
        self.beta = beta
        self.pm = get_pm()
        self.registry = get_registry()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        self.send_lock = asyncio.Lock()
        self.session_id = _id("sess")
        self.llm_model = llm_model or rt.get("llm_model") or ""
        self.asr_model = rt.get("asr_model") or ""
        self.tts_model = rt.get("tts_model") or ""
        self.instances: dict[str, object] = {}
        self.holds: dict[str, AsyncExitStack] = {}
        self.tts_name: str | None = None       # TTS model currently held
        self.inst_locks: dict[str, asyncio.Lock] = {}
        self.conf = {
            "instructions": "",
            "tools": [],
            "tool_choice": "auto",
            "voice": rt.get("voice") or "",
            "modalities": ["audio"],
            "temperature": float(rt.get("temperature", 0.7)),
            "max_tokens": int(rt.get("max_tokens", 400)),
            "rate": 24000,
            "language": "",
            "turn_detection": {
                "type": "server_vad",
                "threshold": float(rt.get("vad_threshold", 0.5)),
                "prefix_padding_ms": int(rt.get("prefix_padding_ms", 300)),
                "silence_duration_ms": int(rt.get("silence_duration_ms", 500)),
                "speculative_ms": int(rt.get("speculative_ms", 200)),
                "barge_in_ms": int(rt.get("barge_in_ms", 250)),
                "min_speech_ms": int(rt.get("min_speech_ms", 96)),
                "create_response": True,
                "interrupt_response": True,
            },
        }
        self.language = rt.get("language") or None   # sticky: language of the last reply
        self.items: list[dict] = []    # {"id", "msg"} in conversation order
        self.vad: SileroVad | None = None
        self.vad_path = self._vad_path()
        self.buf = np.zeros(0, dtype=np.int16)
        self.buf_start = 0             # sample index of buf[0] in the input stream
        self.total = 0                 # input samples received
        self.user_item_id = _id("item")
        self._reset_vad_state()
        self.tentative: Turn | None = None
        self.active: Turn | None = None
        self.last: Turn | None = None
        self.warm_task: asyncio.Task | None = None
        self.bg: set[asyncio.Task] = set()
        self.warnings: list[str] = []         # reported after a session.update

    # -- plumbing ----------------------------------------------------------
    def _vad_path(self) -> str:
        p = self.rt.get("vad_model") or "silero_vad.onnx"
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = Path(self.cfg["zallama"]["models_dir"]).expanduser() / path
        return str(path)

    def _reset_vad_state(self) -> None:
        self.in_speech = False
        self.speech_run = 0
        self.speech_ms = 0
        self.silence_run = 0
        self.utt_start = 0
        if self.vad:
            self.vad.reset()

    async def send(self, event: dict) -> None:
        name = event["type"]
        if self.beta and name in _BETA_NAMES:
            name = _BETA_NAMES[name]
            if name is None:
                return
            event = {**event, "type": name}
        event.setdefault("event_id", _id("event"))
        async with self.send_lock:
            try:
                await self.ws.send_text(json.dumps(event, ensure_ascii=False))
            except Exception:
                pass    # the receive loop notices the disconnect

    async def error(self, message: str, code: str = "invalid_request_error", event_id: str | None = None) -> None:
        await self.send({"type": "error", "error": {"type": code, "message": message, "event_id": event_id}})

    def spawn(self, coro) -> asyncio.Task:
        t = asyncio.create_task(coro)
        self.bg.add(t)
        t.add_done_callback(self.bg.discard)
        return t

    async def instance(self, name: str, endpoint: str):
        """Start (or reuse) a backend and hold it for the whole session so no
        other request can evict it between turns."""
        async with self.inst_locks.setdefault(name, asyncio.Lock()):
            inst = self.instances.get(name)
            if inst is not None and inst.is_alive():
                return inst
            try:
                inst = await _resolve_instance(name, self.pm, self.registry, endpoint=endpoint)
            except HTTPException as e:
                raise RuntimeError(f"{name}: {e.detail}") from None
            hold = AsyncExitStack()
            await hold.enter_async_context(self.pm.serving(inst))
            self.holds[name] = hold
            self.instances[name] = inst
            return inst

    async def release(self, name: str) -> None:
        """Stop holding a backend, so it can be evicted again."""
        self.instances.pop(name, None)
        hold = self.holds.pop(name, None)
        if hold:
            await hold.aclose()

    # -- models ------------------------------------------------------------
    def tts_voices(self) -> list[str] | None:
        """Voices the session's TTS offers, or None when unknown (then any
        voice is passed through)."""
        try:
            name, entry = self.tts_target()
            from ..backends import get_backend
            voices_of = getattr(get_backend(ModelRegistry.backend_of(entry)), "voices", None)
            return list(voices_of(self.registry.resolve_path(entry))) if voices_of else None
        except Exception:
            return None

    def tts_language(self, phrase: str) -> str | None:
        """Language to speak a reply in: forced by the session, else detected
        on the reply's first phrase, else the previous reply's (detection
        abstains on short phrases like "Salut !"). The user's language is not
        used: the reply's is what the voice must match."""
        return self.conf["language"] or detect_language(phrase) or self.language

    def tts_target(self, lang: str | None = None) -> tuple[str, dict]:
        """The TTS entry to use: a language-routing entry (params.languages)
        picks its model from `lang` (default: the session's)."""
        entry = self.registry.get(self.tts_model)
        routes = (entry.get("params") or {}).get("languages")
        if isinstance(routes, dict) and routes:
            params = entry.get("params") or {}
            lang = lang or self.conf["language"] or self.language
            if lang not in routes:
                lang = params.get("default_language")
                if lang not in routes:
                    lang = next(iter(routes))
            name = str(routes[lang])
            return name, self.registry.get(name)
        return self.tts_model, entry

    async def transcribe(self, pcm: np.ndarray) -> str:
        inst = await self.instance(self.asr_model, "audio/transcriptions")
        inst.touch()
        files = {"file": ("speech.wav", _wav_bytes(pcm, self.conf["rate"]), "audio/wav")}
        data = {"model": self.asr_model, "response_format": "json"}
        async with self.pm.serving(inst, "audio/transcriptions"):
            r = await self.http.post(f"{inst.base_url}/v1/audio/transcriptions", data=data, files=files)
        r.raise_for_status()
        try:
            text = r.json().get("text", "")
        except ValueError:
            text = r.text
        return text.strip()

    async def chat_stream(self, messages: list):
        """Yield the `delta` dicts of a streamed chat completion."""
        inst = await self.instance(self.llm_model, "chat/completions")
        inst.touch()
        body = self._chat_body(messages)
        async with self.pm.serving(inst, "chat/completions", stream=True):
            async with self.http.stream("POST", f"{inst.base_url}/v1/chat/completions", json=body,
                                        timeout=httpx.Timeout(120.0, connect=5.0)) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"LLM {r.status_code}: {(await r.aread())[:300]!r}")
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[6:])
                    except ValueError:
                        continue
                    for choice in chunk.get("choices") or []:
                        if choice.get("delta"):
                            yield choice["delta"]

    def _chat_body(self, messages: list, **extra) -> dict:
        body = {
            "model": self.llm_model,
            "messages": messages,
            "stream": True,
            "temperature": self.conf["temperature"],
            "max_tokens": self.conf["max_tokens"],
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        tools = _chat_tools(self.conf["tools"])
        if tools:
            body["tools"] = tools
            body["tool_choice"] = self.conf["tool_choice"]
        body.update(extra)
        return body

    async def tts_instance(self, lang: str | None):
        """The TTS backend for `lang`. One per-language model (pocket-tts) is
        held at a time: switching releases the previous one first so it can be
        evicted to make room, and if the new one can't start the previous one
        keeps speaking (wrong accent beats silence)."""
        name, entry = self.tts_target(lang)
        prev = self.tts_name
        if prev and prev != name:
            await self.release(prev)
            try:
                inst = await self.instance(name, "audio/speech")
            except Exception as e:
                logger.warning("realtime: TTS %s unavailable, staying on %s: %s", name, prev, e)
                name, entry = prev, self.registry.get(prev)
                inst = await self.instance(name, "audio/speech")
        else:
            inst = await self.instance(name, "audio/speech")
        self.tts_name = name
        return inst, name, entry

    async def synthesize(self, text: str, lang: str | None = None):
        """Yield 24 kHz PCM16 bytes for `text` as the TTS produces it."""
        inst, name, entry = await self.tts_instance(lang)
        inst.touch()
        backend = ModelRegistry.backend_of(entry)
        voice = self.conf["voice"]
        if backend in ("pocket-tts-server", "xtts-server"):
            from ..backends import PocketTtsServerBackend, XttsServerBackend
            body = {"text": text}
            if backend == "xtts-server":
                # One model, every language: say which. An unknown voice
                # would fail the stream, so it falls back to the default.
                lang = lang or self.conf["language"] or self.language
                if lang in XttsServerBackend.LANGUAGES:
                    body["language"] = lang
                if voice:
                    known = XttsServerBackend().voices(self.registry.resolve_path(entry))
                    if match := XttsServerBackend.match_voice(voice, known):
                        body["voice"] = match
            # A predefined name, or inline audio to clone ("data:audio/wav;
            # base64,..."); never a path or hf:// URL (see openai.py).
            elif voice in PocketTtsServerBackend.VOICES or (
                    voice.startswith("data:audio/") and "base64," in voice):
                body["voice"] = voice
            async with self.pm.serving(inst, "audio/speech", stream=True):
                async with self.http.stream("POST", f"{inst.base_url}/stream", json=body) as r:
                    if r.status_code != 200:
                        raise RuntimeError(f"TTS {r.status_code}: {(await r.aread())[:300]!r}")
                    carry = b""
                    async for chunk in r.aiter_bytes():
                        chunk = carry + chunk
                        cut = len(chunk) & ~1
                        carry = chunk[cut:]
                        if cut:
                            yield chunk[:cut]
            return
        # Any other TTS: one non-streamed request per phrase, through the
        # regular /v1/audio/speech route (voice fallback, normalizer).
        body = {"model": name, "input": text, "response_format": "wav"}
        if voice:
            body["voice"] = voice
        port = self.cfg["zallama"]["port"]
        r = await self.http.post(f"http://127.0.0.1:{port}/v1/audio/speech", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"TTS {r.status_code}: {r.text[:300]}")
        yield _wav_to_pcm(r.content).astype("<i2").tobytes()

    def messages(self) -> list:
        system = self.rt.get("style_prompt", STYLE_PROMPT)
        if self.conf["instructions"]:
            system = f"{system}\n\n{self.conf['instructions']}".strip()
        msgs = [{"role": "system", "content": system}] if system else []
        self._trim_history(msgs)
        return msgs + [it["msg"] for it in self.items]

    def _llm_ctx(self) -> int:
        """Context of one llama-server slot of the voice LLM."""
        try:
            entry = self.registry.get(self.llm_model)
        except Exception:
            return 4096
        params = {**(self.cfg.get("llama_server") or {}).get("default_params", {}),
                  **(entry.get("params") or {})}
        ctx = int(params.get("ctx_size") or 4096)
        parallel = max(1, int(params.get("parallel") or 1))
        return ctx if params.get("kv_unified") else ctx // parallel

    def _trim_history(self, head: list) -> None:
        """Drop the oldest turns once the prompt would overflow the LLM's
        context (llama-server rejects it, and the turn would fail).

        Tokens are estimated at 3 characters each (French runs ~3.5-4). It
        trims down to 60% of the budget at once rather than one turn per
        reply: every trim changes the prompt's start, so the whole history is
        prefilled again, and that should happen rarely.
        """
        budget = int(self.rt.get("history_tokens") or 0) or (
            self._llm_ctx() - self.conf["max_tokens"] - 256)
        est = lambda obj: len(json.dumps(obj, ensure_ascii=False)) // 3 + 4
        fixed = sum(est(m) for m in head) + est(_chat_tools(self.conf["tools"]))
        sizes = [est(it["msg"]) for it in self.items]
        if fixed + sum(sizes) <= budget:
            return
        target, total, cut = int(budget * 0.6), fixed + sum(sizes), 0
        while cut < len(self.items) and total > target:
            total -= sizes[cut]
            cut += 1
        # Start on a user message: never on a tool result or an assistant
        # tool call whose request was dropped.
        while cut < len(self.items) and self.items[cut]["msg"].get("role") != "user":
            cut += 1
        if cut:
            logger.info("realtime %s: history trimmed, %d of %d items dropped (budget %d tokens)",
                        self.session_id, cut, len(self.items), budget)
            del self.items[:cut]

    async def warm(self) -> None:
        """Load the models and prefill the system prompt + tools, so the first
        turn neither waits for a model load nor recomputes the prompt."""
        tasks = [self.instance(self.llm_model, "chat/completions"),
                 self.instance(self.asr_model, "audio/transcriptions")]
        if "audio" in self.conf["modalities"]:
            tasks.append(self.tts_instance(None))
        await asyncio.gather(*tasks)
        self._prefill()

    def _prefill(self) -> None:
        if self.warm_task and not self.warm_task.done():
            self.warm_task.cancel()

        async def run():
            try:
                inst = await self.instance(self.llm_model, "chat/completions")
                body = self._chat_body(self.messages(), stream=False, max_tokens=1)
                await self.http.post(f"{inst.base_url}/v1/chat/completions", json=body)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.debug("realtime prefill failed: %s", e)
        self.warm_task = self.spawn(run())

    # -- session config ----------------------------------------------------
    def session_obj(self) -> dict:
        td = self.conf["turn_detection"]
        return {
            "id": self.session_id, "object": "realtime.session", "type": "realtime",
            "model": self.llm_model, "instructions": self.conf["instructions"],
            "output_modalities": self.conf["modalities"], "tools": self.conf["tools"],
            "tool_choice": self.conf["tool_choice"], "max_output_tokens": self.conf["max_tokens"],
            "audio": {
                "input": {"format": {"type": "audio/pcm", "rate": self.conf["rate"]},
                          "turn_detection": td,
                          "transcription": {"model": self.asr_model}},
                "output": {"format": {"type": "audio/pcm", "rate": OUT_RATE},
                           "voice": self.conf["voice"][:64]},
            },
            "zallama": {"asr_model": self.asr_model, "tts_model": self.tts_model,
                        "voices": self.tts_voices() or [],
                        "language": self.conf["language"] or self.language},
        }

    def apply_session(self, s: dict) -> bool:
        """Apply a session.update (GA or beta shape). Returns True when the
        prompt changed (instructions/tools), i.e. the prefill must be redone."""
        conf = self.conf
        before = (conf["instructions"], json.dumps(conf["tools"]), conf["tool_choice"])
        audio = s.get("audio") or {}
        ain, aout = audio.get("input") or {}, audio.get("output") or {}
        if "instructions" in s:
            conf["instructions"] = str(s["instructions"] or "")
        if "tools" in s:
            conf["tools"] = list(s["tools"] or [])
        if "tool_choice" in s:
            conf["tool_choice"] = s["tool_choice"]
        mods = s.get("output_modalities") or s.get("modalities")
        if mods:
            conf["modalities"] = ["audio"] if "audio" in mods else ["text"]
        voice = aout.get("voice", s.get("voice"))
        if isinstance(voice, str):
            voice = voice.strip()
            known = self.tts_voices()
            if voice and known and voice not in known:
                from ..backends import XttsServerBackend
                voice = XttsServerBackend.match_voice(voice, known) or voice
            if not voice or voice.startswith("data:audio/") or known is None or voice in known:
                conf["voice"] = voice
            else:
                # OpenAI SDKs send "alloy", "marin"...: the TTS has none of
                # them. Say so instead of silently speaking the default.
                self.warnings.append(
                    f"Unknown voice '{voice[:40]}' for {self.tts_model}; keeping "
                    f"'{conf['voice'] or 'default'}'. Voices: {', '.join(known)}")
        for key in ("temperature",):
            if key in s:
                conf[key] = float(s[key])
        mt = s.get("max_output_tokens", s.get("max_response_output_tokens"))
        if mt is not None:
            conf["max_tokens"] = 4096 if mt == "inf" else int(mt)
        if s.get("language") is not None:
            conf["language"] = str(s["language"])
        fmt = ain.get("format")
        if isinstance(fmt, dict) and fmt.get("rate"):
            rate = int(fmt["rate"])
            if rate != conf["rate"]:
                conf["rate"] = rate
                self.vad = None
        if "turn_detection" in ain or "turn_detection" in s:
            td = ain.get("turn_detection", s.get("turn_detection"))
            if td is None:
                conf["turn_detection"] = None
            elif isinstance(td, dict):
                cur = conf["turn_detection"] or {}
                merged = {**self._default_td(), **cur, **td}
                merged["type"] = "server_vad"
                conf["turn_detection"] = merged
        if s.get("model") and s["model"] != self.llm_model and self._is_text_model(s["model"]):
            self.llm_model = s["model"]
            return True
        return before != (conf["instructions"], json.dumps(conf["tools"]), conf["tool_choice"])

    def _default_td(self) -> dict:
        rt = self.rt
        return {"type": "server_vad", "threshold": float(rt.get("vad_threshold", 0.5)),
                "prefix_padding_ms": int(rt.get("prefix_padding_ms", 300)),
                "silence_duration_ms": int(rt.get("silence_duration_ms", 500)),
                "speculative_ms": int(rt.get("speculative_ms", 200)),
                "barge_in_ms": int(rt.get("barge_in_ms", 250)),
                "min_speech_ms": int(rt.get("min_speech_ms", 96)),
                "create_response": True, "interrupt_response": True}

    def _is_text_model(self, name: str) -> bool:
        try:
            return ModelRegistry.modality_of(self.registry.get(name)) == "text"
        except Exception:
            return False

    # -- audio in ----------------------------------------------------------
    def ms(self, samples: int) -> int:
        return int(samples * 1000 / self.conf["rate"])

    def samples(self, ms: float) -> int:
        return int(ms * self.conf["rate"] / 1000)

    def audio_between(self, start: int, end: int) -> np.ndarray:
        return self.buf[max(0, start - self.buf_start):max(0, end - self.buf_start)].copy()

    async def on_audio(self, pcm: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, pcm])
        td = self.conf["turn_detection"]
        if td is None:
            self.total += len(pcm)
            return
        if self.vad is None:
            self.vad = SileroVad(self.vad_path, self.conf["rate"])
            self._reset_vad_state()
        frame = self.samples(FRAME_MS)
        thr = float(td["threshold"])
        for p in self.vad.probs(pcm):
            self.total = min(self.total + frame, self.buf_start + len(self.buf))
            await self._vad_step(p, thr, td)
        if not self.in_speech:
            keep = self.samples(td["prefix_padding_ms"] + td["min_speech_ms"] + 200)
            if len(self.buf) > keep * 4:
                drop = len(self.buf) - keep
                self.buf = self.buf[drop:]
                self.buf_start += drop

    async def _vad_step(self, p: float, thr: float, td: dict) -> None:
        if not self.in_speech:
            self.speech_run = self.speech_run + 1 if p >= thr else 0
            if self.speech_run * FRAME_MS >= td["min_speech_ms"]:
                self.in_speech = True
                self.speech_ms = self.speech_run * FRAME_MS
                self.silence_run = 0
                lead = self.samples(self.speech_ms + td["prefix_padding_ms"])
                self.utt_start = max(self.buf_start, self.total - lead)
                self.user_item_id = _id("item")
                await self.send({"type": "input_audio_buffer.speech_started",
                                 "audio_start_ms": self.ms(self.utt_start), "item_id": self.user_item_id})
                self._maybe_barge(td)
            return
        if p >= thr:
            if self.silence_run and self.tentative:
                self.tentative.cancel()     # the pause was not the end of the turn
                self.tentative = None
            self.silence_run = 0
            self.speech_ms += FRAME_MS
            self._maybe_barge(td)
        elif p < thr - 0.15:
            self.silence_run += 1
        silence = self.silence_run * FRAME_MS
        if not silence:
            return
        speaking = self.active is not None
        if (not speaking and self.tentative is None and td.get("create_response", True)
                and silence >= td["speculative_ms"] and td["speculative_ms"] < td["silence_duration_ms"]):
            self.tentative = Turn(self, self.audio_between(self.utt_start, self.total), gated=True,
                                  speech_end=time.monotonic() - silence / 1000)
        if silence >= td["silence_duration_ms"]:
            await self._end_of_turn(td, silence)

    def _maybe_barge(self, td: dict) -> None:
        if not td.get("interrupt_response", True) or self.speech_ms < td["barge_in_ms"]:
            return
        if self.active is not None:
            self.interrupt_now()
        elif self.last is not None and not self.last.finished_playing():
            self._truncate(self.last, self.last.played_ms())
            self.last = None

    async def _end_of_turn(self, td: dict, silence: int) -> None:
        end = self.total
        speech_ms = self.speech_ms
        item_id = self.user_item_id
        self._reset_vad_state()
        turn, self.tentative = self.tentative, None
        if self.active is not None:
            # A short noise while the assistant talks (shorter than barge_in_ms):
            # not a turn.
            if turn:
                turn.cancel()
            return
        await self.send({"type": "input_audio_buffer.speech_stopped",
                         "audio_end_ms": self.ms(end), "item_id": item_id})
        await self.send({"type": "input_audio_buffer.committed", "item_id": item_id,
                         "previous_item_id": self.items[-1]["id"] if self.items else None})
        if turn is None:
            audio = self.audio_between(self.utt_start, end)
            if not td.get("create_response", True):
                self.spawn(self._commit_audio(audio, item_id))
                return
            turn = Turn(self, audio, gated=True, speech_end=time.monotonic() - silence / 1000)
        turn.user_item_id = item_id
        self.active = turn
        self.spawn(self._release(turn))

    async def _release(self, turn: Turn) -> None:
        """Open a committed turn's gate once its transcript is in the history."""
        text = await turn.transcript
        if self.active is not turn:
            # Interrupted before it spoke: the words were still said.
            if text:
                await self._add_user_transcript(turn.user_item_id, text)
            return
        if not text:
            turn.cancel()
            self.active = None
            await self.send({"type": "conversation.item.input_audio_transcription.completed",
                             "item_id": turn.user_item_id, "content_index": 0, "transcript": ""})
            return
        await self._add_user_transcript(turn.user_item_id, text)
        turn.gate.set()

    async def _commit_audio(self, audio: np.ndarray, item_id: str) -> None:
        try:
            text = await self.transcribe(audio)
        except Exception as e:
            await self.error(f"transcription failed: {e}", "server_error")
            return
        if text:
            await self._add_user_transcript(item_id, text)

    async def _add_user_transcript(self, item_id: str, text: str) -> None:
        prev = self.items[-1]["id"] if self.items else None
        self.items.append({"id": item_id, "msg": {"role": "user", "content": text}})
        item = {"id": item_id, "object": "realtime.item", "type": "message", "role": "user",
                "status": "completed", "content": [{"type": "input_audio", "transcript": text}]}
        await self.send({"type": "conversation.item.added", "previous_item_id": prev, "item": item})
        await self.send({"type": "conversation.item.input_audio_transcription.completed",
                         "item_id": item_id, "content_index": 0, "transcript": text})
        await self.send({"type": "conversation.item.done", "previous_item_id": prev, "item": item})

    # -- responses ---------------------------------------------------------
    def turn_finished(self, turn: Turn) -> None:
        """Sender drained (response.done sent): record the reply in the history."""
        if self.active is turn:
            self.active = None
        if not turn.finished:
            turn.finished = True
            self._record(turn, None)
            self.last = turn

    def _record(self, turn: Turn, played_ms: float | None) -> None:
        text = turn.heard_text(played_ms) if played_ms is not None else turn.text
        tool_calls = turn.tool_calls if turn.tools_sent else []
        if not text and not tool_calls:
            return
        msg: dict = {"role": "assistant", "content": text or None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.items.append({"id": turn.msg_item_id or _id("item"), "msg": msg, "turn": turn})

    def _truncate(self, turn: Turn, played_ms: float) -> None:
        for it in reversed(self.items):
            if it.get("turn") is turn:
                it["msg"]["content"] = turn.heard_text(played_ms) or None
                if not it["msg"]["content"] and not it["msg"].get("tool_calls"):
                    self.items.remove(it)
                return

    def interrupt_now(self, reason: str = "turn_detected") -> None:
        """Barge-in, from inside the VAD loop: drop the response at once (so
        the rest of this utterance is a new turn) and notify in the background."""
        turn = self.active
        if turn is not None:
            self.active = None
            self.spawn(self.interrupt(reason, turn))

    async def interrupt(self, reason: str = "turn_detected", turn: Turn | None = None) -> None:
        if turn is None:
            turn, self.active = self.active, None
        if turn is None:
            return
        played = turn.played_ms()
        turn.cancel(keep_transcript=True)
        if not turn.finished:
            turn.finished = True
            if turn.gate.is_set():
                self._record(turn, played)
        await self.send({"type": "response.done", "response": {
            **turn._response_obj("cancelled"),
            "status_details": {"type": "cancelled", "reason": reason},
            "zallama": {"played_ms": round(played)}}})

    async def create_response(self, event_id: str | None) -> None:
        if self.active is not None:
            await self.error("Conversation already has an active response",
                             "conversation_already_has_active_response", event_id)
            return
        if self.tentative:
            self.tentative.cancel()
            self.tentative = None
        self.active = Turn(self, None, gated=False)

    # -- client events -----------------------------------------------------
    async def handle(self, ev: dict) -> None:
        kind = ev.get("type", "")
        eid = ev.get("event_id")
        if kind == "input_audio_buffer.append":
            try:
                pcm = np.frombuffer(base64.b64decode(ev.get("audio", "")), dtype="<i2")
            except Exception:
                await self.error("audio must be base64 PCM16", event_id=eid)
                return
            await self.on_audio(pcm)
        elif kind == "session.update":
            if self.apply_session(ev.get("session") or {}):
                self._prefill()
            for msg in self.warnings:
                await self.send({"type": "error", "error": {
                    "type": "invalid_request_error", "code": "unknown_voice",
                    "param": "session.audio.output.voice", "message": msg, "event_id": eid}})
            self.warnings.clear()
            await self.send({"type": "session.updated", "session": self.session_obj()})
        elif kind == "response.create":
            r = ev.get("response") or {}
            if r.get("instructions"):
                # one-off instructions: appended as a system note for this reply
                self.items.append({"id": _id("item"), "msg": {"role": "system", "content": r["instructions"]}})
            await self.create_response(eid)
        elif kind == "response.cancel":
            await self.interrupt("client_cancelled")
        elif kind == "input_audio_buffer.commit":
            audio = self.buf.copy()
            self.buf = np.zeros(0, dtype=np.int16)
            self.buf_start = self.total
            item_id = _id("item")
            await self.send({"type": "input_audio_buffer.committed", "item_id": item_id,
                             "previous_item_id": self.items[-1]["id"] if self.items else None})
            await self._commit_audio(audio, item_id)
        elif kind == "input_audio_buffer.clear":
            self.buf = np.zeros(0, dtype=np.int16)
            self.buf_start = self.total
            self._reset_vad_state()
            await self.send({"type": "input_audio_buffer.cleared"})
        elif kind == "conversation.item.create":
            await self._create_item(ev.get("item") or {}, ev.get("previous_item_id"), eid)
        elif kind == "conversation.item.delete":
            iid = ev.get("item_id")
            self.items = [it for it in self.items if it["id"] != iid]
            await self.send({"type": "conversation.item.deleted", "item_id": iid})
        elif kind == "conversation.item.truncate":
            iid, end_ms = ev.get("item_id"), ev.get("audio_end_ms")
            turn = next((t for t in (self.active, self.last) if t and t.msg_item_id == iid), None)
            turn = turn or next((it["turn"] for it in self.items if it["id"] == iid and it.get("turn")), None)
            if turn is not None and end_ms is not None:
                turn.client_played = float(end_ms)   # the client knows what it played
                if turn.finished:
                    self._truncate(turn, float(end_ms))
            await self.send({"type": "conversation.item.truncated", "item_id": iid,
                             "content_index": ev.get("content_index", 0), "audio_end_ms": end_ms})
        elif kind == "output_audio_buffer.clear":
            await self.send({"type": "output_audio_buffer.cleared"})
        else:
            await self.error(f"Unsupported event type '{kind}'", event_id=eid)

    async def _create_item(self, item: dict, previous: str | None, eid: str | None) -> None:
        itype = item.get("type", "message")
        iid = item.get("id") or _id("item")
        if itype == "function_call_output":
            msg = {"role": "tool", "tool_call_id": item.get("call_id", ""),
                   "content": str(item.get("output", ""))}
        elif itype == "function_call":
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": item.get("call_id") or _id("call"), "type": "function",
                "function": {"name": item.get("name", ""), "arguments": item.get("arguments", "{}")}}]}
        elif itype == "message":
            role = item.get("role", "user")
            parts = []
            for c in item.get("content") or []:
                if c.get("type") in ("input_text", "text", "output_text"):
                    parts.append(c.get("text", ""))
                elif c.get("transcript"):
                    parts.append(c["transcript"])
                elif c.get("type") == "input_audio" and c.get("audio"):
                    pcm = np.frombuffer(base64.b64decode(c["audio"]), dtype="<i2")
                    parts.append(await self.transcribe(pcm))
            msg = {"role": role if role in ("user", "assistant", "system") else "user",
                   "content": " ".join(p for p in parts if p)}
        else:
            await self.error(f"Unsupported item type '{itype}'", event_id=eid)
            return
        prev = self.items[-1]["id"] if self.items else None
        self.items.append({"id": iid, "msg": msg})
        out = {**item, "id": iid, "object": "realtime.item", "status": "completed"}
        await self.send({"type": "conversation.item.added", "previous_item_id": prev, "item": out})
        await self.send({"type": "conversation.item.done", "previous_item_id": prev, "item": out})

    async def close(self) -> None:
        for t in (self.tentative, self.active):
            if t:
                t.cancel()
        for t in list(self.bg):
            t.cancel()
        for name in list(self.holds):
            await self.release(name)
        await self.http.aclose()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
def _authorized(ws: WebSocket) -> tuple[bool, str | None]:
    """Same API key as the HTTP routes (the HTTP middleware doesn't see
    WebSockets). Browsers can't set headers on a WebSocket, so the key may
    also come as the `openai-insecure-api-key.<key>` subprotocol, like
    OpenAI's. Returns (ok, subprotocol to accept)."""
    protocols = [p.strip() for p in ws.headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
    accept = "realtime" if "realtime" in protocols else None
    check = getattr(ws.app.state, "check_api_key", None)
    if check is None or (ws.client and ws.client.host in ("127.0.0.1", "::1")):
        return True, accept
    auth = ws.headers.get("authorization", "")
    token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
    for p in protocols:
        if p.startswith("openai-insecure-api-key."):
            token = token or p[len("openai-insecure-api-key."):]
    return check(token), accept


@router.websocket("/v1/realtime")
async def realtime(ws: WebSocket):
    ok, subprotocol = _authorized(ws)
    if not ok:
        await ws.close(code=1008, reason="Invalid or missing API key")
        return
    await ws.accept(subprotocol=subprotocol)
    cfg = ws.app.state.cfg
    model = ws.query_params.get("model")
    session = RealtimeSession(ws, cfg, beta=ws.query_params.get("events") == "beta",
                              llm_model=model if model and _is_text(model) else None)
    missing = [k for k, v in (("realtime.llm_model", session.llm_model),
                              ("realtime.asr_model", session.asr_model),
                              ("realtime.tts_model", session.tts_model)) if not v]
    if missing:
        await session.error(f"Not configured: {', '.join(missing)} (config.yaml)", "server_error")
        await ws.close()
        return
    if not Path(session.vad_path).is_file():
        await session.error(
            f"VAD model not found: {session.vad_path} (download "
            "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx "
            "into models_dir)", "server_error")
        await ws.close()
        return
    try:
        try:
            await session.warm()
        except Exception as e:
            await session.error(f"Failed to start models: {e}", "server_error")
            await ws.close()
            return
        await session.send({"type": "session.created", "session": session.session_obj()})
        while True:
            raw = await ws.receive_text()
            try:
                ev = json.loads(raw)
            except ValueError:
                await session.error("Invalid JSON")
                continue
            await session.handle(ev)
    except WebSocketDisconnect:
        pass
    except RuntimeError as e:   # receive after close
        logger.debug("realtime socket closed: %s", e)
    finally:
        await session.close()


def _is_text(name: str) -> bool:
    try:
        return ModelRegistry.modality_of(get_registry().get(name)) == "text"
    except Exception:
        return False


@router.get("/realtime", include_in_schema=False)
async def realtime_demo(request: Request):
    """Browser demo: talk to /v1/realtime with the microphone. Off unless
    `realtime.demo: true` (`zallama realtime demo on`)."""
    if not (request.app.state.cfg.get("realtime") or {}).get("demo"):
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(Path(__file__).with_name("realtime_demo.html"), media_type="text/html")


def make_key_checker(expected_digest: str, expires_at):
    """Used by main._install_auth to share the API-key check with /v1/realtime."""
    from datetime import datetime, timezone

    def check(token: str) -> bool:
        digest = hashlib.sha256(token.encode()).hexdigest()
        if not secrets.compare_digest(digest, expected_digest):
            return False
        return not (expires_at and datetime.now(timezone.utc) > expires_at)
    return check
