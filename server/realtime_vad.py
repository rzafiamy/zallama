"""
realtime_vad.py — Voice activity detection for /v1/realtime

Silero VAD (ONNX, ~2 MB, CPU, well under 1 ms per 32 ms frame) decides when
the user starts and stops talking. It runs in the zallama process itself: a
separate backend would add an HTTP round trip to every 32 ms frame.

Silero takes 16 kHz frames of 512 samples, each preceded by the last 64
samples of the previous one (the official OnnxWrapper's context), plus a
recurrent state. Clients send 24 kHz PCM (the OpenAI Realtime default), so the
input is resampled for the VAD only; the ASR gets the original audio.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

VAD_RATE = 16000
FRAME = 512             # samples per VAD frame at 16 kHz (32 ms)
FRAME_MS = FRAME * 1000 // VAD_RATE
_CONTEXT = 64

_sessions: dict[str, object] = {}
_sessions_lock = threading.Lock()


def _session(model_path: str):
    """One onnxruntime session per model file, shared by every connection
    (each connection keeps its own recurrent state)."""
    with _sessions_lock:
        sess = _sessions.get(model_path)
        if sess is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            sess = ort.InferenceSession(
                model_path, sess_options=opts, providers=["CPUExecutionProvider"])
            _sessions[model_path] = sess
        return sess


class LinearResampler:
    """Streaming linear-interpolation resampler for mono float32.

    Good enough for a VAD (which only needs the speech envelope); the fractional
    read position carries over between calls so chunk boundaries don't click.
    """

    def __init__(self, src_rate: int, dst_rate: int):
        self.step = src_rate / dst_rate
        self.pos = 0.0                  # next read position, relative to `self.prev`
        self.prev = np.zeros(1, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.step == 1.0:
            return x
        buf = np.concatenate([self.prev, x])
        n = int((len(buf) - 1 - self.pos) // self.step) + 1
        if n <= 0:
            self.prev = buf
            return np.zeros(0, dtype=np.float32)
        idx = self.pos + np.arange(n) * self.step
        out = np.interp(idx, np.arange(len(buf)), buf).astype(np.float32)
        nxt = self.pos + n * self.step
        keep = int(nxt)                 # drop the samples fully consumed
        self.prev = buf[keep:]
        self.pos = nxt - keep
        return out


class SileroVad:
    """Per-connection Silero state: feed PCM at `src_rate`, get one speech
    probability per 32 ms frame."""

    def __init__(self, model_path: str | Path, src_rate: int):
        self.sess = _session(str(model_path))
        self.resample = LinearResampler(src_rate, VAD_RATE)
        self.pending = np.zeros(0, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def probs(self, pcm16: np.ndarray) -> list[float]:
        x = self.resample(pcm16.astype(np.float32) / 32768.0)
        self.pending = np.concatenate([self.pending, x])
        out: list[float] = []
        sr = np.array(VAD_RATE, dtype=np.int64)
        while len(self.pending) >= FRAME:
            frame = self.pending[:FRAME][None, :]
            self.pending = self.pending[FRAME:]
            inp = np.concatenate([self.context, frame], axis=1)
            prob, self.state = self.sess.run(
                None, {"input": inp, "state": self.state, "sr": sr})
            self.context = inp[:, -_CONTEXT:]
            out.append(float(prob[0][0]))
        return out
