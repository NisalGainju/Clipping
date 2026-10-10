"""Audio energy analysis (loudness spikes, sudden changes) on 16 kHz mono WAV."""
from __future__ import annotations

import wave
from dataclasses import dataclass

import numpy as np


@dataclass
class AudioFeatures:
    hop: float
    level_z: np.ndarray   # robust z-score of log-RMS per frame
    jump_z: np.ndarray    # robust z-score of |frame-to-frame change|

    def window(self, start: float, end: float) -> dict:
        a, b = int(start / self.hop), max(int(end / self.hop), int(start / self.hop) + 1)
        lvl, jmp = self.level_z[a:b], self.jump_z[a:b]
        if lvl.size == 0:
            return {"energy": 0.0, "peak": 0.0, "jump": 0.0, "score": 0.0}
        k = max(1, int(lvl.size * 0.2))
        energy = float(np.clip(np.sort(lvl)[-k:].mean() / 3.0, 0, 1))
        peak = float(np.clip(lvl.max() / 4.0, 0, 1))
        jump = float(np.clip(np.sort(jmp)[-k:].mean() / 3.0, 0, 1))
        return {"energy": energy, "peak": peak, "jump": jump,
                "score": 0.5 * energy + 0.2 * peak + 0.3 * jump}


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / (mad if mad > 1e-6 else (x.std() or 1.0))


def compute_audio_features(wav_path: str, hop: float = 0.5) -> AudioFeatures:
    rms: list[float] = []
    with wave.open(str(wav_path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise ValueError("Expected 16-bit PCM WAV (use extract_audio).")
        frames = int(sr * hop)
        while True:
            raw = w.readframes(frames)  # streamed: constant memory for multi-hour audio
            if not raw:
                break
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                x = x[: len(x) - len(x) % ch].reshape(-1, ch).mean(axis=1)
            rms.append(float(np.sqrt(np.mean(x * x) + 1e-12)))
    log_rms = np.log(np.array(rms) + 1e-5)
    jumps = np.abs(np.diff(log_rms, prepend=log_rms[:1]))
    return AudioFeatures(hop=hop, level_z=_robust_z(log_rms), jump_z=_robust_z(jumps))
