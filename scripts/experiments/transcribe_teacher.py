"""Transcribe teacher's recorded remarks (mp4 -> wav -> text) with faster-whisper."""
import os
import sys
import time
import wave

import numpy as np

# route model downloads through the reachable mirror
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

from faster_whisper import WhisperModel


def load_wav(path: str, target_sr: int = 16000):
    """Read a PCM wav into a float32 numpy array, bypassing PyAV."""
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        sr = w.getframerate()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    if sr != target_sr:
        # naive linear resample (16k is our target and already correct)
        raise ValueError(f"unexpected sample rate {sr}")
    return data.astype(np.float32)


def main():
    audio = sys.argv[1] if len(sys.argv) > 1 else ".tmp_audio/teacher.wav"
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"
    out_txt = sys.argv[3] if len(sys.argv) > 3 else ".tmp_audio/teacher_transcript.txt"

    print(f"[transcribe] loading model '{model_size}' ...", flush=True)
    t0 = time.time()
    model = WhisperModel(model_size, device="cpu", compute_type="int8",
                         download_root=".tmp_audio/models")
    print(f"[transcribe] model loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"[transcribe] transcribing {audio} ...", flush=True)
    t0 = time.time()
    samples = load_wav(audio, target_sr=16000)
    print(f"[transcribe] {samples.shape[0] / 16000:.1f}s of audio loaded", flush=True)
    segments, info = model.transcribe(
        samples,
        language="zh",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    lines = []
    for seg in segments:
        line = f"[{_ts(seg.start)} -> {_ts(seg.end)}] {seg.text.strip()}"
        lines.append(line)
        print(line, flush=True)

    text = "\n".join(lines)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[done] {len(lines)} segments in {time.time() - t0:.1f}s -> {out_txt}")


def _ts(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    main()
