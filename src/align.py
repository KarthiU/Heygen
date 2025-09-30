from pathlib import Path
import re, glob
import numpy as np, librosa, soundfile as sf
from typing import List, Tuple, Optional

TS = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})")
# Regex for SRT timestamp lines.


def _to_ms(h, m, s, ms):
    """
    Convert (h, m, s, ms) to total milliseconds.
    """
    return (int(h)*3600 + int(m)*60 + int(s))*1000 + int(ms)

def _ms_to_ts(ms):
    """
    Format ms as SRT timestamp (hh:mm:ss,ms).
    """
    if ms < 0: ms = 0
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def read_srt(path):
    """
    Parse SRT file and return list of (start_ms, end_ms, text) tuples.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out, block = [], []
    for ln in lines + [""]:
        if ln.strip():
            block.append(ln.strip()); continue
        if len(block) >= 2:
            m = TS.match(block[1])
            if m:
                h1, m1, s1, ms1, h2, m2, s2, ms2 = m.groups()
                start = _to_ms(h1, m1, s1, ms1)
                end = _to_ms(h2, m2, s2, ms2)
                text = " ".join(t for t in block[2:] if t.strip())
                out.append((start, end, text))
        block = []
    return out

def load_mono(path, sr_target=None):
    """
    Load audio file as mono. Optionally resample to sr_target. Returns (audio, sample_rate).
    """
    y, sr = librosa.load(str(path), sr=None, mono=True)
    if sr_target is not None and sr != sr_target:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    return y, sr

def build_timeline(segments, sr, total_ms):
    """
    Place each audio segment at its start_ms in a global buffer of total_ms.
    Overwrites overlapping regions (no mixing).
    """
    total_samples = int(np.ceil(total_ms * sr / 1000.0))
    out = np.zeros(total_samples, dtype=np.float32)
    for seg in segments:
        a = int(np.round(seg['start_ms'] * sr / 1000.0))
        b = a + len(seg['audio_np'])
        if a < 0 or a >= total_samples:
            continue
        seg_data = seg['audio_np']
        if b > total_samples:
            seg_data = seg_data[: total_samples - a]
            b = total_samples
        out[a:b] = seg_data[: b - a]
    return out

def segment_audio_by_cues(full_audio_path, cues, sr):
    """
    Slice the full synthesized audio into segments according to SRT cue timings.
    Returns list of dicts: {'start_ms', 'end_ms', 'audio_np'}
    """
    y_full, sr_full = load_mono(full_audio_path, sr_target=sr)
    segments = []
    for (start_ms, end_ms, _) in cues:
        a = int(np.round(start_ms * sr_full / 1000.0))
        b = int(np.round(end_ms * sr_full / 1000.0))
        seg_audio = y_full[a:b]
        segments.append({
            'start_ms': start_ms,
            'end_ms': end_ms,
            'audio_np': seg_audio.astype(np.float32)
        })
    return segments

def ms_to_srt_timestamp(ms):
    """
    Convert milliseconds to SRT timestamp string (hh:mm:ss,ms).
    """
    h = int(ms // 3600000)
    ms = ms % 3600000
    m = int(ms // 60000)
    ms = ms % 60000
    s = int(ms // 1000)
    ms = int(ms % 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_translated_srt(out_path, cues, translated_texts):
    """
    Write translated SRT file using cues and translated texts.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, ((start_ms, end_ms, _), text_de) in enumerate(zip(cues, translated_texts), 1):
            f.write(f"{idx}\n")
            f.write(f"{ms_to_srt_timestamp(start_ms)} --> {ms_to_srt_timestamp(end_ms)}\n")
            f.write(f"{text_de}\n\n")