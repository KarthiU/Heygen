from pathlib import Path
import re, glob, os

import numpy as np
import librosa, soundfile as sf
from src.translate import translate_texts  


# --- SRT helpers -------------------------------------------------------------

TS = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})")

def _to_ms(h, m, s, ms):
    return (int(h)*3600 + int(m)*60 + int(s)) * 1000 + int(ms)

def _ms_to_ts(ms):
    if ms < 0: ms = 0
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def read_srt(path):
    """
    Return list of (start_ms, end_ms, full_text_per_cue).
    Note: text lines inside a cue are joined with spaces to keep 1 cue = 1 string.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    out, block = [], []
    for ln in lines + [""]:  # force flush at EOF
        if ln.strip():
            block.append(ln.strip())
            continue
        if len(block) >= 2:
            m = TS.match(block[1])
            if m:
                h1,m1,s1,ms1,h2,m2,s2,ms2 = m.groups()
                start = _to_ms(h1,m1,s1,ms1)
                end   = _to_ms(h2,m2,s2,ms2)
                text  = " ".join(t for t in block[2:] if t.strip())
                out.append((start, end, text))
        block = []
    return out

def write_srt(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for i, (start_ms, end_ms, text) in enumerate(items, 1):
            f.write(str(i) + "\n")
            f.write(f"{_ms_to_ts(start_ms)} --> {_ms_to_ts(end_ms)}\n")
            f.write((text or "").strip() + "\n\n")


# --- optional: segments + stretch -------------------------------------------

def list_segments(spec):
    if not spec: return []
    p = Path(spec)
    if p.is_dir():
        return sorted([x for x in p.iterdir() if x.suffix.lower() in (".wav",".flac",".mp3")])
    return sorted(Path(x) for x in glob.glob(spec))

def stretch_to_window(wav_in, target_ms, max_ratio=1.05):
    if target_ms <= 0:
        return str(wav_in)
    y, sr = librosa.load(str(wav_in), sr=None, mono=True)
    if len(y) == 0:
        return str(wav_in)
    want = int(target_ms * sr / 1000)
    if want <= 0:
        return str(wav_in)
    ratio = want / len(y)
    ratio = max(min(ratio, max_ratio), 1.0/max_ratio)
    y2 = librosa.effects.time_stretch(y, rate=1.0/ratio)
    if len(y2) < want:
        y2 = np.pad(y2, (0, want - len(y2)))
    else:
        y2 = y2[:want]
    out = Path(wav_in).with_suffix("")  # drop ext
    out = out.with_name(out.name + "_stretch.wav")
    sf.write(str(out), y2, sr)
    return str(out)


# --- main --------------------------------------------------------------------

if __name__ == "__main__":
    SRT = "samples/Tanzania-caption.srt"
    SEGMENTS = None                      # e.g. "gen_audio/*.wav" or a folder
    MAX_RATIO = 1.05
    OUT_SRT = "outputs/translated.srt"
    TMP_LINES = "outputs/_tmp_cue_lines.txt"  # temp: one line per cue

    subs = read_srt(SRT)  # [(start_ms, end_ms, full_cue_text), ...]

    # 1) Write a temp file with ONE LINE PER CUE so translate_texts returns 1:1
    Path(OUT_SRT).parent.mkdir(parents=True, exist_ok=True)
    with open(TMP_LINES, "w", encoding="utf-8") as f:
        for _, _, cue_text in subs:
            f.write(cue_text.strip() + "\n")

    # 2) Use your existing translate_texts() on the temp file
    trans = translate_texts(TMP_LINES)  # list[str], same length as subs

    # 3) Align by index
    n = min(len(subs), len(trans))
    aligned = []
    segs = list_segments(SEGMENTS) if SEGMENTS else []

    for i in range(n):
        start_ms, end_ms, _ = subs[i]
        text = trans[i]
        if segs and i < len(segs):
            _ = stretch_to_window(segs[i], end_ms - start_ms, MAX_RATIO)  # optional
        aligned.append((start_ms, end_ms, text))

    write_srt(OUT_SRT, aligned)
    print(f"wrote {OUT_SRT} (cues: {len(aligned)})")

