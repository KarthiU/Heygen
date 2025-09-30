
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer

import torch 
import subprocess

from src.align import read_srt  

# Extracts a short reference audio clip from an MP4 file
# and saves it as a mono WAV in the same directory.

def extract_ref(path, start=0.0, duration=12.0):

    path = Path(path)
    out_wav = path.with_name(path.stem + "-ref.wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(path),
        "-ss", str(start),
        "-t", str(duration),
        "-ar", "22050",   # resample for XTTS (22.05 kHz is safe)
        "-ac", "1",       # mono
        "-vn",            # no video
        str(out_wav)
    ]
    subprocess.run(cmd, check=True)
    print(f"Extracted reference wav: {out_wav}")

    return out_wav


def translate_texts(
    srt_path,
    model_name="Helsinki-NLP/opus-mt-en-de"
):
    """
    Translate each SRT cue text into German.
    Returns list of translated strings (1 per cue).
    """
    cues = read_srt(srt_path)  # [(start_ms, end_ms, text)]
    texts = [text for _, _, text in cues]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print (f"Using device: {device}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # <-- Add this line

    
    # IMPORTANT: pass attention_mask explicitly
    translated = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    outputs = tokenizer.batch_decode(translated, skip_special_tokens=True)

    return outputs
