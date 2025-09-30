from pathlib import Path
import os
from TTS.api import TTS


# Model (XTTS v2 = multilingual + voice cloning)
model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name=model_name, progress_bar=False, gpu=True)  # set gpu=True if you installed CUDA wheels

# Text & output path
text = "Tansania – Heimat einiger der atemberaubendsten Tierwelten der Erde.", "Hier, im Herzen Ostafrikas, befindet sich der große Serengeti-Nationalpark"
out_path = Path("outputs/test_xtts.wav")
out_path.parent.mkdir(parents=True, exist_ok=True)

# Reference voice (use an absolute path; or set to None to disable cloning)
ref = "samples/Tanzania-ref.wav"


# (Tip) 6–15s of clean, single-speaker speech works best for cloning
for i, sent in enumerate(text, 1):
    tts.tts_to_file(
        text=sent,
        file_path=f"outputs/seg_{i}.wav",
        speaker_wav=ref,
        language="de",
    )
    
print(f"Generated {out_path} (cloning: {ref is not None})")
