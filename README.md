# Heygen: Offline Translation + Voice Cloning Pipeline

This project provides an offline workflow for translating subtitles from English to German and generating synchronized German audio that mimics the original speaker’s voice.  
It is designed to take an `.srt` subtitle file and a reference audio clip (English speaker) and output a German-dubbed audio file aligned with the original video timing.

---

## Features
- Automatic Subtitle Translation – English → German via [MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-en-de).
- Voice Cloning & Speech Synthesis – German audio generated using [Coqui XTTS v2](https://github.com/coqui-ai/TTS).
- Synchronization – Audio stretched/truncated to match each subtitle cue’s duration.
- Offline-Friendly – No reliance on cloud APIs (translation and TTS both run locally).

---

## Setup

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Coqui TTS (if not already pulled in)
The [Coqui TTS API](https://github.com/coqui-ai/TTS) sometimes requires manual installation:
```bash
pip install TTS
```

GPU users should install CUDA wheels for faster inference. Example:
```bash
pip install TTS --extra-index-url https://download.pytorch.org/whl/cu121
```

### 4. Prepare resources
- **Reference audio**: A clean 6–15s clip of the original English speaker (`samples/Tanzania-ref.wav` provided).
- **Subtitles**: English `.srt` file with timestamps (`samples/Tanzania-caption.srt` provided).

---

## Usage

Run the main pipeline:

```bash
python main.py
```

### Inputs
- English subtitle file (`.srt`)
- Reference speaker audio (`.wav`)

### Outputs
- German audio segments (`outputs/seg_###.wav`)
- Final concatenated German audio track (`outputs/final.wav`)

---

## Design Decisions

### Why XTTS v2?
- Multilingual support – Handles German synthesis out-of-the-box.
- Voice cloning – Generates speech in German using an English speaker’s reference voice.
- Offline support – No need for cloud APIs, unlike ElevenLabs or Azure TTS.

### Why MarianMT?
- High-quality parallel corpora – Trained on OPUS datasets (English↔German is particularly strong).
- Offline availability – Runs with Hugging Face Transformers, no API keys required.
- Lightweight – Fast enough for processing whole subtitle files locally.

### Cue-Based Processing
- Each subtitle cue is translated and synthesized individually.
- Ensures synchronization with video timing, even if some segments sound slightly inconsistent.
- Audio is trimmed/resampled (`librosa`) to avoid silences and stretched to exactly match subtitle windows.

---

## Common Issues

### 1. Inconsistent voice across cues
This happens because XTTS re-estimates speaker embeddings per segment.  
Workaround: Use longer reference audio (clean speech) and apply cross-fading when concatenating.

### 2. Coqui TTS import not working
Make sure `TTS` is installed separately (see [Setup Step 3](#3-install-coqui-tts-if-not-already-pulled-in)).

### 3. CUDA not detected
Check PyTorch install:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, reinstall with GPU wheels from [PyTorch](https://pytorch.org/get-started/locally/).

---

## Example Run

```bash
python main.py
```

**Input:**
- `samples/Tanzania-caption.srt`
- `samples/Tanzania-ref.wav`

**Output:**
- `outputs/final.wav`

Plays German audio of the translated subtitles, spoken in the reference speaker’s cloned voice, aligned to the original timings.

---

## License
MIT License © 2025 [KarthiU](https://github.com/KarthiU)
