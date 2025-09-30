from pathlib import Path
import numpy as np, librosa, soundfile as sf, subprocess
from TTS.api import TTS

from src.translate import extract_ref, translate_texts
from src.align import read_srt, write_translated_srt, build_timeline, load_mono
from src.tts import text_to_speech



# ---------- main pipeline ----------
def main():
    # --- Setup paths ---
    orig_video = Path("samples/Tanzania.mp4")
    orig_srt   = Path("samples/Tanzania-caption.srt")
    out_dir    = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)
    tts_dir    = out_dir / "tts"; tts_dir.mkdir(parents=True, exist_ok=True)

    # --- 0) Reference voice extraction ---
    ref_voice = extract_ref(orig_video, start=13.0, duration=9.0)

    # --- 1) Read cues and translate ---
    cues = read_srt(orig_srt)                   # list of (start_ms, end_ms, text_en)
    german_texts = translate_texts(orig_srt)    # list of de strings aligned to cues

    # --- 2) Initialize TTS model ---
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=True)


    # ========================================================================
    # === SPLIT WAV WORKFLOW: Synthesize each cue separately =================
    # ========================================================================
    print("Synthesizing audio per cue...")
    generated = []
    sr_final = None
    for i, ((start_ms, end_ms, _), text_de) in enumerate(zip(cues, german_texts), 1):
        target_ms = max(1, end_ms - start_ms)
        seg_path = tts_dir / f"seg_raw_{i:03d}.wav"
        text_to_speech(tts, text_de, seg_path, reference_wav=ref_voice, language="de")

        y, sr = load_mono(seg_path, sr_target=sr_final)
        if sr_final is None:
            sr_final = sr  # lock SR based on first synth

        sf.write(str(tts_dir / f"seg_final_{i:03d}.wav"), y, sr_final)

        generated.append({
            'start_ms': start_ms,
            'end_ms':   end_ms,
            'audio_np': y.astype(np.float32)
        })

    # Build timeline and save split audio
    total_ms = max(end for (_, end, _) in cues) + 200  # extra tail
    y_full_split = build_timeline(generated, sr_final, total_ms)
    final_wav_split = out_dir / "new_audio.wav"
    sf.write(str(final_wav_split), y_full_split, sr_final)

    # Mux split audio to video
    out_video_split = out_dir / "video_with_new_audio.mp4"
    cmd_split = [
        "ffmpeg", "-y",
        "-i", str(orig_video),
        "-i", str(final_wav_split),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(out_video_split)
    ]
    subprocess.run(cmd_split, check=True)
    print(f"wrote {out_video_split}")

    out_srt = out_dir / "translated.srt"
    write_translated_srt(out_srt, cues, german_texts)
    print(f"Wrote translated SRT to {out_srt}")


if __name__ == "__main__":
    main()