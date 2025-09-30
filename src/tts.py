from pathlib import Path
from TTS.api import TTS

def text_to_speech(
    tts, 
    text, output_path,
    reference_wav=None,
    language="de",
):
    
    """
    Synthesize speech from text using a TTS model and save to output_path.
    Optionally uses a reference wav for voice cloning and sets the language.
    """
    out_path = Path(output_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(text=text, file_path=str(out_path),
                    speaker_wav=reference_wav, language=language)
    return out_path
