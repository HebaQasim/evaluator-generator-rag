"""Part 1 (helper): speech to text with Faster-Whisper, for audio sources (WAV files).
The same approach as the voice data analysis project: the model runs on this computer, with no API key."""
import io
import logging

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

MODEL_SIZE = "small"   # multilingual: English and Arabic; a good balance between quality and speed on a CPU

model = None   # loaded the first time it is needed


def get_model():
    """Load the model once: the first time it is downloaded (about 500 MB), then it is read from the disk."""
    global model
    if model is None:
        logger.info("Loading the Whisper %s model", MODEL_SIZE)
        model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return model


def transcribe(data, language=None):
    """Turn the bytes of an audio file into text. language: "en", "ar" or None to let Whisper detect it."""
    # Example: a recording of a meeting note returns {"text": "Starting next month ...", "language": "en"}
    segments, info = get_model().transcribe(io.BytesIO(data), language=language, vad_filter=True)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise ValueError(f"No speech was found in the recording ({info.duration:.1f} seconds long).")
    return {"text": text, "language": info.language}
