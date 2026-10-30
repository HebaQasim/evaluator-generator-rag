"""Speech to text with Faster-Whisper, for audio sources (WAV files).
The same approach as the voice data analysis project: the model runs on this computer, with no API key."""
import io
import logging

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class SpeechToText:
    """Turns the bytes of a recording into text. The Whisper model is loaded the first time it is needed, only once."""

    def __init__(self, model_size="small"):
        self.model_size = model_size   # "small": English and Arabic, a good balance between quality and speed on a CPU
        self.model = None              # not loaded yet

    def _load_model(self):
        """The first time, the model is downloaded (about 500 MB). After that it is read from the disk."""
        if self.model is None:
            logger.info("Loading the Whisper %s model", self.model_size)
            self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return self.model

    def transcribe(self, data, language=None):
        """data: the bytes of an audio file. language: "en", "ar" or None to let Whisper detect it.
        Example: a recording of a meeting note -> {"text": "Starting next month ...", "language": "en"}"""
        segments, info = self._load_model().transcribe(io.BytesIO(data), language=language, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        if not text:
            raise ValueError(f"No speech was found in the recording ({info.duration:.1f} seconds long).")
        return {"text": text, "language": info.language}


if __name__ == "__main__":
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    speech = SpeechToText()   # one object: the model is not loaded yet
    print("Model loaded at the start?", speech.model is not None)

    recording = Path("sample_sources/meeting_note.wav").read_bytes()
    print(speech.transcribe(recording))   # the first call loads the model
    print("Model loaded after the first call?", speech.model is not None)

    print(speech.transcribe(recording, language="en"))   # the second call uses the same model (no "Loading" line)
