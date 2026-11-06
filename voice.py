"""Step 4: Gnani speech-to-text (ears) and text-to-speech (voice). Gnani only, no other speech service."""
import os
import wave
from pathlib import Path

from . import config
from .agent import normalize_asr

TTS_LANG = {"en": "en-IN", "hi": "hi-IN", "hi-en": "hi-en", "bn": "bn-IN", "gu": "gu-IN", "kn": "kn-IN",
            "ml": "ml-IN", "mr": "mr-IN", "pa": "pa-IN", "ta": "ta-IN", "te": "te-IN"}
UNCLEAR = "Sorry, I didn't catch that. Could you say it again?"
MAX_SECONDS = 55                      # REST STT accepts clips up to 60 s


class Voice:
    def __init__(self):
        config.load_env()
        if not config.env("GNANI_API_KEY"):
            raise RuntimeError("GNANI_API_KEY is missing in .env")
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient
        self.stt = GnaniSTTClient()                  # reads GNANI_API_KEY from the environment
        self.tts = GnaniTTSClient()
        self.stt_lang = config.env("STT_LANGUAGE", "en-IN")
        self.model = config.env("TTS_MODEL", "timbre-v2.5")
        self.voice = config.env("TTS_VOICE", "Kaveri")

    # ---------- listen ----------
    def _transcribe_one(self, path, lang):
        res = self.stt.transcribe(str(path), language_code=lang)
        if isinstance(res, dict):
            return (res.get("transcript") or "").strip()
        return (getattr(res, "transcript", "") or "").strip()

    def listen(self, audio_path, lang=None):
        """Audio file -> text. Long WAV files are cut into <=55 s pieces automatically."""
        lang = lang or self.stt_lang
        path = Path(audio_path)
        try:
            if path.suffix.lower() == ".wav":
                parts = self._split_wav(path)
                if len(parts) > 1:
                    text = " ".join(self._transcribe_one(p, lang) for p in parts)
                    for p in parts:
                        p.unlink(missing_ok=True)
                    return normalize_asr(text.strip())
            return normalize_asr(self._transcribe_one(path, lang))
        except Exception as e:
            print(f"   [STT problem: {type(e).__name__}: {e}]")
            return ""

    @staticmethod
    def _split_wav(path):
        with wave.open(str(path), "rb") as w:
            rate, frames, params = w.getframerate(), w.getnframes(), w.getparams()
            if frames / rate <= MAX_SECONDS:
                return [path]
            step, parts, i = rate * MAX_SECONDS, [], 0
            while True:
                data = w.readframes(step)
                if not data:
                    break
                p = path.with_name(f"{path.stem}.part{i}.wav")
                with wave.open(str(p), "wb") as out:
                    out.setparams(params)
                    out.writeframes(data)
                parts.append(p)
                i += 1
            return parts

    # ---------- speak ----------
    def speak(self, text, out_file, language="en"):
        lang = TTS_LANG.get(language, language if "-" in language else "en-IN")
        self.tts.synthesize(text, voice=self.voice, model=self.model, language=lang, output_file=str(out_file))
        return out_file

    # ---------- one full turn: audio in -> (heard, reply, turn) -> audio out ----------
    def voice_turn(self, agent, audio_path, out_wav, lang=None):
        heard = self.listen(audio_path, lang)
        if not heard.strip():                         # unclear / silent audio: recover politely
            from .agent import Turn
            turn = Turn(reply=UNCLEAR, decision="clarify", language="en")
        else:
            turn = agent.respond(heard)
        turn.heard = heard
        try:
            self.speak(turn.reply, out_wav, turn.language)
        except Exception as e:
            print(f"   [TTS problem: {type(e).__name__}: {e}]")
        return turn


def record_wav(path, seconds=8, rate=16000):
    """Record from the microphone (needs: pip install sounddevice numpy)."""
    import sounddevice as sd
    data = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="int16")
    sd.wait()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())
    return path


def play(path):
    """Play a wav file with the operating system's default player."""
    try:
        if os.name == "nt":
            os.startfile(str(path))                   # noqa: S606  (Windows)
        else:
            os.system(f'xdg-open "{path}" >/dev/null 2>&1 &')
    except Exception:
        print(f"   (audio saved at {path})")
