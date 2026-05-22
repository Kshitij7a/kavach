from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import whisper
from transformers import pipeline
from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)


class IntentEngineInitializationError(Exception):
    """Raised when local STT models cannot be initialized."""


class IntentAnalysisError(Exception):
    """Raised when transcription or scam intent analysis fails."""


@dataclass(frozen=True)
class RiskIndicator:
    """Weighted scam indicator rule for lexical pattern matching."""

    pattern: re.Pattern[str]
    weight: float


class ScamIntentEngine:
    """Local-only transcription and scam intent scoring engine for live call audio."""

    def __init__(
        self,
        sample_rate: int = 16000,
        stt_backend: str = "whisper",
        whisper_model_name: str = "tiny",
        hf_model_name: str = "openai/whisper-tiny",
        force_english_translation: bool = False,
        language_hint: Optional[str] = None,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.stt_backend: str = stt_backend.strip().lower()
        self.force_english_translation: bool = force_english_translation
        self.language_hint: Optional[str] = language_hint
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self._whisper_model: Optional[whisper.Whisper] = None
        self._hf_asr_pipeline: Optional[AutomaticSpeechRecognitionPipeline] = None

        self._risk_indicators: list[RiskIndicator] = self._build_risk_indicators()
        self._context_patterns: list[re.Pattern[str]] = self._build_context_patterns()

        self._initialize_models(
            whisper_model_name=whisper_model_name,
            hf_model_name=hf_model_name,
        )

    def analyze_intent(self, audio_bytes: bytes) -> tuple[str, float]:
        """Transcribe input audio and return (transcribed_text, scam_threat_score)."""
        try:
            waveform: np.ndarray = self._decode_audio_bytes(audio_bytes)
            transcription: str = self._transcribe_waveform(waveform)
            score: float = self._score_scam_risk(transcription)
            return transcription, score
        except IntentAnalysisError:
            raise
        except Exception as exc:
            self._logger.exception("Unexpected intent analysis failure: %s", exc)
            raise IntentAnalysisError("Unexpected failure during scam intent analysis.") from exc

    def _initialize_models(self, whisper_model_name: str, hf_model_name: str) -> None:
        """Initialize configured local STT backend and validate inference capability."""
        try:
            if self.stt_backend == "whisper":
                self._whisper_model = whisper.load_model(whisper_model_name)
                return

            if self.stt_backend == "transformers":
                self._hf_asr_pipeline = pipeline(
                    task="automatic-speech-recognition",
                    model=hf_model_name,
                    device=-1,
                )
                return

            raise IntentEngineInitializationError(
                "Invalid STT backend. Use 'whisper' or 'transformers'."
            )
        except IntentEngineInitializationError:
            raise
        except Exception as exc:
            raise IntentEngineInitializationError(
                f"Failed to initialize local STT backend '{self.stt_backend}': {exc}"
            ) from exc

    def _decode_audio_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """Decode PCM16 mono bytes into a normalized float32 waveform."""
        if len(audio_bytes) == 0:
            raise IntentAnalysisError("Incoming audio chunk is empty.")

        if len(audio_bytes) % 2 != 0:
            raise IntentAnalysisError(
                "Incoming audio chunk is corrupted: PCM16 byte length is misaligned."
            )

        int16_audio: np.ndarray = np.frombuffer(audio_bytes, dtype=np.int16)
        if int16_audio.size == 0:
            raise IntentAnalysisError("Decoded audio contains zero samples.")

        waveform: np.ndarray = int16_audio.astype(np.float32) / 32768.0
        if not np.all(np.isfinite(waveform)):
            raise IntentAnalysisError("Decoded waveform contains non-finite values.")

        peak: float = float(np.max(np.abs(waveform)))
        if peak > 1.0:
            waveform = waveform / peak

        return waveform

    def _transcribe_waveform(self, waveform: np.ndarray) -> str:
        """Run local STT inference and return cleaned transcription text."""
        if self.stt_backend == "whisper":
            if self._whisper_model is None:
                raise IntentAnalysisError("Whisper model is not initialized.")
            return self._transcribe_with_whisper(waveform)

        if self.stt_backend == "transformers":
            if self._hf_asr_pipeline is None:
                raise IntentAnalysisError("Transformers ASR pipeline is not initialized.")
            return self._transcribe_with_transformers(waveform)

        raise IntentAnalysisError("Unsupported STT backend at inference time.")

    def _transcribe_with_whisper(self, waveform: np.ndarray) -> str:
        """Transcribe with local openai-whisper model."""
        if self._whisper_model is None:
            raise IntentAnalysisError("Whisper model is not initialized.")

        options: dict[str, object] = {
            "fp16": False,
            "task": "translate" if self.force_english_translation else "transcribe",
        }
        if self.language_hint is not None and self.language_hint.strip() != "":
            options["language"] = self.language_hint.strip()

        try:
            result: dict[str, object] = self._whisper_model.transcribe(
                audio=waveform,
                **options,
            )
        except Exception as exc:
            raise IntentAnalysisError(f"Local Whisper transcription failed: {exc}") from exc

        text_obj: object = result.get("text", "")
        text: str = str(text_obj).strip()
        if text == "":
            raise IntentAnalysisError("Whisper produced empty transcription.")
        return text

    def _transcribe_with_transformers(self, waveform: np.ndarray) -> str:
        """Transcribe with local Hugging Face ASR pipeline."""
        if self._hf_asr_pipeline is None:
            raise IntentAnalysisError("Transformers ASR pipeline is not initialized.")

        payload: dict[str, object] = {
            "array": waveform,
            "sampling_rate": self.sample_rate,
        }

        generate_kwargs: dict[str, object] = {}
        if self.language_hint is not None and self.language_hint.strip() != "":
            generate_kwargs["language"] = self.language_hint.strip()
        if self.force_english_translation:
            generate_kwargs["task"] = "translate"

        try:
            if len(generate_kwargs) > 0:
                output: dict[str, object] = self._hf_asr_pipeline(
                    payload,
                    generate_kwargs=generate_kwargs,
                )
            else:
                output = self._hf_asr_pipeline(payload)
        except Exception as exc:
            raise IntentAnalysisError(f"Local transformers ASR failed: {exc}") from exc

        text_obj: object = output.get("text", "")
        text: str = str(text_obj).strip()
        if text == "":
            raise IntentAnalysisError("Transformers ASR produced empty transcription.")
        return text

    def _score_scam_risk(self, transcription: str) -> float:
        """Compute risk score in [0.0, 1.0] from lexical hits, density, and context boosts."""
        normalized_text: str = self._normalize_text(transcription)
        if normalized_text == "":
            return 0.0

        token_count: int = self._token_count(normalized_text)
        total_weight: float = 0.0
        matched_patterns: int = 0

        for indicator in self._risk_indicators:
            occurrences: int = len(indicator.pattern.findall(normalized_text))
            if occurrences > 0:
                total_weight += indicator.weight * float(occurrences)
                matched_patterns += 1

        density_component: float = min(1.0, total_weight / max(1.0, token_count * 0.8))
        breadth_component: float = min(1.0, matched_patterns / 8.0)

        context_hits: int = 0
        for context_rule in self._context_patterns:
            if context_rule.search(normalized_text) is not None:
                context_hits += 1

        context_component: float = min(1.0, 0.2 * float(context_hits))

        raw_score: float = (
            (0.45 * density_component)
            + (0.35 * breadth_component)
            + (0.20 * context_component)
        )

        if raw_score < 0.0:
            return 0.0
        if raw_score > 1.0:
            return 1.0
        return float(raw_score)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for robust multilingual keyword matching."""
        lowered: str = text.lower().strip()
        lowered = re.sub(r"\s+", " ", lowered)
        return lowered

    @staticmethod
    def _token_count(text: str) -> int:
        """Count Latin and Devanagari word-like tokens."""
        tokens: list[str] = re.findall(r"[a-z0-9_\u0900-\u097f]+", text)
        return len(tokens)

    @staticmethod
    def _build_risk_indicators() -> list[RiskIndicator]:
        """Construct weighted indicator patterns for scam detection."""
        indicators: list[tuple[str, float]] = [
            (r"\bupi\b", 1.2),
            (r"\botp\b", 1.4),
            (r"\bimmediate transfer\b", 1.6),
            (r"\btransfer now\b", 1.4),
            (r"\baccount blocked\b", 1.6),
            (r"\bkyc\b", 1.0),
            (r"\bpolice arrest\b", 1.8),
            (r"\barrest warrant\b", 1.8),
            (r"\blottery\b", 1.5),
            (r"\bkaun banega crorepati\b", 1.9),
            (r"\bprize money\b", 1.4),
            (r"\burgent\b", 1.1),
            (r"\bimmediately\b", 1.1),
            (r"\bverify now\b", 1.2),
            (r"\bbank account\b", 1.2),
            (r"\bcredit card\b", 1.0),
            (r"\bdebit card\b", 1.0),
            (r"\bshare your screen\b", 1.7),
            (r"\bremote access\b", 1.7),
            (r"\bdownload app\b", 1.2),
            (r"\bblock ho jayega\b", 1.5),
            (r"\baccount block\b", 1.4),
            (r"\bturant\b", 1.0),
            (r"\bjaldi\b", 1.0),
            (r"\bpolice case\b", 1.7),
            (r"\bpayment request\b", 1.3),
            (r"\bqr code\b", 1.1),
            (r"\bone time password\b", 1.5),
        ]

        compiled: list[RiskIndicator] = []
        for pattern, weight in indicators:
            compiled.append(RiskIndicator(pattern=re.compile(pattern), weight=weight))
        return compiled

    @staticmethod
    def _build_context_patterns() -> list[re.Pattern[str]]:
        """Construct context rules indicating coercive and financial scam intent."""
        rules: list[str] = [
            r"(urgent|immediately|turant|jaldi).*(transfer|payment|upi|otp)",
            r"(police|arrest|case).*(pay|transfer|upi|fine)",
            r"(lottery|prize|crorepati).*(fee|tax|processing|transfer)",
            r"(account blocked|account block|kyc).*(verify|otp|transfer)",
            r"(remote access|share your screen|download app).*(bank|account|payment)",
        ]
        return [re.compile(rule) for rule in rules]
