from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from kavach.audio.streamer import AudioStreamer, AudioStreamError
from kavach.engines.intent_engine import (
    IntentAnalysisError,
    IntentEngineInitializationError,
    ScamIntentEngine,
)
from kavach.features.extractor import AudioFeatureExtractor, FeatureExtractionError
from kavach.models.classifier import (
    DeepfakeDetector,
    InferenceError,
    ModelInitializationError,
)


class Ansi:
    """ANSI color/style codes for highly visible terminal output."""

    RESET: str = "\033[0m"
    BOLD: str = "\033[1m"
    DIM: str = "\033[2m"
    RED: str = "\033[31m"
    GREEN: str = "\033[32m"
    YELLOW: str = "\033[33m"
    CYAN: str = "\033[36m"
    MAGENTA: str = "\033[35m"
    WHITE: str = "\033[37m"


@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration for end-to-end on-device real-time inference."""

    sample_rate: int = 16000
    window_seconds: float = 2.0
    chunk_size: int = 1024
    silence_threshold: float = 0.01
    deepfake_threshold: float = 0.70
    scam_threshold: float = 0.60
    max_workers: int = 2


class KavachOrchestrator:
    """Coordinates audio capture, deepfake detection, and scam intent analysis."""

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self.config: OrchestratorConfig = config or OrchestratorConfig()
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self.streamer: AudioStreamer = AudioStreamer(
            sample_rate=self.config.sample_rate,
            channels=1,
            chunk_size=self.config.chunk_size,
        )
        self.extractor: AudioFeatureExtractor = AudioFeatureExtractor(
            sample_rate=self.config.sample_rate,
            window_seconds=self.config.window_seconds,
            n_mfcc=40,
        )
        self.detector: DeepfakeDetector = DeepfakeDetector(
            n_mfcc=40,
            num_frames=self.extractor.expected_frames,
        )
        self.intent_engine: ScamIntentEngine = ScamIntentEngine(
            sample_rate=self.config.sample_rate,
            stt_backend="whisper",
            whisper_model_name="tiny",
            force_english_translation=False,
            language_hint=None,
        )

        self._stop_event: threading.Event = threading.Event()
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="KavachWorker",
        )

        self._window_bytes: int = int(
            self.config.sample_rate * self.config.window_seconds * 2
        )

    def run(self) -> None:
        """Start capture and continuously analyze each consolidated 2-second audio window."""
        self._register_signal_handlers()

        try:
            self.streamer.start_streaming()
            print(f"{Ansi.CYAN}{Ansi.BOLD}[LIVE] Capturing Audio...{Ansi.RESET}")
            self._consume_and_analyze()
        except AudioStreamError as exc:
            self._logger.error("Audio stream startup failed: %s", exc)
            print(
                f"{Ansi.RED}{Ansi.BOLD}[ERROR] Audio stream startup failed: {exc}{Ansi.RESET}"
            )
        except KeyboardInterrupt:
            print(
                f"\n{Ansi.YELLOW}{Ansi.BOLD}[SHUTDOWN] Keyboard interrupt received.{Ansi.RESET}"
            )
        except Exception as exc:
            self._logger.exception("Fatal orchestrator failure: %s", exc)
            print(f"{Ansi.RED}{Ansi.BOLD}[FATAL] {exc}{Ansi.RESET}")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Release threads and audio resources without leaks."""
        self._stop_event.set()
        try:
            self.streamer.stop_streaming()
        except Exception as exc:
            self._logger.error("Error while stopping audio streamer: %s", exc)

        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
        except Exception as exc:
            self._logger.error("Error while shutting down worker pool: %s", exc)

        print(f"{Ansi.DIM}[INFO] Resources released. Exiting safely.{Ansi.RESET}")

    def _consume_and_analyze(self) -> None:
        """Aggregate queue frames into fixed windows and run detectors concurrently."""
        buffer: bytearray = bytearray()

        while not self._stop_event.is_set():
            try:
                frame: bytes = self.streamer.audio_queue.get(timeout=0.25)
                buffer.extend(frame)
            except queue.Empty:
                continue

            while len(buffer) >= self._window_bytes and not self._stop_event.is_set():
                window_bytes: bytes = bytes(buffer[: self._window_bytes])
                del buffer[: self._window_bytes]
                self._analyze_window(window_bytes)

    def _analyze_window(self, window_bytes: bytes) -> None:
        """Dispatch deepfake and scam pipelines concurrently and print decision logs."""
        audio_array: np.ndarray = self._bytes_to_audio_array(window_bytes)
        rms_energy: float = self._compute_rms_energy(audio_array)

        if rms_energy < self.config.silence_threshold:
            print(f"{Ansi.DIM}[VAD] Silence detected, skipping...{Ansi.RESET}")
            print(f"{Ansi.DIM}{'-' * 72}{Ansi.RESET}")
            return

        deepfake_future: Future[float] = self._executor.submit(
            self._deepfake_pipeline,
            window_bytes,
        )
        scam_future: Future[tuple[str, float]] = self._executor.submit(
            self.intent_engine.analyze_intent,
            window_bytes,
        )

        deepfake_score: Optional[float] = None
        transcript: str = ""
        scam_score: Optional[float] = None

        try:
            deepfake_score = deepfake_future.result(timeout=8.0)
        except Exception as exc:
            self._logger.error("Deepfake pipeline error: %s", exc)

        try:
            transcript, scam_score = scam_future.result(timeout=12.0)
        except Exception as exc:
            self._logger.error("Scam intent pipeline error: %s", exc)

        self._print_analysis(deepfake_score, transcript, scam_score)

    def _deepfake_pipeline(self, window_bytes: bytes) -> float:
        """Extract acoustic features then run classifier inference."""
        try:
            features = self.extractor.extract_features(window_bytes)
            score: float = self.detector.predict(features)
            return score
        except (FeatureExtractionError, InferenceError) as exc:
            raise RuntimeError(f"Deepfake analysis failed: {exc}") from exc

    def _print_analysis(
        self,
        deepfake_score: Optional[float],
        transcript: str,
        scam_score: Optional[float],
    ) -> None:
        """Render formatted live status and threshold-triggered alerts."""
        if deepfake_score is None:
            deepfake_line: str = (
                f"{Ansi.YELLOW}[DEEPFAKE DETECTOR] Score: unavailable{Ansi.RESET}"
            )
        else:
            deepfake_pct: float = deepfake_score * 100.0
            deepfake_color: str = (
                Ansi.RED if deepfake_score >= self.config.deepfake_threshold else Ansi.GREEN
            )
            deepfake_line = (
                f"{deepfake_color}[DEEPFAKE DETECTOR] Score: {deepfake_pct:.2f}%{Ansi.RESET}"
            )

        safe_transcript: str = self._sanitize_transcript(transcript)
        transcript_line: str = (
            f"{Ansi.MAGENTA}[TRANSLATION] "
            f'"{self._truncate_for_log(safe_transcript, max_len=180)}"{Ansi.RESET}'
        )

        if scam_score is None:
            scam_line: str = f"{Ansi.YELLOW}[SCAM INTENT] Threat Level: UNKNOWN{Ansi.RESET}"
        else:
            level: str = "HIGH" if scam_score >= self.config.scam_threshold else "LOW"
            scam_color: str = Ansi.RED if level == "HIGH" else Ansi.GREEN
            scam_line = (
                f"{scam_color}[SCAM INTENT] Threat Level: {level} "
                f"({scam_score * 100.0:.2f}%){Ansi.RESET}"
            )

        print(deepfake_line)
        print(transcript_line)
        print(scam_line)

        deepfake_alert: bool = (
            deepfake_score is not None
            and deepfake_score >= self.config.deepfake_threshold
        )
        scam_alert: bool = scam_score is not None and scam_score >= self.config.scam_threshold

        if deepfake_alert or scam_alert:
            print(
                f"{Ansi.RED}{Ansi.BOLD}"
                "🚨 CRITICAL ALERT: POTENTIAL VOICE SCAM DETECTED! 🚨"
                f"{Ansi.RESET}"
            )

        print(f"{Ansi.DIM}{'-' * 72}{Ansi.RESET}")

    def _bytes_to_audio_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert raw PCM16 audio bytes to a normalized float32 NumPy array."""
        if len(audio_bytes) == 0:
            return np.zeros(0, dtype=np.float32)

        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]

        int16_audio: np.ndarray = np.frombuffer(audio_bytes, dtype=np.int16)
        if int16_audio.size == 0:
            return np.zeros(0, dtype=np.float32)

        audio_array: np.ndarray = int16_audio.astype(np.float32) / 32768.0
        return np.clip(audio_array, -1.0, 1.0)

    @staticmethod
    def _compute_rms_energy(audio_array: np.ndarray) -> float:
        """Compute RMS energy for VAD gating."""
        if audio_array.size == 0:
            return 0.0

        mean_square: float = float(np.mean(np.square(audio_array, dtype=np.float32)))
        return float(np.sqrt(mean_square))

    def _register_signal_handlers(self) -> None:
        """Map OS termination signals to graceful stop behavior."""
        def _handle_signal(sig: int, _frame: object) -> None:
            self._logger.info("Received signal %s; initiating shutdown.", sig)
            self._stop_event.set()

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except Exception as exc:
            self._logger.warning("Signal handler registration skipped: %s", exc)

    @staticmethod
    def _sanitize_transcript(text: str) -> str:
        """Remove control characters and normalize spacing for clean terminal output."""
        clean: str = " ".join(text.replace("\n", " ").replace("\r", " ").split())
        return clean

    @staticmethod
    def _truncate_for_log(text: str, max_len: int) -> str:
        """Bound transcript length to keep live logs readable in terminal streams."""
        if len(text) <= max_len:
            return text
        return f"{text[: max_len - 3]}..."


def _configure_logging() -> None:
    """Set process-wide logging format for observability during live runtime."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _bootstrap_runtime_checks() -> None:
    """Run startup checks that can fail before main processing starts."""
    try:
        _ = OrchestratorConfig()
    except Exception as exc:
        raise RuntimeError(f"Orchestrator configuration validation failed: {exc}") from exc


if __name__ == "__main__":
    _configure_logging()

    try:
        _bootstrap_runtime_checks()
        orchestrator: KavachOrchestrator = KavachOrchestrator()
        orchestrator.run()
    except (ModelInitializationError, IntentEngineInitializationError) as exc:
        print(f"{Ansi.RED}{Ansi.BOLD}[INIT ERROR] {exc}{Ansi.RESET}")
    except Exception as exc:
        print(f"{Ansi.RED}{Ansi.BOLD}[UNRECOVERABLE ERROR] {exc}{Ansi.RESET}")
