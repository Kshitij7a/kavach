from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

import pyaudio


class AudioStreamError(Exception):
    """Raised when the audio stream cannot be started or operated safely."""


class AudioStreamer:
    """Capture microphone audio in real time and buffer raw frames in a thread-safe queue."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        queue_maxsize: int = 512,
        device_index: Optional[int] = None,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.channels: int = channels
        self.chunk_size: int = chunk_size
        self.device_index: Optional[int] = device_index

        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=queue_maxsize)

        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._record_thread: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._start_stop_lock: threading.Lock = threading.Lock()
        self._overrun_count: int = 0

    @property
    def is_running(self) -> bool:
        """Return True when the background recording loop is active."""
        return (
            self._record_thread is not None
            and self._record_thread.is_alive()
            and not self._stop_event.is_set()
        )

    @property
    def overrun_count(self) -> int:
        """Return total number of input overrun-like read failures observed."""
        return self._overrun_count

    def start_streaming(self) -> None:
        """Start microphone capture and the background recording thread."""
        with self._start_stop_lock:
            if self.is_running:
                self._logger.warning("Audio streamer is already running.")
                return

            self._stop_event.clear()
            self._initialize_stream()

            self._record_thread = threading.Thread(
                target=self._record_loop,
                name="AudioStreamerThread",
                daemon=True,
            )
            self._record_thread.start()
            self._logger.info(
                "Audio streaming started: sample_rate=%s, channels=%s, chunk_size=%s, device_index=%s",
                self.sample_rate,
                self.channels,
                self.chunk_size,
                self.device_index,
            )

    def stop_streaming(self) -> None:
        """Stop recording safely and release all audio resources."""
        with self._start_stop_lock:
            self._stop_event.set()

            if self._record_thread is not None and self._record_thread.is_alive():
                self._record_thread.join(timeout=2.0)
                if self._record_thread.is_alive():
                    self._logger.warning("Recording thread did not stop within timeout.")

            self._record_thread = None
            self._close_stream()
            self._logger.info("Audio streaming stopped.")

    def _initialize_stream(self) -> None:
        """Initialize PyAudio and open the input stream with robust error mapping."""
        try:
            self._pa = pyaudio.PyAudio()
        except Exception as exc:
            raise AudioStreamError("Failed to initialize PyAudio.") from exc

        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
            )
            self._stream.start_stream()
        except OSError as exc:
            self._close_stream()
            message: str = str(exc)
            lowered: str = message.lower()
            if "unanticipated host error" in lowered or "permission" in lowered:
                raise AudioStreamError(
                    "Microphone permission denied or blocked by the host audio backend."
                ) from exc
            if "invalid input device" in lowered or "no default input device" in lowered:
                raise AudioStreamError(
                    "No valid microphone input device found. Verify microphone availability."
                ) from exc
            raise AudioStreamError(f"Failed to open microphone stream: {message}") from exc
        except Exception as exc:
            self._close_stream()
            raise AudioStreamError("Unexpected error while opening microphone stream.") from exc

    def _close_stream(self) -> None:
        """Close stream and PyAudio instances safely without leaking resources."""
        if self._stream is not None:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
            except Exception as exc:
                self._logger.error("Failed to stop stream cleanly: %s", exc)
            finally:
                try:
                    self._stream.close()
                except Exception as exc:
                    self._logger.error("Failed to close stream cleanly: %s", exc)
                self._stream = None

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception as exc:
                self._logger.error("Failed to terminate PyAudio cleanly: %s", exc)
            finally:
                self._pa = None

    def _record_loop(self) -> None:
        """Read microphone bytes continuously and enqueue frames for downstream processing."""
        while not self._stop_event.is_set():
            if self._stream is None:
                self._logger.error("Audio stream is not available in recording loop.")
                self._stop_event.set()
                break

            try:
                frame: bytes = self._stream.read(
                    self.chunk_size,
                    exception_on_overflow=False,
                )
                self._enqueue_frame(frame)
            except OSError as exc:
                self._overrun_count += 1
                self._logger.warning("Audio read overrun/device error detected: %s", exc)
                time.sleep(0.02)
            except Exception as exc:
                self._logger.exception("Unexpected failure while reading audio stream: %s", exc)
                self._stop_event.set()
                break

    def _enqueue_frame(self, frame: bytes) -> None:
        """Enqueue an audio frame and prevent producer blocking under high load."""
        try:
            self.audio_queue.put_nowait(frame)
        except queue.Full:
            try:
                _ = self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(frame)
            except queue.Empty:
                self._logger.debug("Queue became empty while handling full condition.")
            except queue.Full:
                self._logger.debug("Queue remained full after dropping one frame.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    streamer: AudioStreamer = AudioStreamer()

    try:
        streamer.start_streaming()
        print("Streaming started. Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)
            qsize: int = streamer.audio_queue.qsize()
            if qsize > 0:
                print(
                    f"Audio queue size: {qsize} | Overrun count: {streamer.overrun_count}"
                )
    except KeyboardInterrupt:
        print("Stopping stream...")
    except AudioStreamError as exc:
        print(f"Audio streamer failed to start: {exc}")
    except Exception as exc:
        print(f"Unexpected runtime error: {exc}")
    finally:
        streamer.stop_streaming()
