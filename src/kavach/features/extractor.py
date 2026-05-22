from __future__ import annotations

import logging
from typing import Final

import librosa
import numpy as np
from librosa.util.exceptions import ParameterError


class FeatureExtractionError(Exception):
    """Raised when audio features cannot be extracted from incoming bytes."""


class AudioFeatureExtractor:
    """Extract fixed-shape MFCC and delta features from raw PCM16 microphone bytes."""

    PCM16_DTYPE: Final[np.dtype[np.int16]] = np.dtype(np.int16)

    def __init__(
        self,
        sample_rate: int = 16000,
        window_seconds: float = 2.0,
        n_mfcc: int = 40,
        n_fft: int = 400,
        hop_length: int = 160,
        normalize: bool = True,
    ) -> None:
        self.sample_rate: int = sample_rate
        self.window_seconds: float = window_seconds
        self.n_mfcc: int = n_mfcc
        self.n_fft: int = n_fft
        self.hop_length: int = hop_length
        self.normalize: bool = normalize

        self.target_samples: int = int(round(self.sample_rate * self.window_seconds))
        self.expected_frames: int = self._compute_expected_frames(
            target_samples=self.target_samples,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

    def extract_features(self, audio_bytes: bytes) -> np.ndarray:
        """
        Convert PCM16 audio bytes to fixed-shape MFCC + delta features.

        Returns a float32 array of shape (2 * n_mfcc, expected_frames), ready for CNN input.
        """
        try:
            waveform: np.ndarray = self._bytes_to_waveform(audio_bytes)
            waveform = self._fit_waveform_length(waveform)

            mfcc: np.ndarray = librosa.feature.mfcc(
                y=waveform,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                center=False,
            )
            delta: np.ndarray = librosa.feature.delta(mfcc, order=1, axis=-1)

            features: np.ndarray = np.vstack((mfcc, delta)).astype(np.float32, copy=False)
            features = self._fit_feature_frames(features)

            if self.normalize:
                features = self._normalize_features(features)

            if not np.all(np.isfinite(features)):
                raise FeatureExtractionError("Extracted features contain non-finite values.")

            return features
        except FeatureExtractionError:
            raise
        except ParameterError as exc:
            raise FeatureExtractionError(f"Invalid librosa extraction parameters: {exc}") from exc
        except ValueError as exc:
            raise FeatureExtractionError(f"Audio bytes are corrupted or incompatible: {exc}") from exc
        except Exception as exc:
            self._logger.exception("Unexpected feature extraction failure: %s", exc)
            raise FeatureExtractionError("Unexpected failure during feature extraction.") from exc

    def _bytes_to_waveform(self, audio_bytes: bytes) -> np.ndarray:
        """Decode little-endian PCM16 bytes into normalized float32 waveform in [-1, 1]."""
        if len(audio_bytes) == 0:
            raise FeatureExtractionError("Incoming audio payload is empty.")

        itemsize: int = self.PCM16_DTYPE.itemsize
        if len(audio_bytes) % itemsize != 0:
            raise FeatureExtractionError(
                "Incoming audio payload length is misaligned for PCM16 decoding."
            )

        int16_audio: np.ndarray = np.frombuffer(audio_bytes, dtype=self.PCM16_DTYPE)
        if int16_audio.size == 0:
            raise FeatureExtractionError("Decoded audio payload has zero samples.")

        waveform: np.ndarray = int16_audio.astype(np.float32) / 32768.0
        if not np.all(np.isfinite(waveform)):
            raise FeatureExtractionError("Decoded waveform contains non-finite values.")

        return waveform

    def _fit_waveform_length(self, waveform: np.ndarray) -> np.ndarray:
        """Pad or truncate waveform to a deterministic 2-second (or configured) window."""
        sample_count: int = int(waveform.shape[0])

        if sample_count == self.target_samples:
            return waveform

        if sample_count < self.target_samples:
            pad_width: int = self.target_samples - sample_count
            return np.pad(waveform, (0, pad_width), mode="constant", constant_values=0.0)

        return waveform[: self.target_samples]

    def _fit_feature_frames(self, features: np.ndarray) -> np.ndarray:
        """Force a consistent time-frame axis for downstream model input."""
        current_frames: int = int(features.shape[1])

        if current_frames == self.expected_frames:
            return features

        if current_frames < self.expected_frames:
            pad_frames: int = self.expected_frames - current_frames
            return np.pad(
                features,
                ((0, 0), (0, pad_frames)),
                mode="constant",
                constant_values=0.0,
            )

        return features[:, : self.expected_frames]

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply per-feature z-normalization with numerical stability guards."""
        mean: np.ndarray = np.mean(features, axis=1, keepdims=True, dtype=np.float32)
        std: np.ndarray = np.std(features, axis=1, keepdims=True, dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        normalized: np.ndarray = (features - mean) / std
        return normalized.astype(np.float32, copy=False)

    @staticmethod
    def _compute_expected_frames(target_samples: int, n_fft: int, hop_length: int) -> int:
        """Compute deterministic frame count for librosa STFT-family features with center=False."""
        if target_samples < n_fft:
            return 1
        return 1 + ((target_samples - n_fft) // hop_length)
