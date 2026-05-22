from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kavach.features.extractor import AudioFeatureExtractor, FeatureExtractionError
from src.kavach.models.classifier import DeepfakeCNN, ModelInitializationError


LOGGER: logging.Logger = logging.getLogger("train")
TARGET_SAMPLE_RATE: int = 16000
BATCH_SIZE: int = 8
EPOCHS: int = 10
LEARNING_RATE: float = 1e-3
REAL_DIR: Path = PROJECT_ROOT / "dataset" / "real_voices"
FAKE_DIR: Path = PROJECT_ROOT / "dataset" / "fake_voices"
MODEL_PATH: Path = PROJECT_ROOT / "src" / "kavach" / "models" / "kavach_model_v1.pth"


class TrainingDataError(Exception):
    """Raised when training data is missing, malformed, or unavailable."""


@dataclass(frozen=True)
class AudioSample:
    """A single audio example and its integer class label."""

    path: Path
    label: int


class AudioDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load WAV files from real and fake directories and convert them to model features."""

    def __init__(
        self,
        real_dir: Path,
        fake_dir: Path,
        feature_extractor: AudioFeatureExtractor,
    ) -> None:
        self.real_dir: Path = real_dir
        self.fake_dir: Path = fake_dir
        self.feature_extractor: AudioFeatureExtractor = feature_extractor
        self.samples: list[AudioSample] = self._discover_samples()

        if len(self.samples) == 0:
            raise TrainingDataError("No audio files were found in the dataset directories.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sample: AudioSample = self.samples[index]
        features: np.ndarray = self._load_and_extract(sample.path)
        feature_tensor: Tensor = torch.from_numpy(features.astype(np.float32, copy=False))
        label_tensor: Tensor = torch.tensor([float(sample.label)], dtype=torch.float32)
        return feature_tensor, label_tensor

    def _discover_samples(self) -> list[AudioSample]:
        """Collect WAV files and assign labels for binary classification."""
        if not self.real_dir.exists():
            raise TrainingDataError(f"Missing real voice directory: {self.real_dir}")
        if not self.fake_dir.exists():
            raise TrainingDataError(f"Missing fake voice directory: {self.fake_dir}")

        real_files: list[Path] = sorted(self.real_dir.glob("*.wav"))
        fake_files: list[Path] = sorted(self.fake_dir.glob("*.wav"))

        if len(real_files) == 0:
            raise TrainingDataError(f"No WAV files found in {self.real_dir}")
        if len(fake_files) == 0:
            raise TrainingDataError(f"No WAV files found in {self.fake_dir}")

        samples: list[AudioSample] = [AudioSample(path=path, label=0) for path in real_files]
        samples.extend(AudioSample(path=path, label=1) for path in fake_files)
        return samples

    def _load_and_extract(self, path: Path) -> np.ndarray:
        """Load a WAV file, convert to PCM16 bytes, and extract fixed-shape features."""
        try:
            waveform, sample_rate = sf.read(file=path.as_posix(), dtype="float32", always_2d=False)
        except Exception as exc:
            raise TrainingDataError(f"Failed to read WAV file: {path}") from exc

        waveform_array: np.ndarray = np.asarray(waveform, dtype=np.float32)
        if waveform_array.ndim > 1:
            waveform_array = waveform_array.mean(axis=1)
        waveform_array = waveform_array.reshape(-1)
        if waveform_array.size == 0:
            raise TrainingDataError(f"Decoded empty waveform from: {path}")

        if int(sample_rate) != TARGET_SAMPLE_RATE:
            try:
                waveform_array = librosa.resample(
                    y=waveform_array,
                    orig_sr=int(sample_rate),
                    target_sr=TARGET_SAMPLE_RATE,
                )
            except Exception as exc:
                raise TrainingDataError(f"Failed to resample audio file: {path}") from exc

        waveform_array = np.clip(waveform_array, -1.0, 1.0)
        pcm16: np.ndarray = (waveform_array * 32767.0).astype(np.int16)
        audio_bytes: bytes = pcm16.tobytes()

        try:
            features: np.ndarray = self.feature_extractor.extract_features(audio_bytes)
        except FeatureExtractionError as exc:
            raise TrainingDataError(f"Feature extraction failed for {path}") from exc

        if features.ndim != 2:
            raise TrainingDataError(f"Unexpected feature shape for {path}: {features.shape}")

        return features


def configure_logging() -> None:
    """Configure process-wide logging for training progress."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducible training behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    """Resolve the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train for a single epoch and return loss and accuracy."""
    model.train()
    running_loss: float = 0.0
    correct: int = 0
    total: int = 0

    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device=device, dtype=torch.float32)
        batch_labels = batch_labels.to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits: Tensor = model(batch_features)
        probabilities: Tensor = torch.sigmoid(logits)
        loss: Tensor = criterion(probabilities, batch_labels)
        loss.backward()
        optimizer.step()

        batch_size: int = int(batch_labels.size(0))
        running_loss += float(loss.item()) * batch_size
        predictions: Tensor = (probabilities >= 0.5).to(dtype=torch.float32)
        correct += int((predictions == batch_labels).sum().item())
        total += batch_size

    avg_loss: float = running_loss / max(1, total)
    accuracy: float = correct / max(1, total)
    return avg_loss, accuracy


def main() -> None:
    """Train the deepfake classifier on local real/fake WAV samples."""
    configure_logging()
    set_seed()

    if not REAL_DIR.exists():
        raise TrainingDataError(f"Missing real voice directory: {REAL_DIR}")
    if not FAKE_DIR.exists():
        raise TrainingDataError(f"Missing fake voice directory: {FAKE_DIR}")

    device: torch.device = resolve_device()
    LOGGER.info("Using device: %s", device)

    feature_extractor: AudioFeatureExtractor = AudioFeatureExtractor(
        sample_rate=TARGET_SAMPLE_RATE,
        window_seconds=2.0,
        n_mfcc=40,
    )
    dataset: AudioDataset = AudioDataset(
        real_dir=REAL_DIR,
        fake_dir=FAKE_DIR,
        feature_extractor=feature_extractor,
    )
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    try:
        model: DeepfakeCNN = DeepfakeCNN(
            in_channels=2 * feature_extractor.n_mfcc,
            num_frames=feature_extractor.expected_frames,
        ).to(device)
    except Exception as exc:
        raise ModelInitializationError(f"Failed to initialize DeepfakeCNN: {exc}") from exc

    criterion: nn.Module = nn.BCELoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    LOGGER.info("Starting training with %s samples (%s batches per epoch).", len(dataset), len(loader))

    for epoch in range(1, EPOCHS + 1):
        loss, accuracy = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        print(f"Epoch {epoch:02d}/{EPOCHS} | Training Loss: {loss:.4f} | Accuracy: {accuracy * 100.0:.2f}%")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    LOGGER.info("Training complete. Saved model state_dict to %s", MODEL_PATH)


if __name__ == "__main__":
    try:
        main()
    except TrainingDataError as exc:
        LOGGER.error("Training data error: %s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        LOGGER.warning("Training interrupted by user.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.exception("Unexpected training failure: %s", exc)
        raise SystemExit(1) from exc
