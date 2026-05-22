from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn


class ModelInitializationError(Exception):
    """Raised when the deepfake model cannot be initialized or loaded."""


class InferenceError(Exception):
    """Raised when inference fails due to invalid inputs or runtime errors."""


class DeepfakeCNN(nn.Module):
    """Lightweight 1D-CNN for deepfake voice classification from MFCC+delta features."""

    def __init__(self, in_channels: int, num_frames: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.in_channels: int = in_channels
        self.num_frames: int = num_frames

        self.conv1: nn.Conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=5,
            padding=2,
        )
        self.bn1: nn.BatchNorm1d = nn.BatchNorm1d(64)
        self.pool1: nn.MaxPool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2: nn.Conv1d = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.bn2: nn.BatchNorm1d = nn.BatchNorm1d(128)
        self.pool2: nn.MaxPool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3: nn.Conv1d = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.bn3: nn.BatchNorm1d = nn.BatchNorm1d(128)
        self.pool3: nn.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(output_size=16)

        self.dropout: nn.Dropout = nn.Dropout(p=dropout)
        self.fc1: nn.Linear = nn.Linear(128 * 16, 64)
        self.fc_out: nn.Linear = nn.Linear(64, 1)
        self.activation: nn.ReLU = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Expected input shape: (batch_size, in_channels, num_frames)
        Returns logits shape: (batch_size, 1)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.pool3(x)

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        logits: Tensor = self.fc_out(x)
        return logits


class DeepfakeDetector:
    """Wrapper for model lifecycle and robust deepfake probability inference."""

    def __init__(
        self,
        n_mfcc: int = 40,
        num_frames: int = 198,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.n_mfcc: int = n_mfcc
        self.expected_channels: int = 2 * self.n_mfcc
        self.num_frames: int = num_frames
        self._logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self.device: torch.device = self._resolve_device(device)
        self.model: DeepfakeCNN = self._initialize_model(
            expected_channels=self.expected_channels,
            num_frames=self.num_frames,
            model_path=model_path,
        )
        self.model.eval()

    def predict(self, features: np.ndarray) -> float:
        """Run model inference and return deepfake probability in [0.0, 1.0]."""
        try:
            validated_features: np.ndarray = self._validate_and_prepare_features(features)
            input_tensor: Tensor = torch.from_numpy(validated_features).to(self.device)

            with torch.inference_mode():
                logits: Tensor = self.model(input_tensor)
                probs: Tensor = torch.sigmoid(logits)

            probability: float = float(probs.squeeze().item())
            if probability < 0.0:
                return 0.0
            if probability > 1.0:
                return 1.0
            return probability
        except InferenceError:
            raise
        except Exception as exc:
            self._logger.exception("Unexpected deepfake inference failure: %s", exc)
            raise InferenceError("Unexpected deepfake inference failure.") from exc

    def _initialize_model(
        self,
        expected_channels: int,
        num_frames: int,
        model_path: Optional[str],
    ) -> DeepfakeCNN:
        """Create model and optionally load weights from a checkpoint path."""
        try:
            model: DeepfakeCNN = DeepfakeCNN(
                in_channels=expected_channels,
                num_frames=num_frames,
                dropout=0.3,
            ).to(self.device)

            if model_path is not None:
                resolved_path: Path = Path(model_path).expanduser().resolve()
                if not resolved_path.exists() or not resolved_path.is_file():
                    raise ModelInitializationError(
                        f"Model checkpoint does not exist: {resolved_path}"
                    )

                checkpoint: object = torch.load(
                    resolved_path,
                    map_location=self.device,
                )

                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict_obj: object = checkpoint["state_dict"]
                    if not isinstance(state_dict_obj, dict):
                        raise ModelInitializationError(
                            "Checkpoint key 'state_dict' is not a valid dictionary."
                        )
                    model.load_state_dict(state_dict_obj, strict=True)
                elif isinstance(checkpoint, dict):
                    model.load_state_dict(checkpoint, strict=True)
                else:
                    raise ModelInitializationError(
                        "Unsupported checkpoint format for DeepfakeCNN."
                    )

            return model
        except ModelInitializationError:
            raise
        except RuntimeError as exc:
            raise ModelInitializationError(
                f"Model weights are incompatible with architecture: {exc}"
            ) from exc
        except Exception as exc:
            raise ModelInitializationError(
                f"Failed to initialize deepfake model: {exc}"
            ) from exc

    def _validate_and_prepare_features(self, features: np.ndarray) -> np.ndarray:
        """
        Validate input shape and convert to model-ready batch tensor format.

        Accepted inputs:
        - (channels, frames)
        - (1, channels, frames)
        Returns shape: (1, channels, frames)
        """
        if not isinstance(features, np.ndarray):
            raise InferenceError("Features must be provided as a NumPy array.")

        if features.size == 0:
            raise InferenceError("Input features are empty.")

        if not np.issubdtype(features.dtype, np.floating):
            features = features.astype(np.float32, copy=False)
        else:
            features = features.astype(np.float32, copy=False)

        if not np.all(np.isfinite(features)):
            raise InferenceError("Input features contain non-finite values.")

        if features.ndim == 2:
            channels: int = int(features.shape[0])
            frames: int = int(features.shape[1])
            batch: np.ndarray = np.expand_dims(features, axis=0)
        elif features.ndim == 3:
            if int(features.shape[0]) != 1:
                raise InferenceError(
                    "Batch inference is not supported in this wrapper. Expected batch size 1."
                )
            channels = int(features.shape[1])
            frames = int(features.shape[2])
            batch = features
        else:
            raise InferenceError(
                "Invalid feature dimensions. Expected 2D (channels, frames) or 3D (1, channels, frames)."
            )

        if channels != self.expected_channels:
            raise InferenceError(
                f"Channel mismatch. Expected {self.expected_channels}, got {channels}."
            )

        if frames != self.num_frames:
            raise InferenceError(
                f"Frame-length mismatch. Expected {self.num_frames}, got {frames}."
            )

        if not batch.flags["C_CONTIGUOUS"]:
            batch = np.ascontiguousarray(batch, dtype=np.float32)

        return batch

    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        """Resolve compute device with safe defaults for local/on-device inference."""
        if device is not None:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
