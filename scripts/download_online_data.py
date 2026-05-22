from __future__ import annotations

import argparse
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


LOGGER: logging.Logger = logging.getLogger("download_online_data")
TARGET_SAMPLE_RATE: int = 16000
REAL_SAMPLE_COUNT: int = 20
FAKE_SAMPLE_COUNT: int = 20
CHUNK_SECONDS: float = 2.0
CHUNK_SAMPLES: int = int(TARGET_SAMPLE_RATE * CHUNK_SECONDS)
TOTAL_SAMPLE_COUNT: int = REAL_SAMPLE_COUNT + FAKE_SAMPLE_COUNT


class DatasetDownloadError(Exception):
    """Raised when the remote sample cannot be loaded or saved."""


def configure_logging(verbosity: int) -> None:
    """Configure console logging for progress and debugging."""
    level: int = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_output_directories(root_dir: Path) -> tuple[Path, Path]:
    """Create the real and fake voice directories if they do not exist."""
    real_dir: Path = root_dir / "dataset" / "real_voices"
    fake_dir: Path = root_dir / "dataset" / "fake_voices"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    return real_dir, fake_dir


def resolve_remote_example() -> str:
    """Resolve a remote built-in example path from librosa with graceful fallbacks."""
    candidates: tuple[str, ...] = ("vibeace", "trumpet", "choice", "brahms")
    for candidate in candidates:
        try:
            return librosa.ex(candidate)
        except Exception:
            continue
    raise DatasetDownloadError("Could not resolve any remote librosa example audio file.")


def load_example_audio() -> tuple[np.ndarray, int]:
    """Load a remote built-in example audio file and resample it to the target rate."""
    try:
        example_path: str = resolve_remote_example()
        waveform, sample_rate = librosa.load(example_path, sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise DatasetDownloadError("Failed to load remote example audio with librosa.") from exc

    if waveform.size == 0 or sample_rate != TARGET_SAMPLE_RATE:
        raise DatasetDownloadError("Loaded example audio is empty or invalid.")

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    waveform = np.clip(waveform, -1.0, 1.0)
    return waveform, sample_rate


def chunk_audio(waveform: np.ndarray) -> list[np.ndarray]:
    """Split waveform into 20 fixed 2-second chunks, padding the tail as needed."""
    if waveform.size == 0:
        raise DatasetDownloadError("Cannot chunk an empty waveform.")

    required_samples: int = TOTAL_SAMPLE_COUNT * CHUNK_SAMPLES
    if waveform.size < required_samples:
        repeats: int = int(np.ceil(required_samples / float(waveform.size)))
        waveform = np.tile(waveform, repeats)

    chunks: list[np.ndarray] = []
    for index in range(TOTAL_SAMPLE_COUNT):
        start: int = index * CHUNK_SAMPLES
        stop: int = start + CHUNK_SAMPLES
        chunk: np.ndarray = waveform[start:stop]
        if chunk.size < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - chunk.size), mode="constant")
        chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))

    return chunks


def save_waveform(path: Path, waveform: np.ndarray) -> None:
    """Write normalized float32 waveform to disk as WAV."""
    try:
        sf.write(
            file=path.as_posix(),
            data=waveform.astype(np.float32, copy=False),
            samplerate=TARGET_SAMPLE_RATE,
        )
    except Exception as exc:
        raise DatasetDownloadError(f"Failed to write audio file: {path}") from exc


def create_fake_voice(waveform: np.ndarray, index: int) -> np.ndarray:
    """Apply deterministic distortion to simulate synthetic/robotic speech."""
    try:
        pitched: np.ndarray = librosa.effects.pitch_shift(
            y=waveform,
            sr=TARGET_SAMPLE_RATE,
            n_steps=2.5 if index % 2 == 0 else -2.0,
        )
        noise_scale: float = 0.012 + (index * 0.0005)
        noisy: np.ndarray = pitched + np.random.default_rng(seed=index + 1337).normal(
            loc=0.0,
            scale=noise_scale,
            size=pitched.shape,
        ).astype(np.float32)
        distorted: np.ndarray = np.tanh(noisy * 1.8).astype(np.float32, copy=False)
        return np.clip(distorted, -1.0, 1.0)
    except Exception as exc:
        raise DatasetDownloadError(f"Failed to distort audio sample at index {index}.") from exc


def download_dataset_artifacts(
    real_dir: Path,
    fake_dir: Path,
) -> None:
    """Download the remote example, build 20 real and 20 fake chunks, and save them."""
    waveform, _sample_rate = load_example_audio()
    chunks: list[np.ndarray] = chunk_audio(waveform)

    for index, chunk in enumerate(chunks):
        if index < REAL_SAMPLE_COUNT:
            output_path: Path = real_dir / f"real_{index:03d}.wav"
            save_waveform(output_path, chunk)
            LOGGER.info("Saved real sample %s/%s -> %s", index + 1, REAL_SAMPLE_COUNT, output_path)
            continue

        fake_index: int = index - REAL_SAMPLE_COUNT
        fake_waveform: np.ndarray = create_fake_voice(chunk, fake_index)
        output_path = fake_dir / f"fake_{fake_index:03d}.wav"
        save_waveform(output_path, fake_waveform)
        LOGGER.info("Saved fake sample %s/%s -> %s", fake_index + 1, FAKE_SAMPLE_COUNT, output_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the dataset downloader."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Fetch a remote example audio file, split it into real/fake training samples, and save them locally."
    )
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Project root directory where dataset/ will be created.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for dataset download and artifact generation."""
    args: argparse.Namespace = parse_args()
    configure_logging(1 if bool(args.verbose) else 0)

    root_dir: Path = Path(args.root_dir).expanduser().resolve()
    real_dir, fake_dir = ensure_output_directories(root_dir)

    LOGGER.info("Output directories ready: %s | %s", real_dir, fake_dir)
    LOGGER.info("Using remote librosa example audio source and generating %s total files.", TOTAL_SAMPLE_COUNT)

    try:
        download_dataset_artifacts(
            real_dir=real_dir,
            fake_dir=fake_dir,
        )
    except DatasetDownloadError as exc:
        LOGGER.error("Dataset download failed: %s", exc)
        raise SystemExit(1) from exc

    LOGGER.info("Success. Generated %s real and %s fake audio files.", REAL_SAMPLE_COUNT, FAKE_SAMPLE_COUNT)


if __name__ == "__main__":
    main()
