
from pathlib import Path
import math

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Specifications, Problem, Resolution

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = (
    BASE_DIR
    / "model"
    / "diarization"
    / "pyannote-speaker-diarization-community-1"
)

# Required for safe loading of this trusted local pyannote checkpoint.
torch.serialization.add_safe_globals(
    [Specifications, Problem, Resolution]
)


def load_audio(audio_path: Path, target_sr: int = 16000):
    """Load audio without using TorchCodec or FFmpeg."""

    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True,
    )

    # Convert stereo/multichannel audio to mono.
    audio = audio.mean(axis=1)

    # Resample to 16 kHz if required.
    if sample_rate != target_sr:
        divisor = math.gcd(int(sample_rate), target_sr)
        audio = resample_poly(
            audio,
            target_sr // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)
        sample_rate = target_sr

    if audio.size == 0:
        raise ValueError("The audio file is empty.")

    # pyannote expects [channels, samples].
    waveform = torch.from_numpy(
        np.ascontiguousarray(audio)
    ).unsqueeze(0)

    return waveform, sample_rate


def main():
    if not MODEL_DIR.joinpath("config.yaml").is_file():
        raise FileNotFoundError(
            f"Model config not found: {MODEL_DIR}"
        )

    audio_path = Path(
        input("Enter audio file path: ").strip().strip('"')
    ).expanduser()

    if not audio_path.is_absolute():
        audio_path = BASE_DIR / audio_path

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print("Loading local diarization pipeline...")
    pipeline = Pipeline.from_pretrained(str(MODEL_DIR))
    pipeline.to(torch.device("cpu"))

    print("Loading audio...")
    waveform, sample_rate = load_audio(audio_path)

    print(f"Audio duration: {waveform.shape[1] / sample_rate:.2f} seconds")
    print("Running speaker diarization...")

    # Passing a waveform dictionary avoids TorchCodec file decoding.
    result = pipeline({
        "waveform": waveform,
        "sample_rate": sample_rate,
    })

    print("\n--- DIARIZATION RESULTS ---")
    turns = list(result.speaker_diarization)

    if not turns:
        print("No speaker segments detected.")
    else:
        for turn, speaker in turns:
            duration = turn.end - turn.start
            print(
                f"{turn.start:8.2f}s - {turn.end:8.2f}s | "
                f"{speaker} | Duration: {duration:.2f}s"
            )

    print("\nDiarization completed.")


if __name__ == "__main__":
    main()
