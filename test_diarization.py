"""Standalone diarization test — runs pyannote pipeline only.

This script does NOT perform speaker identification.
It simply prints "who spoke when" using pyannote.audio.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Specifications, Problem, Resolution

from speaker_utils import BASE_DIR, SR, load_audio

# Project paths
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
    audio = load_audio(audio_path)

    duration = len(audio) / SR
    print(f"Audio duration: {duration:.2f} seconds")
    print("Running speaker diarization...")

    # Passing a waveform dictionary avoids TorchCodec file decoding.
    waveform = torch.from_numpy(
        np.ascontiguousarray(audio)
    ).unsqueeze(0)

    result = pipeline({
        "waveform": waveform,
        "sample_rate": SR,
    })

    print("\n--- DIARIZATION RESULTS ---")
    turns = list(result.speaker_diarization)

    if not turns:
        print("No speaker segments detected.")
    else:
        for turn, speaker in turns:
            seg_duration = turn.end - turn.start
            print(
                f"{turn.start:8.2f}s - {turn.end:8.2f}s | "
                f"{speaker} | Duration: {seg_duration:.2f}s"
            )

    print("\nDiarization completed.")


if __name__ == "__main__":
    main()
