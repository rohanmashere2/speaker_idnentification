"""Quality-weighted ECAPA + WavLM speaker enrollment.

Expected input:
    enroll/Person_A/*.wav
    enroll/Person_B/*.wav

Creates:
    database/Person_A.pt
    database/Person_B.pt

Existing voiceprints are not overwritten without confirmation.
"""

from pathlib import Path

import torch

from speaker_utils import (
    BASE_DIR,
    DEVICE,
    load_models,
    process_recording,
    weighted_centroid,
)

ENROLL_DIR = BASE_DIR / "enroll"
DATABASE_DIR = BASE_DIR / "database"

SUPPORTED = {".wav", ".flac", ".ogg"}

ECAPA_WEIGHT = 0.40
WAVLM_WEIGHT = 0.60


def main():
    if not ENROLL_DIR.is_dir():
        raise FileNotFoundError(
            f"Enrollment directory not found:\n{ENROLL_DIR}"
        )

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    people = sorted(
        path for path in ENROLL_DIR.iterdir()
        if path.is_dir()
    )

    if not people:
        raise RuntimeError(
            f"No person folders found in:\n{ENROLL_DIR}"
        )

    print("=" * 60)
    print("ECAPA-TDNN + WAVLM ENROLLMENT")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"People found: {len(people)}")
    print("Output: .pt voiceprints only")
    print("No CSV or JSON files will be created.\n")

    ecapa, extractor, wavlm = load_models()

    for person_dir in people:
        files = sorted(
            path for path in person_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SUPPORTED
        )

        if not files:
            print(
                f"[SKIP] {person_dir.name}: "
                "no supported audio recordings."
            )
            continue

        destination = DATABASE_DIR / f"{person_dir.name}.pt"

        print("\n" + "-" * 60)
        print(f"PERSON: {person_dir.name}")
        print(f"Recordings found: {len(files)}")

        ecapa_vectors = []
        wavlm_vectors = []
        weights = []

        for audio_path in files:
            try:
                result = process_recording(
                    audio_path,
                    ecapa,
                    extractor,
                    wavlm,
                )

                if result is None:
                    print(
                        f"  [LOW QUALITY] {audio_path.name}: "
                        "no usable speech segments."
                    )
                    continue

                ecapa_vectors.append(result["ecapa"])
                wavlm_vectors.append(result["wavlm"])
                weights.append(float(result["quality"]))

                print(
                    f"  [OK] {audio_path.name}: "
                    f"{result['segments_used']}/"
                    f"{result['segments_total']} segments, "
                    f"quality={result['quality']:.3f}"
                )

            except Exception as exc:
                print(
                    f"  [ERROR] {audio_path.name}: "
                    f"{type(exc).__name__}: {exc}"
                )

        if not weights:
            print(
                f"[FAILED] {person_dir.name}: "
                "no usable recordings; no voiceprint created."
            )
            continue

        print(f"\nUsable recordings: {len(weights)}/{len(files)}")

        # Never overwrite an existing voiceprint accidentally.
        if destination.exists():
            answer = input(
                f"{destination.name} already exists. "
                "Replace it? Type YES to confirm: "
            ).strip()

            if answer != "YES":
                print("[SKIP] Existing voiceprint preserved.")
                continue

        try:
            ecapa_centroid = weighted_centroid(
                ecapa_vectors,
                weights,
            )

            wavlm_centroid = weighted_centroid(
                wavlm_vectors,
                weights,
            )

            voiceprint = {
                "version": 2,
                "person": person_dir.name,
                "model": "ECAPA-TDNN + WavLMForXVector",
                "sample_rate": 16000,
                "ecapa_weight": ECAPA_WEIGHT,
                "wavlm_weight": WAVLM_WEIGHT,
                "recordings_found": len(files),
                "recordings_used": len(weights),
                "ecapa": torch.tensor(
                    ecapa_centroid,
                    dtype=torch.float32,
                ),
                "wavlm": torch.tensor(
                    wavlm_centroid,
                    dtype=torch.float32,
                ),
            }

            torch.save(voiceprint, destination)

            print(f"[SAVED] {destination}")
            print(
                f"  Enrollment quality average: "
                f"{sum(weights) / len(weights):.3f}"
            )

        except Exception as exc:
            print(
                f"[FAILED] Could not save {person_dir.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 60)
    print("ENROLLMENT FINISHED")
    print("=" * 60)
    print("Review all LOW QUALITY, ERROR, and SKIP messages.")
    print("Only .pt voiceprints are written by this script.")


if __name__ == "__main__":
    main()