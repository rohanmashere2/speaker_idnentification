"""Quality-weighted ECAPA + WavLM speaker enrollment with outlier rejection.

Expected input:
    enroll/Person_A/*.wav
    enroll/Person_B/*.wav

Creates:
    database/Person_A.pt
    database/Person_B.pt

Existing voiceprints are not overwritten without confirmation.
"""

from pathlib import Path

import numpy as np
import torch

from speaker_utils import (
    BASE_DIR,
    DATABASE_DIR,
    DEVICE,
    ECAPA_WEIGHT,
    WAVLM_WEIGHT,
    load_models,
    process_recording,
    weighted_centroid,
    reject_outliers_mad,
    normalize,
)

ENROLL_DIR = BASE_DIR / "enroll"

SUPPORTED = {".wav"}


def main():
    if not ENROLL_DIR.is_dir():
        raise FileNotFoundError(
            f"Enrollment directory not found:\n{ENROLL_DIR}"
        )

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    people = sorted(
        path for path in ENROLL_DIR.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
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
    print(f"Fusion: ECAPA={ECAPA_WEIGHT:.0%}, WavLM={WAVLM_WEIGHT:.0%}")
    print("Output: .pt voiceprints only")
    print("MAD outlier rejection enabled (threshold=3.0)")
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
                        f"  [WARN] {audio_path.name}: "
                        "embedding extraction failed on all segments."
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

        # ------------------------------------------------
        # MAD outlier rejection — remove anomalous
        # recordings that would corrupt the centroid.
        # ------------------------------------------------
        clean_ecapa, ecapa_mask = reject_outliers_mad(
            ecapa_vectors, threshold=3.0,
        )
        clean_wavlm, wavlm_mask = reject_outliers_mad(
            wavlm_vectors, threshold=3.0,
        )

        # Use the intersection: keep only recordings that
        # passed BOTH ECAPA and WavLM outlier checks.
        combined_mask = ecapa_mask & wavlm_mask
        n_rejected = int(np.sum(~combined_mask))

        if n_rejected > 0:
            print(
                f"  MAD outlier rejection: "
                f"{n_rejected} recording(s) flagged as outliers"
            )
            for idx, kept in enumerate(combined_mask):
                if not kept:
                    print(f"    - Rejected: {files[idx].name}")

        # Filter to kept recordings.
        kept_ecapa = [v for v, m in zip(ecapa_vectors, combined_mask) if m]
        kept_wavlm = [v for v, m in zip(wavlm_vectors, combined_mask) if m]
        kept_weights = [w for w, m in zip(weights, combined_mask) if m]

        if not kept_weights:
            # Fallback: if MAD rejects everything, keep originals.
            print("  MAD rejected all — falling back to full set.")
            kept_ecapa = ecapa_vectors
            kept_wavlm = wavlm_vectors
            kept_weights = weights

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
                kept_ecapa,
                kept_weights,
            )

            wavlm_centroid = weighted_centroid(
                kept_wavlm,
                kept_weights,
            )

            voiceprint = {
                "version": 3,
                "person": person_dir.name,
                "model": "ECAPA-TDNN + WavLMForXVector",
                "sample_rate": 16000,
                "ecapa_weight": ECAPA_WEIGHT,
                "wavlm_weight": WAVLM_WEIGHT,
                "recordings_found": len(files),
                "recordings_used": len(kept_weights),
                "recordings_rejected": n_rejected,
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
                f"  Recordings kept: {len(kept_weights)}"
            )
            print(
                f"  Enrollment quality average: "
                f"{sum(kept_weights) / len(kept_weights):.3f}"
            )

        except Exception as exc:
            print(
                f"[FAILED] Could not save {person_dir.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 60)
    print("ENROLLMENT FINISHED")
    print("=" * 60)
    print("Review all WARN, ERROR, and SKIP messages.")
    print("Only .pt voiceprints are written by this script.")


if __name__ == "__main__":
    main()
