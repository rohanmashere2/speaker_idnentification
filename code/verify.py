"""Offline open-set speaker verification using ECAPA-TDNN + WavLM.

Usage:
    python verify.py
    python verify.py test/test.wav
    python verify.py path/to/audio_folder

This script prints results to the terminal.
It does not write CSV or JSON output files.
"""

from pathlib import Path
import sys

import numpy as np
import torch

from speaker_utils import (
    BASE_DIR,
    DEVICE,
    ECAPA_WEIGHT,
    WAVLM_WEIGHT,
    load_models,
    load_audio,
    denoise,
    segments,
    quality,
    ecapa_embedding,
    wavlm_embedding,
    combine_versions,
    cosine,
    normalize,
)

DATABASE_DIR = BASE_DIR / "database"
TEST_DIR = BASE_DIR / "test"

SUPPORTED = {".wav", ".flac", ".ogg"}

# Requested operating thresholds.
MATCH_THRESHOLD = 0.70
MIN_MARGIN = 0.15
MIN_ECAPA_SCORE = 0.45
MIN_VALID_SEGMENTS = 2


# ============================================================
# LOAD ENROLLED VOICEPRINTS
# ============================================================

def load_voiceprints():
    if not DATABASE_DIR.is_dir():
        raise FileNotFoundError(
            f"Database folder not found:\n{DATABASE_DIR}"
        )

    profiles = {}

    for path in sorted(DATABASE_DIR.glob("*.pt")):
        try:
            obj = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )

            if (
                not isinstance(obj, dict)
                or "ecapa" not in obj
                or "wavlm" not in obj
            ):
                print(
                    f"[SKIP] {path.name}: incompatible voiceprint. "
                    "Re-enroll using the current enroll.py."
                )
                continue

            ecapa_vector = normalize(
                torch.as_tensor(
                    obj["ecapa"], dtype=torch.float32
                ).cpu().numpy()
            )

            wavlm_vector = normalize(
                torch.as_tensor(
                    obj["wavlm"], dtype=torch.float32
                ).cpu().numpy()
            )

            profiles[path.stem] = {
                "ecapa": ecapa_vector,
                "wavlm": wavlm_vector,
            }

        except Exception as exc:
            print(
                f"[SKIP] {path.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    if not profiles:
        raise RuntimeError(
            "No compatible voiceprints found. "
            "Check database/ or run enroll.py."
        )

    return profiles


# ============================================================
# SCORE ONE TEST RECORDING
# ============================================================

def score_file(
    path,
    profiles,
    ecapa_model,
    extractor,
    wavlm_model,
):
    raw_audio = load_audio(path)
    clean_audio = denoise(raw_audio)

    raw_segments = segments(raw_audio)
    clean_segments = segments(clean_audio)

    segment_count = min(
        len(raw_segments),
        len(clean_segments),
    )

    scores = {
        name: {
            "fused": [],
            "ecapa": [],
            "wavlm": [],
        }
        for name in profiles
    }

    usable_segments = 0

    for index in range(segment_count):
        raw_segment = raw_segments[index]
        clean_segment = clean_segments[index]

        # Quality checks are based on the original audio.
        if quality(raw_segment) <= 0:
            print(
                f"  Segment {index + 1}: "
                "low quality; skipped."
            )
            continue

        try:
            ecapa_raw = ecapa_embedding(
                ecapa_model,
                raw_segment,
            )
            ecapa_clean = ecapa_embedding(
                ecapa_model,
                clean_segment,
            )

            wavlm_raw = wavlm_embedding(
                wavlm_model,
                extractor,
                raw_segment,
            )
            wavlm_clean = wavlm_embedding(
                wavlm_model,
                extractor,
                clean_segment,
            )

            ecapa_vector, ecapa_agreement = combine_versions(
                ecapa_raw,
                ecapa_clean,
            )
            wavlm_vector, wavlm_agreement = combine_versions(
                wavlm_raw,
                wavlm_clean,
            )

            for person, profile in profiles.items():
                ecapa_score = cosine(
                    ecapa_vector,
                    profile["ecapa"],
                )

                wavlm_score = cosine(
                    wavlm_vector,
                    profile["wavlm"],
                )

                fused_score = (
                    ECAPA_WEIGHT * ecapa_score
                    + WAVLM_WEIGHT * wavlm_score
                )

                scores[person]["ecapa"].append(ecapa_score)
                scores[person]["wavlm"].append(wavlm_score)
                scores[person]["fused"].append(fused_score)

            usable_segments += 1

        except Exception as exc:
            print(
                f"  Segment {index + 1}: "
                f"{type(exc).__name__}: {exc}; skipped."
            )

    if usable_segments == 0:
        return {
            "decision": "UNKNOWN",
            "reason": "no usable speech segments",
            "ranked": [],
            "used": 0,
            "total": segment_count,
            "margin": None,
        }

    # Aggregate per-segment scores for each enrolled person.
    ranked = []

    for person, person_scores in scores.items():
        if not person_scores["fused"]:
            continue

        fused = np.asarray(
            person_scores["fused"],
            dtype=np.float32,
        )
        ecapa = np.asarray(
            person_scores["ecapa"],
            dtype=np.float32,
        )
        wavlm = np.asarray(
            person_scores["wavlm"],
            dtype=np.float32,
        )

        median_fused = float(np.median(fused))
        percentile_75 = float(np.percentile(fused, 75))

        robust_fused = (
            0.75 * median_fused
            + 0.25 * percentile_75
        )

        ranked.append({
            "person": person,
            "fused": robust_fused,
            "median_fused": median_fused,
            "ecapa": float(np.median(ecapa)),
            "wavlm": float(np.median(wavlm)),
        })

    ranked.sort(
        key=lambda item: item["fused"],
        reverse=True,
    )

    if not ranked:
        return {
            "decision": "UNKNOWN",
            "reason": "no valid candidate scores",
            "ranked": [],
            "used": usable_segments,
            "total": segment_count,
            "margin": None,
        }

    best = ranked[0]

    second_score = (
        ranked[1]["fused"]
        if len(ranked) > 1
        else -1.0
    )

    margin = best["fused"] - second_score

    # Require all identity checks to pass.
    reasons = []

    if usable_segments < MIN_VALID_SEGMENTS:
        reasons.append("too few usable speech segments")

    if best["fused"] < MATCH_THRESHOLD:
        reasons.append("fused score below threshold")

    if margin < MIN_MARGIN:
        reasons.append("top candidates too close")

    if best["ecapa"] < MIN_ECAPA_SCORE:
        reasons.append("ECAPA score below threshold")

    if reasons:
        decision = "UNKNOWN"
        reason = "; ".join(reasons)
    else:
        decision = f"MATCH: {best['person']}"
        reason = "all identity checks passed"

    return {
        "decision": decision,
        "reason": reason,
        "ranked": ranked,
        "used": usable_segments,
        "total": segment_count,
        "margin": margin,
    }


# ============================================================
# FIND TEST FILES
# ============================================================

def get_test_files():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1]).expanduser()

        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        if path.is_file():
            if path.suffix.lower() not in SUPPORTED:
                raise ValueError(
                    f"Unsupported audio format: {path.suffix}"
                )
            return [path]

        if path.is_dir():
            return sorted(
                item for item in path.rglob("*")
                if item.is_file()
                and item.suffix.lower() in SUPPORTED
            )

        raise FileNotFoundError(
            f"Test path not found:\n{path}"
        )

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test folder not found:\n{TEST_DIR}"
        )

    return sorted(
        item for item in TEST_DIR.rglob("*")
        if item.is_file()
        and item.suffix.lower() in SUPPORTED
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("OFFLINE SPEAKER VERIFICATION")
    print("=" * 65)

    print("\nDecision thresholds:")
    print(f"  Fused score : {MATCH_THRESHOLD:.2f}")
    print(f"  Margin      : {MIN_MARGIN:.2f}")
    print(f"  ECAPA score : {MIN_ECAPA_SCORE:.2f}")
    print(f"  Min segments: {MIN_VALID_SEGMENTS}")

    profiles = load_voiceprints()
    files = get_test_files()

    if not files:
        raise RuntimeError(
            "No supported audio recordings found."
        )

    ecapa_model, extractor, wavlm_model = load_models()

    print(f"\nDevice: {DEVICE}")
    print(
        f"Loaded {len(profiles)} voiceprints: "
        f"{', '.join(sorted(profiles))}"
    )

    for path in files:
        print("\n" + "=" * 65)
        print(f"FILE: {path}")

        try:
            result = score_file(
                path,
                profiles,
                ecapa_model,
                extractor,
                wavlm_model,
            )

            print(
                f"Usable speech segments: "
                f"{result['used']}/{result['total']}"
            )

            if result["ranked"]:
                print("\nCandidate scores:")

                for item in result["ranked"]:
                    print(
                        f"  {item['person']:<14} "
                        f"fused={item['fused']:.4f} "
                        f"median={item['median_fused']:.4f} "
                        f"ECAPA={item['ecapa']:.4f} "
                        f"WavLM={item['wavlm']:.4f}"
                    )

                if result["margin"] is not None:
                    print(
                        f"\nTop-vs-second margin: "
                        f"{result['margin']:.4f}"
                    )

            print(f"\nFINAL RESULT: {result['decision']}")
            print(f"Reason: {result['reason']}")

        except Exception as exc:
            print(
                f"ERROR: {type(exc).__name__}: {exc}"
            )

    print("\n" + "=" * 65)
    print("VERIFICATION COMPLETE")
    print("Console only: no CSV or JSON files saved.")
    print(
        "Thresholds must be validated on held-out "
        "known and unknown speakers."
    )


if __name__ == "__main__":
    main()