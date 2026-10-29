"""Offline speaker diarization + identification.

Uses pyannote.audio for speaker diarization ("who spoke when"),
then matches each diarized speaker cluster to enrolled voiceprints
using dual-model ECAPA-TDNN + WavLM fusion.

Usage:
    python diarize_identify.py
    python diarize_identify.py path/to/audio.wav
    python diarize_identify.py path/to/audio.wav --min-speakers 2 --max-speakers 4
    python diarize_identify.py path/to/audio.wav --num-speakers 3

Output is printed to the terminal (no files are written).
"""

import os

# Force local/offline model loading where supported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from pyannote.audio import Pipeline
from pyannote.audio.core.task import Specifications, Problem, Resolution

from speaker_utils import (
    BASE_DIR,
    DEVICE,
    SR,
    ECAPA_WEIGHT,
    WAVLM_WEIGHT,
    MATCH_THRESHOLD,
    MIN_MARGIN,
    MIN_ECAPA_SCORE,
    MIN_VALID_SEGMENTS,
    load_audio,
    denoise,
    quality,
    load_models,
    ecapa_embedding,
    wavlm_embedding,
    combine_versions,
    normalize,
    cosine,
    load_voiceprints,
)


# ============================================================
# DIARIZATION CONFIGURATION
# ============================================================

DIARIZATION_DIR = (
    BASE_DIR / "model" / "diarization"
    / "pyannote-speaker-diarization-community-1"
)

# Short turns remain in the timeline but may get too few
# matching windows for a confident identity decision.
MIN_SEGMENT_SECONDS = 1.5

# Long turns are split into overlapping windows for scoring.
MATCH_WINDOW_SECONDS = 3.0
MATCH_WINDOW_OVERLAP_SECONDS = 1.0

# Report pauses at least this long.
MIN_PAUSE_SECONDS = 0.25


# ============================================================
# LOAD DIARIZATION PIPELINE
# ============================================================

def load_diarization_pipeline():
    """Load the local pyannote speaker diarization pipeline."""
    if not DIARIZATION_DIR.exists():
        raise FileNotFoundError(
            f"Diarization model directory missing:\n"
            f"{DIARIZATION_DIR}"
        )

    # Allow the trusted local pyannote checkpoint classes.
    torch.serialization.add_safe_globals([
        Specifications,
        Problem,
        Resolution,
    ])

    print("\nLoading local pyannote diarization pipeline...")

    pipeline = Pipeline.from_pretrained(str(DIARIZATION_DIR))

    # Use GPU when available for faster segmentation
    # and embedding extraction.
    device = torch.device(DEVICE)
    pipeline.to(device)

    print(f"Pyannote pipeline loaded on {DEVICE}.")
    return pipeline


# ============================================================
# MATCHING WINDOWS
# ============================================================

def make_matching_windows(segment):
    """Split a diarized speech turn into overlapping windows.

    Each window is scored independently against enrolled
    voiceprints.  Overlapping windows give more robust
    aggregate scores for longer turns.
    """
    segment = np.asarray(segment, dtype=np.float32)

    min_samples = int(MIN_SEGMENT_SECONDS * SR)
    window_samples = int(MATCH_WINDOW_SECONDS * SR)
    step_samples = int(
        (MATCH_WINDOW_SECONDS - MATCH_WINDOW_OVERLAP_SECONDS)
        * SR
    )

    if len(segment) < min_samples:
        return []

    if len(segment) <= window_samples:
        return [segment]

    windows = []
    start = 0

    while start < len(segment):
        end = min(start + window_samples, len(segment))
        chunk = segment[start:end]

        if len(chunk) >= min_samples:
            windows.append(chunk)

        if end >= len(segment):
            break

        start += step_samples

    return windows


# ============================================================
# SCORE ONE MATCHING WINDOW
# ============================================================

def score_window(
    window,
    voiceprints,
    ecapa_model,
    extractor,
    wavlm_model,
):
    """Extract dual embeddings and score against all voiceprints.

    The full denoising + raw/clean combination pipeline is used
    here, matching the enrollment and verification paths.  This
    ensures consistent scoring across all three workflows.

    Returns (scores_dict, quality_weight) or None on failure.
    """
    raw = window
    clean = denoise(window)

    q = quality(raw)

    try:
        ecapa_raw = ecapa_embedding(ecapa_model, raw)
        ecapa_clean = ecapa_embedding(ecapa_model, clean)

        wavlm_raw = wavlm_embedding(
            wavlm_model, extractor, raw,
        )
        wavlm_clean = wavlm_embedding(
            wavlm_model, extractor, clean,
        )

        ecapa_vec, _ = combine_versions(ecapa_raw, ecapa_clean)
        wavlm_vec, _ = combine_versions(wavlm_raw, wavlm_clean)

    except Exception as exc:
        print(
            f"    Embedding failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None

    scores = {}

    for person, profile in voiceprints.items():
        ecapa_score = cosine(ecapa_vec, profile["ecapa"])
        wavlm_score = cosine(wavlm_vec, profile["wavlm"])
        fused_score = (
            ECAPA_WEIGHT * ecapa_score
            + WAVLM_WEIGHT * wavlm_score
        )

        scores[person] = {
            "fused": float(fused_score),
            "ecapa": float(ecapa_score),
            "wavlm": float(wavlm_score),
        }

    return scores, q


# ============================================================
# AGGREGATE SCORES FOR ONE DIARIZED SPEAKER
# ============================================================

def robust_score(values):
    """Combine per-window scores using median + 75th percentile.

    This mixture (75% median + 25% p75) is more stable than
    a simple mean because it resists single noisy windows
    while still rewarding consistently high scores.
    """
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return -1.0

    median = float(np.median(values))
    p75 = float(np.percentile(values, 75))

    return 0.75 * median + 0.25 * p75


def identify_speaker(segment_score_rows):
    """Assign an enrolled identity to a diarization cluster.

    All four conditions must pass to declare a match:
      1. Fused score >= MATCH_THRESHOLD
      2. Margin over second-best >= MIN_MARGIN
      3. ECAPA score >= MIN_ECAPA_SCORE
      4. Usable matching windows >= MIN_VALID_SEGMENTS

    If any condition fails, the cluster is labelled UNKNOWN.
    """
    person_scores = defaultdict(lambda: defaultdict(list))

    for row in segment_score_rows:
        for person, scores in row.items():
            for key in ("fused", "ecapa", "wavlm"):
                person_scores[person][key].append(scores[key])

    ranked = []

    for person, scores in person_scores.items():
        ranked.append({
            "person": person,
            "score": robust_score(scores["fused"]),
            "ecapa": float(np.median(scores["ecapa"])),
            "wavlm": float(np.median(scores["wavlm"])),
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)

    if not ranked:
        return {
            "person": "UNKNOWN",
            "score": None,
            "margin": None,
            "windows_used": 0,
            "reason": "no usable matching windows",
            "ranked": [],
        }

    best = ranked[0]
    second_score = ranked[1]["score"] if len(ranked) > 1 else -1.0
    margin = best["score"] - second_score
    windows_used = len(segment_score_rows)

    # All conditions must pass.
    reasons = []

    if windows_used < MIN_VALID_SEGMENTS:
        reasons.append(
            f"only {windows_used} usable matching window(s)"
        )
    if best["score"] < MATCH_THRESHOLD:
        reasons.append("fused score below threshold")
    if margin < MIN_MARGIN:
        reasons.append("top candidates too close")
    if best["ecapa"] < MIN_ECAPA_SCORE:
        reasons.append("ECAPA support below threshold")

    if reasons:
        identity = "UNKNOWN"
        reason = "; ".join(reasons)
    else:
        identity = best["person"]
        reason = "all identity checks passed"

    return {
        "person": identity,
        "score": round(best["score"], 4),
        "margin": round(margin, 4),
        "windows_used": windows_used,
        "reason": reason,
        "ranked": ranked,
    }


# ============================================================
# SPEAKING-TIME HELPER
# ============================================================

def union_duration(intervals):
    """Calculate interval union so overlaps are not double-counted."""
    if not intervals:
        return 0.0

    intervals = sorted(intervals)
    merged_start, merged_end = intervals[0]
    total = 0.0

    for start, end in intervals[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
        else:
            total += merged_end - merged_start
            merged_start, merged_end = start, end

    total += merged_end - merged_start
    return total


# ============================================================
# ARGUMENT PARSING
# ============================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Speaker diarization + identification using "
            "pyannote.audio + ECAPA-TDNN + WavLM."
        ),
    )

    parser.add_argument(
        "audio",
        nargs="?",
        default=None,
        help="Path to audio file (prompted if omitted)",
    )

    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum expected number of speakers",
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum expected number of speakers",
    )

    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact number of speakers (overrides min/max)",
    )

    return parser.parse_args()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    args = parse_args()

    # Resolve audio file path.
    if args.audio:
        audio_path = Path(args.audio).expanduser()
    else:
        raw_path = input(
            "Enter audio file path: "
        ).strip().strip('"')
        audio_path = Path(raw_path).expanduser()

    if not audio_path.is_absolute():
        audio_path = BASE_DIR / audio_path

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    # Load voiceprints and models.
    voiceprints = load_voiceprints()

    print(
        f"\nLoaded {len(voiceprints)} voiceprints: "
        f"{', '.join(sorted(voiceprints))}"
    )

    diarization_pipeline = load_diarization_pipeline()
    ecapa_model, extractor, wavlm_model = load_models()

    # Load and preprocess audio.
    audio = load_audio(audio_path)
    duration = len(audio) / SR

    print(f"\nAudio file: {audio_path.name}")
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Device: {DEVICE}")
    print(
        f"Decision settings: "
        f"fused>={MATCH_THRESHOLD:.2f}, "
        f"margin>={MIN_MARGIN:.2f}, "
        f"ECAPA>={MIN_ECAPA_SCORE:.2f}, "
        f"min_segments>={MIN_VALID_SEGMENTS}"
    )
    print("Running speaker diarization...")

    # Run pyannote diarization.
    waveform = torch.from_numpy(
        np.ascontiguousarray(audio)
    ).unsqueeze(0)

    pipeline_input = {
        "waveform": waveform,
        "sample_rate": SR,
    }

    # Speaker count constraints.
    pipeline_kwargs = {}

    if args.num_speakers is not None:
        pipeline_kwargs["num_speakers"] = args.num_speakers
        print(f"  Exact speakers: {args.num_speakers}")
    else:
        if args.min_speakers is not None:
            pipeline_kwargs["min_speakers"] = args.min_speakers
            print(f"  Min speakers: {args.min_speakers}")
        if args.max_speakers is not None:
            pipeline_kwargs["max_speakers"] = args.max_speakers
            print(f"  Max speakers: {args.max_speakers}")

    diarization_result = diarization_pipeline(
        pipeline_input,
        **pipeline_kwargs,
    )

    diarization = diarization_result.speaker_diarization

    # Process each diarized speech turn.
    turns = []
    cluster_scores = defaultdict(list)

    for turn, cluster_label in diarization:
        start = max(0.0, float(turn.start))
        end = min(duration, float(turn.end))
        cluster_label = str(cluster_label)

        if end <= start:
            continue

        turns.append((start, end, cluster_label))

        start_index = int(start * SR)
        end_index = int(end * SR)
        segment = audio[start_index:end_index]
        segment_duration = len(segment) / SR

        print(
            f"\nSegment: {start:.2f}-{end:.2f}s"
            f" | {cluster_label} | {segment_duration:.2f}s"
        )

        windows = make_matching_windows(segment)

        if not windows:
            print(
                f"  Too short for matching "
                f"(<{MIN_SEGMENT_SECONDS:.1f}s); skipped."
            )
            continue

        print(f"  Matching windows: {len(windows)}")

        for window_index, window in enumerate(windows, start=1):
            result = score_window(
                window,
                voiceprints,
                ecapa_model,
                extractor,
                wavlm_model,
            )

            if result is not None:
                scores, q = result
                cluster_scores[cluster_label].append(scores)

                # Show the best candidate for this window.
                best_person = max(
                    scores,
                    key=lambda p: scores[p]["fused"],
                )
                best_fused = scores[best_person]["fused"]

                print(
                    f"    Window {window_index}: "
                    f"quality={q:.3f}, "
                    f"best={best_person} "
                    f"({best_fused:.4f})"
                )

    # Assign an identity to each diarization cluster.
    cluster_labels = sorted({
        cluster for _, _, cluster in turns
    })

    identity_by_cluster = {}

    for cluster in cluster_labels:
        identity_by_cluster[cluster] = identify_speaker(
            cluster_scores[cluster]
        )

    # Build chronological speech timeline.
    timeline = []
    sorted_turns = sorted(
        turns,
        key=lambda item: (item[0], item[1]),
    )
    cursor = 0.0

    for start, end, cluster in sorted_turns:
        decision = identity_by_cluster[cluster]

        # A gap is only a candidate pause.
        # Overlap is not a pause.
        if start - cursor >= MIN_PAUSE_SECONDS:
            timeline.append({
                "start": round(cursor, 3),
                "end": round(start, 3),
                "duration": round(start - cursor, 3),
                "event_type": "PAUSE / NO SPEECH DETECTED",
                "speaker_cluster": "",
                "identified_person": "",
                "score": None,
            })

        timeline.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "event_type": "SPEECH",
            "speaker_cluster": cluster,
            "identified_person": decision["person"],
            "score": decision["score"],
        })

        cursor = max(cursor, end)

    if duration - cursor >= MIN_PAUSE_SECONDS:
        timeline.append({
            "start": round(cursor, 3),
            "end": round(duration, 3),
            "duration": round(duration - cursor, 3),
            "event_type": "PAUSE / NO SPEECH DETECTED",
            "speaker_cluster": "",
            "identified_person": "",
            "score": None,
        })

    # Calculate union of intervals per identity.
    intervals_by_person = defaultdict(list)

    for start, end, cluster in turns:
        person = identity_by_cluster[cluster]["person"]
        intervals_by_person[person].append((start, end))

    speaking_times = {
        person: union_duration(intervals)
        for person, intervals in intervals_by_person.items()
    }

    matched_people = sorted({
        decision["person"]
        for decision in identity_by_cluster.values()
        if decision["person"] != "UNKNOWN"
    })

    # Print decisions.
    print("\n" + "=" * 60)
    print("DIARIZATION + SPEAKER IDENTIFICATION")
    print("=" * 60)
    print(f"Audio file: {audio_path.name}")
    print(f"Audio duration: {duration:.2f}s")
    print(f"Diarization clusters: {len(cluster_labels)}")
    print(
        "Matched enrolled people:",
        ", ".join(matched_people) if matched_people else "None",
    )

    print("\nSpeaker decisions:")

    for cluster in sorted(identity_by_cluster):
        decision = identity_by_cluster[cluster]
        print(
            f"  {cluster} -> {decision['person']} "
            f"| score={decision['score']} "
            f"| margin={decision['margin']} "
            f"| windows={decision['windows_used']}"
        )
        print(f"    Reason: {decision['reason']}")

        for item in decision["ranked"][:3]:
            print(
                f"    {item['person']}: "
                f"fused={item['score']:.4f}, "
                f"ECAPA={item['ecapa']:.4f}, "
                f"WavLM={item['wavlm']:.4f}"
            )

    print("\nTimeline:")

    for event in timeline:
        if event["event_type"] == "SPEECH":
            print(
                f"  {event['start']:7.2f}-"
                f"{event['end']:7.2f}s | "
                f"{event['identified_person']:<14} | "
                f"{event['duration']:.2f}s"
            )
        else:
            print(
                f"  {event['start']:7.2f}-"
                f"{event['end']:7.2f}s | "
                f"{'PAUSE':<14} | "
                f"{event['duration']:.2f}s"
            )

    print("\nTotal speaking time:")

    for person, seconds in sorted(speaking_times.items()):
        minutes = int(seconds // 60)
        secs = seconds % 60

        if minutes > 0:
            print(f"  {person}: {minutes}m {secs:.1f}s ({seconds:.2f}s)")
        else:
            print(f"  {person}: {seconds:.2f}s")

    # Console only: deliberately no CSV, JSON, or other output files.
    print("\nOutput mode: console only (no files saved).")


if __name__ == "__main__":
    main()
