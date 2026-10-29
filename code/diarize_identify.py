import os

# Force local/offline model loading where supported.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from pathlib import Path
from collections import defaultdict
import math

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from speechbrain.inference.speaker import EncoderClassifier
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Specifications, Problem, Resolution


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DIARIZATION_DIR = (
    BASE_DIR
    / "model"
    / "diarization"
    / "pyannote-speaker-diarization-community-1"
)

ECAPA_DIR = BASE_DIR / "model" / "ecapa"
WAVLM_DIR = BASE_DIR / "model" / "wavlm-base-plus-sv"
DATABASE_DIR = BASE_DIR / "database"

SAMPLE_RATE = 16000

ECAPA_WEIGHT = 0.40
WAVLM_WEIGHT = 0.60

# Requested operating thresholds.
# These require validation on held-out recordings before production use.
MATCH_THRESHOLD = 0.70
MIN_MARGIN = 0.15
MIN_ECAPA_SCORE = 0.45

# Short turns remain in the timeline but are not matched.
MIN_SEGMENT_SECONDS = 1.5

# Long turns are split into overlapping windows for identity matching.
MATCH_WINDOW_SECONDS = 3.0
MATCH_WINDOW_OVERLAP_SECONDS = 1.0
MIN_MATCH_WINDOWS = 2

# Report pauses at least this long.
MIN_PAUSE_SECONDS = 0.25


# ============================================================
# LOAD AUDIO WITHOUT TORCHCODEC
# ============================================================

def load_audio(audio_path):
    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True
    )

    # Convert stereo or multichannel audio to mono.
    audio = audio.mean(axis=1)

    if len(audio) == 0:
        raise ValueError("Audio file is empty.")

    # Resample to 16 kHz.
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), SAMPLE_RATE)

        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            int(sample_rate) // divisor
        ).astype(np.float32)

    return np.nan_to_num(audio).astype(np.float32)


# ============================================================
# LOAD ENROLLED VOICEPRINTS
# ============================================================

def load_voiceprints():
    if not DATABASE_DIR.is_dir():
        raise FileNotFoundError(
            f"Database directory not found:\n{DATABASE_DIR}"
        )

    voiceprints = {}

    for file_path in sorted(DATABASE_DIR.glob("*.pt")):
        try:
            data = torch.load(
                file_path,
                map_location="cpu",
                weights_only=True
            )

            # The current database must contain both embeddings.
            if (
                not isinstance(data, dict)
                or "ecapa" not in data
                or "wavlm" not in data
            ):
                print(
                    f"Skipping incompatible voiceprint: "
                    f"{file_path.name}"
                )
                continue

            ecapa_raw = torch.as_tensor(
                data["ecapa"], dtype=torch.float32
            ).reshape(-1)

            wavlm_raw = torch.as_tensor(
                data["wavlm"], dtype=torch.float32
            ).reshape(-1)

            # Reject invalid or zero-length embeddings.
            if (
                ecapa_raw.numel() == 0
                or wavlm_raw.numel() == 0
                or not torch.isfinite(ecapa_raw).all()
                or not torch.isfinite(wavlm_raw).all()
                or torch.linalg.vector_norm(ecapa_raw).item() < 1e-8
                or torch.linalg.vector_norm(wavlm_raw).item() < 1e-8
            ):
                print(
                    f"Skipping invalid/zero voiceprint: "
                    f"{file_path.name}"
                )
                continue

            ecapa_embedding = F.normalize(
                ecapa_raw.unsqueeze(0), dim=1
            ).squeeze(0)

            wavlm_embedding = F.normalize(
                wavlm_raw.unsqueeze(0), dim=1
            ).squeeze(0)

            voiceprints[file_path.stem] = {
                "ecapa": ecapa_embedding,
                "wavlm": wavlm_embedding
            }

        except Exception as error:
            print(f"Could not load {file_path.name}: {error}")

    if not voiceprints:
        raise FileNotFoundError(
            "No compatible ECAPA + WavLM voiceprints found in "
            f"{DATABASE_DIR}"
        )

    print("\nLoaded voiceprints:")
    for person in sorted(voiceprints):
        print(f"  - {person}")

    return voiceprints


# ============================================================
# LOAD ALL LOCAL MODELS
# ============================================================

def load_models():
    for model_dir in (
        DIARIZATION_DIR,
        ECAPA_DIR,
        WAVLM_DIR
    ):
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory missing:\n{model_dir}"
            )

    # Allow only the pyannote checkpoint classes required
    # by the locally downloaded model.
    torch.serialization.add_safe_globals([
        Specifications,
        Problem,
        Resolution
    ])

    print("\nLoading local pyannote pipeline...")

    pipeline = Pipeline.from_pretrained(str(DIARIZATION_DIR))
    pipeline.to(torch.device("cpu"))

    print("Pyannote pipeline loaded.")
    print("\nLoading ECAPA-TDNN...")

    ecapa_model = EncoderClassifier.from_hparams(
        source=str(ECAPA_DIR),
        savedir=str(ECAPA_DIR),
        run_opts={"device": "cpu"}
    )

    print("ECAPA-TDNN loaded.")
    print("\nLoading WavLM...")

    wavlm_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        str(WAVLM_DIR),
        local_files_only=True
    )

    wavlm_model = WavLMForXVector.from_pretrained(
        str(WAVLM_DIR),
        local_files_only=True
    )

    wavlm_model.eval()
    wavlm_model.cpu()

    print("WavLM loaded.")

    return (
        pipeline,
        ecapa_model,
        wavlm_processor,
        wavlm_model
    )


# ============================================================
# EXTRACT ECAPA + WAVLM EMBEDDINGS
# ============================================================

@torch.inference_mode()
def extract_embeddings(
    audio_segment,
    ecapa_model,
    wavlm_processor,
    wavlm_model
):
    audio_segment = np.asarray(
        audio_segment, dtype=np.float32
    )

    if len(audio_segment) == 0:
        raise ValueError("Cannot embed an empty segment.")

    waveform = torch.from_numpy(
        np.ascontiguousarray(audio_segment)
    ).unsqueeze(0)

    # ECAPA-TDNN embedding.
    ecapa_output = ecapa_model.encode_batch(waveform)

    ecapa_embedding = F.normalize(
        ecapa_output.reshape(1, -1), dim=1
    ).squeeze(0).cpu()

    # WavLM speaker embedding.
    inputs = wavlm_processor(
        audio_segment,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    wavlm_output = wavlm_model(inputs.input_values.cpu())

    wavlm_embedding = F.normalize(
        wavlm_output.embeddings.reshape(1, -1), dim=1
    ).squeeze(0).cpu()

    return ecapa_embedding, wavlm_embedding


def make_matching_windows(segment):
    """Create overlapping windows from a diarized speech turn."""
    segment = np.asarray(segment, dtype=np.float32)

    min_samples = int(MIN_SEGMENT_SECONDS * SAMPLE_RATE)
    window_samples = int(MATCH_WINDOW_SECONDS * SAMPLE_RATE)
    step_samples = int(
        (MATCH_WINDOW_SECONDS - MATCH_WINDOW_OVERLAP_SECONDS)
        * SAMPLE_RATE
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
# COMPARE SEGMENT WITH ENROLLED VOICEPRINTS
# ============================================================

def calculate_scores(
    ecapa_embedding,
    wavlm_embedding,
    voiceprints
):
    results = {}

    for person, voiceprint in voiceprints.items():
        ecapa_score = F.cosine_similarity(
            ecapa_embedding.unsqueeze(0),
            voiceprint["ecapa"].unsqueeze(0)
        ).item()

        wavlm_score = F.cosine_similarity(
            wavlm_embedding.unsqueeze(0),
            voiceprint["wavlm"].unsqueeze(0)
        ).item()

        fused_score = (
            ECAPA_WEIGHT * ecapa_score
            + WAVLM_WEIGHT * wavlm_score
        )

        results[person] = {
            "fused": float(fused_score),
            "ecapa": float(ecapa_score),
            "wavlm": float(wavlm_score)
        }

    return results


# ============================================================
# AGGREGATE SCORES FOR ONE DIARIZED SPEAKER
# ============================================================

def robust_score(values):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return -1.0

    median = np.median(values)
    percentile_75 = np.percentile(values, 75)

    return float(0.75 * median + 0.25 * percentile_75)


def identify_speaker(segment_score_rows):
    person_scores = defaultdict(lambda: defaultdict(list))

    for row in segment_score_rows:
        for person, scores in row.items():
            for score_name in ("fused", "ecapa", "wavlm"):
                person_scores[person][score_name].append(
                    scores[score_name]
                )

    ranked = []

    for person, scores in person_scores.items():
        ranked.append({
            "person": person,
            "score": robust_score(scores["fused"]),
            "ecapa": float(np.median(scores["ecapa"])),
            "wavlm": float(np.median(scores["wavlm"]))
        })

    ranked.sort(key=lambda item: item["score"], reverse=True)

    if not ranked:
        return {
            "person": "UNKNOWN",
            "score": None,
            "margin": None,
            "windows_used": 0,
            "reason": "no usable matching windows",
            "ranked": []
        }

    best = ranked[0]
    second_score = ranked[1]["score"] if len(ranked) > 1 else -1.0
    margin = best["score"] - second_score
    windows_used = len(segment_score_rows)

    # All conditions must pass to assign an enrolled identity.
    if (
        windows_used >= MIN_MATCH_WINDOWS
        and best["score"] >= MATCH_THRESHOLD
        and margin >= MIN_MARGIN
        and best["ecapa"] >= MIN_ECAPA_SCORE
    ):
        identity = best["person"]
        reason = "all identity checks passed"
    else:
        identity = "UNKNOWN"
        reasons = []

        if windows_used < MIN_MATCH_WINDOWS:
            reasons.append(
                f"only {windows_used} usable matching window(s)"
            )
        if best["score"] < MATCH_THRESHOLD:
            reasons.append("fused score below threshold")
        if margin < MIN_MARGIN:
            reasons.append("top candidates too close")
        if best["ecapa"] < MIN_ECAPA_SCORE:
            reasons.append("ECAPA support below threshold")

        reason = "; ".join(reasons) or "identity checks failed"

    return {
        "person": identity,
        "score": round(best["score"], 4),
        "margin": round(margin, 4),
        "windows_used": windows_used,
        "reason": reason,
        "ranked": ranked
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
# MAIN PIPELINE
# ============================================================

def main():
    raw_path = input("Enter audio file path: ").strip().strip('"')

    audio_path = Path(raw_path).expanduser()

    if not audio_path.is_absolute():
        audio_path = BASE_DIR / audio_path

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found:\n{audio_path}"
        )

    voiceprints = load_voiceprints()

    (
        pipeline,
        ecapa_model,
        wavlm_processor,
        wavlm_model
    ) = load_models()

    audio = load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE

    print(f"\nAudio duration: {duration:.2f} seconds")
    print(
        "Decision settings: "
        f"fused>={MATCH_THRESHOLD:.2f}, "
        f"margin>={MIN_MARGIN:.2f}, "
        f"ECAPA>={MIN_ECAPA_SCORE:.2f}, "
        f"matching windows>={MIN_MATCH_WINDOWS}"
    )
    print("Running speaker diarization...")

    waveform = torch.from_numpy(
        np.ascontiguousarray(audio)
    ).unsqueeze(0)

    diarization_result = pipeline({
        "waveform": waveform,
        "sample_rate": SAMPLE_RATE
    })

    diarization = diarization_result.speaker_diarization

    turns = []
    cluster_scores = defaultdict(list)

    # Process each diarized speech turn.
    for turn, cluster_label in diarization:
        start = max(0.0, float(turn.start))
        end = min(duration, float(turn.end))
        cluster_label = str(cluster_label)

        if end <= start:
            continue

        turns.append((start, end, cluster_label))

        start_index = int(start * SAMPLE_RATE)
        end_index = int(end * SAMPLE_RATE)
        segment = audio[start_index:end_index]
        segment_duration = len(segment) / SAMPLE_RATE

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
            try:
                ecapa_embedding, wavlm_embedding = extract_embeddings(
                    window,
                    ecapa_model,
                    wavlm_processor,
                    wavlm_model
                )

                scores = calculate_scores(
                    ecapa_embedding,
                    wavlm_embedding,
                    voiceprints
                )

                cluster_scores[cluster_label].append(scores)

            except Exception as error:
                print(
                    f"  Window {window_index} embedding failed: "
                    f"{type(error).__name__}: {error}"
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
    sorted_turns = sorted(turns, key=lambda item: (item[0], item[1]))
    cursor = 0.0

    for start, end, cluster in sorted_turns:
        decision = identity_by_cluster[cluster]

        # A gap is only a candidate pause. Overlap is not a pause.
        if start - cursor >= MIN_PAUSE_SECONDS:
            timeline.append({
                "start": round(cursor, 3),
                "end": round(start, 3),
                "duration": round(start - cursor, 3),
                "event_type": "PAUSE / NO SPEECH DETECTED",
                "speaker_cluster": "",
                "identified_person": "",
                "score": None
            })

        timeline.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "event_type": "SPEECH",
            "speaker_cluster": cluster,
            "identified_person": decision["person"],
            "score": decision["score"]
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
            "score": None
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
        ", ".join(matched_people) if matched_people else "None"
    )

    print("\nSpeaker decisions:")

    for cluster, decision in identity_by_cluster.items():
        print(
            f"{cluster} -> {decision['person']} "
            f"| score={decision['score']} "
            f"| margin={decision['margin']} "
            f"| windows={decision['windows_used']}"
        )
        print(f"    Decision reason: {decision['reason']}")

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
                f"{event['start']:7.2f}-"
                f"{event['end']:7.2f}s | "
                f"{event['identified_person']} | "
                f"{event['duration']:.2f}s"
            )
        else:
            print(
                f"{event['start']:7.2f}-"
                f"{event['end']:7.2f}s | "
                f"PAUSE / NO SPEECH DETECTED | "
                f"{event['duration']:.2f}s"
            )

    print("\nTotal speaking time:")

    for person, seconds in sorted(speaking_times.items()):
        print(f"  {person}: {seconds:.2f} seconds")

    # Console only: deliberately no CSV, JSON, or other output files.
    print("\nOutput mode: console only (no files saved).")


if __name__ == "__main__":
    main()