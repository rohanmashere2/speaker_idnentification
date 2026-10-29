"""Shared offline audio/model utilities for ECAPA-TDNN + WavLM speaker ID."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from pathlib import Path
from math import gcd
import warnings

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt, resample_poly

try:
    import noisereduce as nr
except Exception:
    nr = None

try:
    import webrtcvad
except Exception:
    webrtcvad = None

from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, WavLMForXVector


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ECAPA_DIR = BASE_DIR / "model" / "ecapa"
WAVLM_DIR = BASE_DIR / "model" / "wavlm-base-plus-sv"
DATABASE_DIR = BASE_DIR / "database"

SR = 16000

SEGMENT_SECONDS = 3.0
OVERLAP = 0.50
MIN_SEGMENT_SECONDS = 1.5

# Quality gates — relaxed so noisy audio is weighted down,
# never discarded entirely.
MIN_RMS_DBFS = -55.0
MIN_SPEECH_RATIO = 0.10

# Fusion weights — WavLM is more noise-robust due to
# self-supervised pre-training with masked denoising.
ECAPA_WEIGHT = 0.35
WAVLM_WEIGHT = 0.65

USE_DENOISING = True
NOISE_REDUCTION_STRENGTH = 0.35
MIN_RAW_CLEAN_COSINE = 0.40

# Identity decision thresholds (research-calibrated).
# ECAPA-TDNN EER is ~0.25 cosine; WavLM EER is ~0.42.
MATCH_THRESHOLD = 0.50
MIN_MARGIN = 0.08
MIN_ECAPA_SCORE = 0.28
MIN_VALID_SEGMENTS = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

warnings.filterwarnings("ignore", message=".*symlink.*")
torch.set_grad_enabled(False)


# ============================================================
# EMBEDDING UTILITIES
# ============================================================

def normalize(v):
    """Convert an embedding to a finite unit-length vector."""
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))

    if not np.isfinite(n) or n < 1e-10:
        raise ValueError("Invalid or zero-length embedding")

    return v / n


def cosine(a, b):
    """Cosine similarity between two embeddings."""
    return float(np.dot(normalize(a), normalize(b)))


# ============================================================
# AUDIO LOADING AND PREPROCESSING
# ============================================================

def load_audio(path):
    """Load audio, convert to mono, resample to 16 kHz, and filter."""
    x, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )

    if x.ndim == 2:
        x = x.mean(axis=1)

    if x.size == 0:
        raise ValueError("Audio file is empty")

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(
            x,
            SR // g,
            sr // g,
        ).astype(np.float32)

    x = x - float(np.mean(x))

    if len(x) > int(SR * 0.15):
        try:
            sos = butter(
                4,
                [80, 7600],
                btype="bandpass",
                fs=SR,
                output="sos",
            )
            x = sosfiltfilt(sos, x).astype(np.float32)
        except Exception:
            pass

    return np.clip(x, -1.0, 1.0).astype(np.float32)


def denoise(x):
    """Apply conservative noise reduction when noisereduce is available.

    Denoising cannot reliably separate overlapping speakers.
    Strength is deliberately low (0.35) — WavLM already performs
    implicit denoising through its self-supervised pretraining.
    """
    if (
        not USE_DENOISING
        or nr is None
        or len(x) < SR // 2
    ):
        return x.copy()

    try:
        y = nr.reduce_noise(
            y=x,
            sr=SR,
            stationary=False,
            prop_decrease=NOISE_REDUCTION_STRENGTH,
            n_fft=512,
            win_length=400,
            hop_length=160,
        )

        y = np.asarray(y, dtype=np.float32)

        if y.shape != x.shape or not np.all(np.isfinite(y)):
            return x.copy()

        return np.clip(y, -1.0, 1.0)

    except Exception as e:
        print(
            f"  Denoising skipped: "
            f"{type(e).__name__}: {e}"
        )
        return x.copy()


# ============================================================
# SPEECH AND AUDIO QUALITY
# ============================================================

def speech_ratio(x):
    """Estimate the proportion of frames containing speech."""
    frame = int(SR * 0.03)
    n = len(x) // frame * frame

    if n < frame:
        return 0.0

    frames = x[:n].reshape(-1, frame)

    if webrtcvad is not None:
        try:
            vad = webrtcvad.Vad(2)
            count = 0

            for f in frames:
                pcm = (
                    np.clip(f, -1, 1) * 32767
                ).astype("<i2").tobytes()

                count += int(vad.is_speech(pcm, SR))

            return count / len(frames)

        except Exception:
            pass

    rms = np.sqrt(
        np.mean(frames * frames, axis=1) + 1e-12
    )

    db = 20 * np.log10(
        np.maximum(rms, 1e-8)
    )

    threshold = max(
        -50.0,
        float(np.percentile(db, 90)) - 25.0,
    )

    return float(np.mean(db > threshold))


def quality(x):
    """Return a heuristic audio-quality weight from 0.05 to 1.0.

    Noisy or quiet segments receive a low weight but are never
    discarded entirely.  The downstream pipeline uses quality as
    a weighting factor so every segment contributes.
    """
    MIN_QUALITY = 0.05

    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    db = 20 * np.log10(max(rms, 1e-8))

    ratio = speech_ratio(x)
    clipping = float(np.mean(np.abs(x) >= 0.999))

    # Continuous scoring — no hard cutoffs.
    # Level: maps -55 dB → 0.10, -26 dB → 1.0
    level_score = float(np.clip((db + 55) / 29, 0.10, 1.0))

    # Speech presence: maps 0 → 0.10, 60%+ → 1.0
    speech_score = float(np.clip(ratio / 0.60, 0.10, 1.0))

    # Clipping penalty.
    clipping_score = max(0.10, 1.0 - 8.0 * clipping)

    score = level_score * speech_score * clipping_score

    return max(MIN_QUALITY, float(score))


# ============================================================
# AUDIO SEGMENTATION
# ============================================================

def segments(x):
    """Split audio into overlapping segments."""
    length = int(SEGMENT_SECONDS * SR)
    minimum = int(MIN_SEGMENT_SECONDS * SR)

    if len(x) < minimum:
        return []

    if len(x) <= length:
        return [x]

    step = max(
        1,
        int(length * (1 - OVERLAP)),
    )

    out = []

    for start in range(0, len(x) - minimum + 1, step):
        chunk = x[start:min(start + length, len(x))]

        if len(chunk) >= minimum:
            out.append(chunk)

        if start + length >= len(x):
            break

    return out


def model_audio(x):
    """Apply consistent RMS normalization before model inference."""
    x = np.asarray(x, dtype=np.float32)

    rms = float(np.sqrt(np.mean(x * x) + 1e-12))

    if rms > 1e-6:
        x = x * (0.10 / rms)

    return np.clip(x, -1, 1).astype(np.float32)


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    """Load ECAPA-TDNN and WavLM from local folders only."""
    for path in (ECAPA_DIR, WAVLM_DIR):
        if not path.exists():
            raise FileNotFoundError(
                f"Local model folder not found: {path}"
            )

    print(f"Loading ECAPA-TDNN from {ECAPA_DIR}")

    ecapa = EncoderClassifier.from_hparams(
        source=str(ECAPA_DIR),
        savedir=str(ECAPA_DIR),
        run_opts={"device": DEVICE},
    )

    print(f"Loading WavLM from {WAVLM_DIR}")

    extractor = AutoFeatureExtractor.from_pretrained(
        str(WAVLM_DIR),
        local_files_only=True,
    )

    wavlm = WavLMForXVector.from_pretrained(
        str(WAVLM_DIR),
        local_files_only=True,
    ).to(DEVICE).eval()

    return ecapa, extractor, wavlm


# ============================================================
# SPEAKER EMBEDDINGS
# ============================================================

def ecapa_embedding(model, x):
    """Generate an ECAPA-TDNN speaker embedding."""
    tensor = torch.from_numpy(
        model_audio(x)
    ).float().unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        out = model.encode_batch(tensor)

    return normalize(
        out.detach().cpu().numpy().reshape(-1)
    )


def wavlm_embedding(model, extractor, x):
    """Generate a WavLM speaker embedding."""
    inputs = extractor(
        model_audio(x),
        sampling_rate=SR,
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        out = model(**inputs)

    return normalize(
        out.embeddings.detach().cpu().numpy().reshape(-1)
    )


def combine_versions(raw, clean):
    """Combine original and denoised embeddings when they agree."""
    agreement = cosine(raw, clean)

    if agreement >= MIN_RAW_CLEAN_COSINE:
        combined = (
            0.70 * normalize(raw)
            + 0.30 * normalize(clean)
        )

        return normalize(combined), agreement

    # Prefer original audio if denoising changes the embedding
    # substantially.
    return normalize(raw), agreement


# ============================================================
# VOICEPRINT AGGREGATION
# ============================================================

def weighted_centroid(vectors, weights):
    """Build a quality-weighted normalized speaker centroid."""
    if not vectors:
        raise ValueError("No usable embeddings")

    mat = np.stack([
        normalize(v) for v in vectors
    ])

    w = np.maximum(
        np.asarray(weights, dtype=np.float32),
        1e-4,
    )

    centroid = np.sum(
        mat * w[:, None],
        axis=0,
    ) / np.sum(w)

    return normalize(centroid)


def reject_outliers_mad(vectors, threshold=3.0):
    """Remove outlier embeddings using Median Absolute Deviation.

    This is the gold-standard approach for enrollment cleanup:
    robust, nearly parameter-free, and keeps the vast majority
    of recordings.  Only truly anomalous segments are removed.

    Args:
        vectors: list of L2-normalized embedding arrays.
        threshold: modified z-score cutoff (3.0 = conservative,
                   2.5 = aggressive).

    Returns:
        (clean_vectors, mask) — filtered list and boolean mask.
        If rejection would discard everything, originals are
        returned unchanged.
    """
    if len(vectors) < 3:
        return list(vectors), np.ones(len(vectors), dtype=bool)

    mat = np.stack([normalize(v) for v in vectors])

    # Initial centroid.
    centroid = np.mean(mat, axis=0)
    norm = float(np.linalg.norm(centroid))

    if norm < 1e-10:
        return list(vectors), np.ones(len(vectors), dtype=bool)

    centroid = centroid / norm

    # Cosine distance of each embedding to the centroid.
    cos_sim = mat @ centroid
    distances = 1.0 - cos_sim

    # MAD-based outlier detection.
    median_dist = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_dist)))

    if mad < 1e-10:
        return list(vectors), np.ones(len(vectors), dtype=bool)

    # Modified z-scores (0.6745 makes MAD consistent with
    # standard deviation for normal distributions).
    modified_z = 0.6745 * (distances - median_dist) / mad
    mask = np.abs(modified_z) <= threshold

    if not np.any(mask):
        # Never discard everything.
        return list(vectors), np.ones(len(vectors), dtype=bool)

    clean = [v for v, m in zip(vectors, mask) if m]
    return clean, mask


# ============================================================
# LOAD ENROLLED VOICEPRINTS
# ============================================================

def load_voiceprints():
    """Load all enrolled voiceprints from the database directory.

    Returns a dict mapping person names to dicts with
    'ecapa' and 'wavlm' numpy embeddings.
    """
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
# PROCESS AN ENROLLMENT RECORDING
# ============================================================

def process_recording(path, ecapa, extractor, wavlm):
    """Extract combined ECAPA and WavLM voiceprints for one file.

    Every segment is used regardless of noise level — quality()
    returns a minimum weight of 0.05, so noisy segments contribute
    with lower influence rather than being discarded.
    """
    raw = load_audio(path)
    clean = denoise(raw)

    raw_parts = segments(raw)
    clean_parts = segments(clean)

    count = min(
        len(raw_parts),
        len(clean_parts),
    )

    ecapa_vectors = []
    wavlm_vectors = []
    weights = []

    for i in range(count):
        original_segment = raw_parts[i]
        clean_segment = clean_parts[i]

        q = quality(original_segment)

        try:
            ecapa_raw = ecapa_embedding(
                ecapa,
                original_segment,
            )
            ecapa_clean = ecapa_embedding(
                ecapa,
                clean_segment,
            )

            wavlm_raw = wavlm_embedding(
                wavlm,
                extractor,
                original_segment,
            )
            wavlm_clean = wavlm_embedding(
                wavlm,
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

            agreement_weight = float(
                np.clip(
                    (ecapa_agreement + wavlm_agreement) / 2,
                    0.35,
                    1.0,
                )
            )

            weight = q * agreement_weight

            ecapa_vectors.append(ecapa_vector)
            wavlm_vectors.append(wavlm_vector)
            weights.append(weight)

        except Exception as exc:
            print(
                f"    Segment {i + 1} failed: "
                f"{type(exc).__name__}: {exc}"
            )

    if not weights:
        return None

    return {
        "ecapa": weighted_centroid(
            ecapa_vectors,
            weights,
        ),
        "wavlm": weighted_centroid(
            wavlm_vectors,
            weights,
        ),
        "quality": float(np.mean(weights)),
        "segments_used": len(weights),
        "segments_total": count,
    }
