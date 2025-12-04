#!/usr/bin/env python3
import os
import glob
import math
import numpy as np
import pandas as pd
import librosa
from typing import Tuple
from pathlib import Path


# -----------------------------
# Config
# -----------------------------
SR = 16000      	# Target sample rate
MAXSECONDS = 60	# Safety cap: analyze up to first 60s (set None for full file)
HOPLENGTH = 512
NMFCC = 20
ROLLPERCENT = 0.85

# -----------------------------
# Helpers
# -----------------------------


def load_audio(path: str, sr: int = 16000, max_seconds: float | None = 60.0) -> Tuple[np.ndarray, int]:
    """
    Load audio as mono, normalized float32. Optionally trim leading/trailing silence and cap length.

    Parameters
    ----------
    path : str
        Path to the audio file (mp3, flac, wav, etc.).
    sr : int, default=16000
        Target sampling rate for resampling.
    max_seconds : float | None, default=60.0
        If set, cap the audio to this many seconds. Use None to keep full length.

    Returns
    -------
    y : np.ndarray
        Mono audio signal in float32, normalized to [-1, 1].
    sr_out : int
        The sampling rate of the returned signal (should equal `sr`).
    """
    # 1) Load as mono and resample
    y, sr_out = librosa.load(path, sr=sr, mono=True)

    # Ensure float32 dtype
    y = y.astype(np.float32, copy=False)

    # 2) Normalize amplitude (avoid division by zero)
    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0.0:
        y = y / peak

    # 3) Trim leading/trailing silence (30 dB below max by default)
    # returns (y_trimmed, [start_sample, end_sample])
    y, _ = librosa.effects.trim(y, top_db=30)

    # 4) Cap length if requested
    if max_seconds is not None and max_seconds > 0:
        max_samples = int(max_seconds * sr_out)
        if y.size > max_samples:
            y = y[:max_samples]

    return y, sr_out


def safemeanstd(arr, axis=None):
    """Return a (mean, std) tuple; works if arr empty by returning NaNs."""
    if arr.size == 0:
        return np.nan, np.nan
    return float(np.mean(arr, axis=axis)), float(np.std(arr, axis=axis))

def summarize_vector(vec, prefix):
	"""Return a dict with mean and std for a 1D vector."""
	m, s = safemeanstd(np.asarray(vec))
	return {f"{prefix}mean": m, f"{prefix}std": s}


def mfcc_stats(mfcc, name):
    """Given (ncoeffs, nframes), return mean/std per coefficient as a dict."""
    feats = {}
    for i in range(mfcc.shape[0]):
        feats[f"{name}{i+1}mean"] = float(np.mean(mfcc[i]))
        feats[f"{name}{i+1}std"]  = float(np.std(mfcc[i]))
    return feats



def pitch_features(y, sr):
    """Estimate pitch using pyin (preferred) or yin fallback; return stats."""
    feats = {
        "voicedratio": np.nan,
        "f0mean": np.nan,
        "f0std": np.nan,
        "f0min": np.nan,
        "f0max": np.nan,
        "f0range": np.nan,
    }
    try:
        # pyin outputs Hz with NaNs for unvoiced frames
        f0 = librosa.pyin(y, fmin=50, fmax=400, sr=sr, hoplength=HOPLENGTH)
        f0 = np.array(f0)
    except Exception:
        try:
            # yin returns strictly positive values; treat zeros/unvoiced heuristically
            f0 = librosa.yin(y, fmin=50, fmax=400, sr=sr, hoplength=HOPLENGTH)
        except Exception:
            f0 = np.array([])

    if f0.size == 0:
        return feats

    # Handle NaNs (pyin) and non-voiced frames
    valid = ~np.isnan(f0)
    voiced = f0[valid]
    feats["voicedratio"] = float(np.mean(valid)) if f0.size > 0 else np.nan
    if voiced.size > 0:
        feats["f0mean"] = float(np.mean(voiced))
        feats["f0std"]  = float(np.std(voiced))
        feats["f0min"]  = float(np.min(voiced))
        feats["f0max"]  = float(np.max(voiced))
        feats["f0range"] = feats["f0max"] - feats["f0min"]
    return feats


def harmonic_energy_ratio(y):
    """Compute ratio of harmonic energy to total using HPSS."""
    try:
        yh, yp = librosa.effects.hpss(y)
        num = np.sum(yh ** 2)
        den = np.sum(yh ** 2) + np.sum(yp ** 2) + 1e-12  # avoid division by zero
        return float(num / den)
    except Exception:
        return np.nan



import librosa
import numpy as np

def extract_features_from_wave(y, sr):
    """Compute a comprehensive set of features for one audio signal."""
    feats = {}

    # Dynamically adjust n_fft and hop_length
    n_fft = min(2048, len(y))                # Ensure n_fft <= signal length
    hop_length = max(64, n_fft // 4)         # Keep hop_length reasonable

    # Duration
    feats["duration_sec"] = float(len(y) / sr)

    # Energy & Zero-Crossing Rate
    rms = librosa.feature.rms(y=y, hop_length=hop_length).squeeze()
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length).squeeze()
    feats.update(summarize_vector(rms, "rms_"))
    feats.update(summarize_vector(zcr, "zcr_"))

    # Spectral basic features
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).squeeze()
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr).squeeze()
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85).squeeze()
    feats.update(summarize_vector(centroid, "spec_centroid_"))
    feats.update(summarize_vector(bandwidth, "spec_bandwidth_"))
    feats.update(summarize_vector(rolloff, "spec_rolloff_"))

    # Spectral contrast
    try:
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        for i in range(contrast.shape[0]):
            feats[f"spec_contrast{i+1}_mean"] = float(np.mean(contrast[i]))
            feats[f"spec_contrast{i+1}_std"] = float(np.std(contrast[i]))
    except Exception:
        pass

    # Chroma features
    try:
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        for i in range(chroma.shape[0]):
            feats[f"chroma{i+1}_mean"] = float(np.mean(chroma[i]))
            feats[f"chroma{i+1}_std"] = float(np.std(chroma[i]))
    except Exception:
        pass

    # Tonnetz (harmonic features)
    try:
        y_harm = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        for i in range(tonnetz.shape[0]):
            feats[f"tonnetz{i+1}_mean"] = float(np.mean(tonnetz[i]))
            feats[f"tonnetz{i+1}_std"] = float(np.std(tonnetz[i]))
    except Exception:
        pass

    # MFCCs + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
    feats.update(mfcc_stats(mfcc, "mfcc"))
    try:
        mfcc_delta = librosa.feature.delta(mfcc)
        feats.update(mfcc_stats(mfcc_delta, "mfcc_delta"))
        mfcc_dd = librosa.feature.delta(mfcc, order=2)
        feats.update(mfcc_stats(mfcc_dd, "mfcc_delta2"))
    except Exception:
        pass

    # Pitch/prosody
    feats.update(pitch_features(y, sr))

    # Harmonic-percussive energy ratio (skip if too short)
    if len(y) >= 512:
        feats["harmonic_energy_ratio"] = harmonic_energy_ratio(y)
    else:
        feats["harmonic_energy_ratio"] = np.nan

    return feats


def extractfeaturesforfile(path, label):
    """Top-level wrapper: load -> extract -> return dict with label & metadata."""
    y, sr = load_audio(path)
    feats = extract_features_from_wave(y, sr)
    feats["path"] = path
    feats["filename"] = os.path.basename(path)
    feats["label"] = label  # 'ai' or 'human', or 1/0
    feats["ext"] = os.path.splitext(path)[1].lower()
    return feats


# -----------------------------
# Main runner
# -----------------------------


def process_dirs(out_csv="features.csv"):
    # Base directory = folder where this script lives
    base_dir = Path(__file__).resolve().parent

    # Paths to your AI and Human folders
    ai_dir = base_dir / "AI Voices"
    human_dir = base_dir / "Human Voices"

    # Output CSV inside the project folder
    out_path = base_dir / out_csv

    # Collect files
    ai_files = []
    for ext in ("*.mp3", "*.wav", "*.flac"):
        ai_files += glob.glob(str(ai_dir / ext))

    human_files = []
    for ext in ("*.mp3", "*.wav", "*.flac"):
        human_files += glob.glob(str(human_dir / ext))

    print(f"[INFO] Found {len(ai_files)} AI files and {len(human_files)} human files.")

    rows = []

    # Extract features for AI voices
    for path in ai_files:
        try:
            rows.append(extractfeaturesforfile(path, label="ai"))
        except Exception as e:
            print(f"[WARN] Failed on AI file {path}: {e}")

    # Extract features for Human voices
    for path in human_files:
        try:
            rows.append(extractfeaturesforfile(path, label="human"))
        except Exception as e:
            print(f"[WARN] Failed on human file {path}: {e}")

    # Build DataFrame
    df = pd.DataFrame(rows)

    if df.empty:
        print("[ERROR] No features extracted.")
        return

    # Order columns
    front = ["label", "filename", "path", "ext", "duration_sec"]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]

    # Save CSV
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path} with shape {df.shape}")

if __name__ == "__main__":
    process_dirs()
