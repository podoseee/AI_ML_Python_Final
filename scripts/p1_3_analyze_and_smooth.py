# scripts/p1_3_analyze_and_smooth.py
"""
Assignment 3 - analysis of raw BlazePose 3D keypoints, provided post-processing
(ground correction), and bonus smoothing (gap-fill + Savitzky-Golay).

Usage (from project root):
    python scripts/p1_3_analyze_and_smooth.py --fps 30
Outputs:
    data/processed/<name>_smooth.npy / _smooth.csv
    results/<name>_metrics.csv
    results/<name>_trajectory.png, <name>_jitter.png, <name>_bones.png
"""
import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw_keypoints")
OUT_DIR = os.path.join(ROOT, "data", "processed")
RES_DIR = os.path.join(ROOT, "results")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

# BlazePose indices
JOINTS = {"L_wrist": 15, "R_wrist": 16, "L_ankle": 27, "R_ankle": 28}
LEFT_FOOT, RIGHT_FOOT = 31, 32
BONES = {  # (a, b)
    "L_upper_arm": (11, 13), "L_forearm": (13, 15),
    "R_upper_arm": (12, 14), "R_forearm": (14, 16),
    "L_thigh": (23, 25), "L_shin": (25, 27),
    "R_thigh": (24, 26), "R_shin": (26, 28),
}


# ---------------------------------------------------------------- loading
def load_raw(npz_path):
    """Return seq (T,33,3), vis (T,33), n_missing. Missing frames -> NaN
    (the provided loader silently drops them and shifts the time axis)."""
    data = np.load(npz_path, allow_pickle=True)["data"]
    df = pd.DataFrame(data, columns=["frame", "landmark", "x", "y", "z", "visibility"])
    df = df.astype({"frame": int, "landmark": int})
    T = int(df["frame"].max()) + 1
    seq = np.full((T, 33, 3), np.nan)
    vis = np.zeros((T, 33))
    f, l = df["frame"].values, df["landmark"].values
    seq[f, l] = df[["x", "y", "z"]].values
    vis[f, l] = df["visibility"].values
    detected = ~np.isnan(seq[:, 0, 0])
    return seq, vis, int((~detected).sum())


# ----------------------------------------------------- provided method
def provided_ground_correction(raw_seq):
    """Same logic as p1_2_post_processing.create_height_corrected_target
    (per-frame foot-y subtraction + clamp). NaN frames are filled first."""
    seq = fill_gaps(raw_seq)
    foot_y = np.minimum(seq[:, LEFT_FOOT, 1], seq[:, RIGHT_FOOT, 1])
    seq[:, :, 1] -= foot_y[:, None]
    seq[:, :, 1] = np.maximum(seq[:, :, 1], 0)
    return seq


# ------------------------------------------------------ bonus methods
def fill_gaps(seq):
    """Linear interpolation over NaN frames per coordinate."""
    out = seq.copy()
    T = out.shape[0]
    t = np.arange(T)
    for j in range(out.shape[1]):
        for c in range(out.shape[2]):
            y = out[:, j, c]
            ok = ~np.isnan(y)
            if ok.sum() == 0:
                continue
            out[:, j, c] = np.interp(t, t[ok], y[ok])
    return out


def remove_outliers(seq, k=6.0):
    """Mark per-joint speed spikes (> median + k*MAD) as NaN, then re-interpolate."""
    v = np.linalg.norm(np.diff(seq, axis=0), axis=2)          # (T-1,33)
    med = np.median(v, axis=0)
    mad = np.median(np.abs(v - med), axis=0) + 1e-9
    spike = v > med + k * mad
    bad = np.zeros(seq.shape[:2], bool)
    bad[1:] |= spike
    out = seq.copy()
    out[bad] = np.nan
    return fill_gaps(out)


def smooth(seq, fps, window_sec=0.25, poly=2):
    """Gap-fill -> outlier removal -> Savitzky-Golay on every coordinate."""
    x = remove_outliers(fill_gaps(seq))
    w = max(int(window_sec * fps) // 2 * 2 + 1, poly + 2 + (poly % 2 == 0))
    w = min(w, len(x) // 2 * 2 - 1)
    return savgol_filter(x, window_length=w, polyorder=poly, axis=0)


def ground_global(seq):
    """Better ground correction: ONE offset for the whole clip, so real
    vertical motion (jumps) is preserved."""
    out = seq.copy()
    out[:, :, 1] -= np.min(out[:, [LEFT_FOOT, RIGHT_FOOT], 1])
    return out


# ------------------------------------------------------------- metrics
def metrics(seq, fps, vis=None, n_missing=0):
    seq = fill_gaps(seq)
    acc = np.diff(seq, n=2, axis=0) * fps ** 2                 # m/s^2
    vel = np.diff(seq, axis=0) * fps
    speed = np.linalg.norm(vel, axis=2)
    med = np.median(speed, axis=0)
    mad = np.median(np.abs(speed - med), axis=0) + 1e-9
    outliers = int((speed > med + 6 * mad).sum())
    bone_cv = []
    for a, b in BONES.values():
        L = np.linalg.norm(seq[:, a] - seq[:, b], axis=1)
        bone_cv.append(L.std() / (L.mean() + 1e-9))
    foot_min = np.minimum(seq[:, LEFT_FOOT, 1], seq[:, RIGHT_FOOT, 1])
    return {
        "mean_accel (jitter, m/s^2)": float(np.linalg.norm(acc, axis=2).mean()),
        "wrist+ankle_accel": float(np.linalg.norm(acc[:, list(JOINTS.values())], axis=2).mean()),
        "speed_outliers": outliers,
        "bone_len_CV (mean)": float(np.mean(bone_cv)),
        "foot_height_std (m)": float(foot_min.std()),
        "missing_frames": n_missing,
        "low_vis_ratio(<0.5)": float((vis < 0.5).mean()) if vis is not None else np.nan,
    }


# --------------------------------------------------------------- plots
def plot_trajectories(name, versions, fps):
    fig, axes = plt.subplots(len(JOINTS), 3, figsize=(14, 2.6 * len(JOINTS)), sharex=True)
    for r, (jn, ji) in enumerate(JOINTS.items()):
        for c, ax_name in enumerate("xyz"):
            ax = axes[r, c]
            for label, s, style in versions:
                t = np.arange(len(s)) / fps
                ax.plot(t, s[:, ji, c], style["fmt"], lw=style["lw"], label=label, alpha=style["a"])
            ax.set_title(f"{jn} - {ax_name}", fontsize=9)
    axes[0, 0].legend(fontsize=7)
    axes[-1, 1].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, f"{name}_trajectory.png"), dpi=130)
    plt.close()


def plot_jitter(name, versions, fps):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for label, s, style in versions:
        a = np.linalg.norm(np.diff(fill_gaps(s), n=2, axis=0), axis=2).mean(axis=1) * fps ** 2
        ax.plot(np.arange(len(a)) / fps, a, style["fmt"], lw=style["lw"], label=label, alpha=style["a"])
    ax.set_xlabel("time (s)"); ax.set_ylabel("mean accel (m/s^2)")
    ax.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, f"{name}_jitter.png"), dpi=130); plt.close()


def plot_bones(name, versions, fps):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True)
    for ax, (bn, (a, b)) in zip(axes, [("L_forearm", BONES["L_forearm"]), ("R_shin", BONES["R_shin"])]):
        for label, s, style in versions:
            L = np.linalg.norm(fill_gaps(s)[:, a] - fill_gaps(s)[:, b], axis=1)
            ax.plot(np.arange(len(L)) / fps, L, style["fmt"], lw=style["lw"], label=label, alpha=style["a"])
        ax.set_title(f"{bn} length (m)", fontsize=9)
    axes[0].legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(RES_DIR, f"{name}_bones.png"), dpi=130); plt.close()


def export_csv(motion, path):
    T, N, _ = motion.shape
    t, l = np.meshgrid(np.arange(T), np.arange(N), indexing="ij")
    df = pd.DataFrame({"frame": t.ravel(), "landmark": l.ravel(),
                       "x": motion[:, :, 0].ravel(), "y": motion[:, :, 1].ravel(),
                       "z": motion[:, :, 2].ravel(), "visibility": 1.0})
    df.to_csv(path, index=False)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=30.0, help="video fps (not stored in npz!)")
    ap.add_argument("--window", type=float, default=0.25, help="Savitzky-Golay window (sec)")
    args = ap.parse_args()

    # supports both flat files and class sub-folders (raw_keypoints/<class>/*.npz)
    paths = sorted(glob(os.path.join(RAW_DIR, "**", "*.npz"), recursive=True))
    print(f"Found {len(paths)} raw keypoint files")
    for path in paths:
        rel = os.path.relpath(path, RAW_DIR)
        sub = os.path.dirname(rel)                       # class folder ("" if flat)
        name = os.path.splitext(os.path.basename(path))[0]
        tag = f"{sub}_{name}" if sub else name           # unique name for results/
        out_dir = os.path.join(OUT_DIR, sub)
        os.makedirs(out_dir, exist_ok=True)
        raw, vis, n_missing = load_raw(path)

        provided = provided_ground_correction(raw)                 # assignment method
        sm = ground_global(smooth(raw, args.fps, args.window))     # bonus
        sm_w2 = ground_global(smooth(raw, args.fps, args.window * 2))  # sensitivity check

        rows = {
            "raw": metrics(raw, args.fps, vis, n_missing),
            "provided (per-frame ground)": metrics(provided, args.fps),
            f"smooth (win {args.window}s)": metrics(sm, args.fps),
            f"smooth (win {args.window*2}s)": metrics(sm_w2, args.fps),
        }
        mdf = pd.DataFrame(rows).T
        mdf.to_csv(os.path.join(RES_DIR, f"{tag}_metrics.csv"))
        print(f"\n=== {tag} (T={len(raw)}) ===")
        print(mdf.round(4).to_string())

        versions = [
            ("raw", fill_gaps(raw), dict(fmt="-", lw=0.8, a=0.6)),
            ("provided", provided, dict(fmt="-", lw=0.8, a=0.8)),
            ("smooth", sm, dict(fmt="-", lw=1.4, a=1.0)),
        ]
        plot_trajectories(tag, versions, args.fps)
        plot_jitter(tag, versions, args.fps)
        plot_bones(tag, versions, args.fps)

        np.save(os.path.join(out_dir, f"{name}_smooth.npy"), sm)
        export_csv(sm, os.path.join(out_dir, f"{name}_smooth.csv"))

    print("\nDone. See results/ for figures and metrics.")


if __name__ == "__main__":
    main()
