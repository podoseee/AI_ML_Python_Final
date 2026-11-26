
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "raw_keypoints"
OUTPUT_DIR = ROOT / "data" / "dataset"

SEQ_LENGTH = 30
STRIDE = 15
VAL_RATIO = 0.2
SEED = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_raw_keypoints(path):
    if path.suffix == ".npy":
        # Processed keypoints: (T, 33, 3)
        motion = np.load(path)

        if motion.ndim == 3 and motion.shape[1:] == (33, 3):
            return motion

        raise ValueError(f"Invalid processed data shape: {motion.shape}")

    elif path.suffix == ".npz":
        # Raw keypoints: (T * 33, 6)
        data = np.load(path, allow_pickle=True)["data"]

        df = pd.DataFrame(
            data,
            columns=["frame", "landmark", "x", "y", "z", "visibility"]
        )

        pts_seq = []

        for _, group in df.groupby("frame"):
            group = group.sort_values("landmark")
            pts = group[["x", "y", "z"]].values
            pts_seq.append(pts)

        return np.array(pts_seq)

def create_sequences(motion):
    """
    motion: (T, 33, 3)
    return: (N, SEQ_LENGTH, 99)
    """
    if motion.ndim != 3 or motion.shape[1:] != (33, 3):
        raise ValueError(f"Invalid motion shape: {motion.shape}")

    if not np.isfinite(motion).all():
        raise ValueError("Motion contains NaN or Inf")

    sequences = []

    for start in range(0, len(motion) - SEQ_LENGTH + 1, STRIDE):
        clip = motion[start:start + SEQ_LENGTH]
        sequences.append(clip.reshape(SEQ_LENGTH, 99))

    return sequences


def split_video_paths(paths, rng):
    """
    Split by video, not by individual sequence.
    """
    paths = list(paths)
    rng.shuffle(paths)

    if len(paths) < 2:
        raise ValueError(
            "Each class needs at least two videos "
            "for train/validation splitting."
        )

    n_val = max(1, round(len(paths) * VAL_RATIO))
    n_val = min(n_val, len(paths) - 1)

    return paths[n_val:], paths[:n_val]


def build_dataset():
    rng = np.random.default_rng(SEED)

    class_dirs = sorted(
        p for p in INPUT_DIR.iterdir() if p.is_dir()
    )

    if not class_dirs:
        raise RuntimeError(f"No class folders found: {INPUT_DIR}")

    labels = {
        class_dir.name: i
        for i, class_dir in enumerate(class_dirs)
    }

    X_train, y_train = [], []
    X_val, y_val = [], []

    for class_dir in class_dirs:
        label = labels[class_dir.name]
        # paths = sorted(class_dir.glob("*.npy"))
        paths = sorted(list(class_dir.glob("*.npz")) +
    list(class_dir.glob("*.npy")))

        train_paths, val_paths = split_video_paths(paths, rng)

        print(f"\nClass: {class_dir.name}")
        print(f"  Train videos: {len(train_paths)}")
        print(f"  Val videos:   {len(val_paths)}")

        for split, selected_paths in [
            ("train", train_paths),
            ("val", val_paths),
        ]:
            for path in selected_paths:
                motion = load_raw_keypoints(path)
                
                # TODO (Optional):
                # 1. Design custom input features from motion data
                #    (e.g., joint angles, distances, or velocities).
                #
                # 2. Generate labels directly from motion data
                #    instead of using folder names.
                
                sequences = create_sequences(motion)

                if not sequences:
                    print(f"  Skipped (too short): {path.name}")
                    continue

                if split == "train":
                    X_train.extend(sequences)
                    y_train.extend([label] * len(sequences))
                else:
                    X_val.extend(sequences)
                    y_val.extend([label] * len(sequences))

    if not X_train or not X_val:
        raise RuntimeError(
            "No valid training or validation sequences were created."
        )

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    np.save(OUTPUT_DIR / "X_train.npy", X_train)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "X_val.npy", X_val)
    np.save(OUTPUT_DIR / "y_val.npy", y_val)

    with open(OUTPUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print("\nDataset created successfully!")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_val:  ", X_val.shape)
    print("y_val:  ", y_val.shape)
    print("Labels: ", labels)


if __name__ == "__main__":
    build_dataset()
