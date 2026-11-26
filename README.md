# AI & ML Final Project

Extract 3D pose landmarks from dance videos and compare raw motion with ground-height correction.

## Setup and run

Run these commands from the repository root using Conda and Python 3.10.

```sh
conda env create -f environment.yml
conda activate ai_ml_final
python scripts/p1_create_raw_keypoints.py
python scripts/p1_2_post_processing.py
```

The first script downloads the five selected videos and saves the 33 BlazePose landmarks per detected frame to `data/raw_keypoints/`. The second exports raw and ground-corrected NumPy arrays and a Unity-compatible CSV for each clip to `data/processed/`. After export, it opens a Viser preview at `http://localhost:4242`. Press Ctrl+C to stop playback.

To generate optional smoothing results, comparison plots, and metrics for the selected 30 FPS clips, run the following command.

```sh
python scripts/p1_3_analyze_and_smooth.py --fps 30
```

Find comparison plots and metrics in `results/` and smoothed motion in `data/processed/`. The repository includes the source code and environment definition. The videos and generated datasets are created locally when the scripts run.
