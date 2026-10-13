"""Build cached voxel grids for every clip in data/<class>/*.mp4

Filenames should look like  class_vXX_sYY.mp4  (vXX = original video number).

Usage:
  python build_dataset.py --data data --out cache_events --C 0.2
  python build_dataset.py --data data --out cache_frames --mode framediff   # baseline
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from sim_events import load_clip, video_to_events, events_to_voxel


def frames_to_diff_stack(frames, bins):
    """Baseline: signed frame differences summed into `bins` time slices."""
    d = np.diff(frames, axis=0)
    idx = np.array_split(np.arange(len(d)), bins)
    return np.stack([d[i].sum(0) if len(i) else np.zeros_like(d[0]) for i in idx]).astype(np.float32)


def video_id(stem, cls):
    parts = stem.split("_")
    return f"{cls}_{parts[1]}" if len(parts) >= 3 else stem  # fall back to clip name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="cache_events")
    ap.add_argument("--mode", choices=["events", "framediff"], default="events")
    ap.add_argument("--C", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.0, help="background events per pixel per second")
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--size", type=int, default=96)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    classes = sorted(p.name for p in Path(a.data).iterdir() if p.is_dir())
    rows = []
    for ci, cls in enumerate(classes):
        clips = sorted(p for p in (Path(a.data) / cls).iterdir()
                       if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"})
        print(f"{cls}: {len(clips)} clips")
        for p in clips:
            try:
                frames, fps = load_clip(p, a.size)
                if a.mode == "events":
                    ev = video_to_events(frames, fps, C=a.C, noise_rate=a.noise)
                    vox = events_to_voxel(ev, a.bins)
                else:
                    vox = frames_to_diff_stack(frames, a.bins)
            except Exception as e:
                print("  skipped", p.name, "-", e)
                continue
            np.save(out / f"{p.stem}.npy", vox)
            rows.append([f"{p.stem}.npy", ci, cls, video_id(p.stem, cls)])

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "label", "class", "video"])
        w.writerows(rows)
    print(f"Saved {len(rows)} samples to {out}/  classes={classes}")


if __name__ == "__main__":
    main()
