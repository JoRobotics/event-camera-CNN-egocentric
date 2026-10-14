# Event-Camera Simulation and CNN Action Classification from Egocentric Household Video

A pilot study: simulate event-camera data from ordinary egocentric video, then train a small CNN to tell two household actions apart (**food** preparation vs **ironing**).

## What it does
1. **Simulates events** from video with a log-intensity contrast-threshold model (C = 0.2, per-pixel threshold mismatch), producing +/- polarity events with timestamps.
2. **Bins events** into a 5-slice signed voxel grid (96x96) per 3-second clip.
3. **Classifies** with a small 3-block CNN (~24k parameters).
4. **Compares** against a baseline that uses plain frame differences instead of events.

## Results
Splits are made by original recording (no clips from one recording appear in more than one split), repeated over 3 random seeds.

| Input | Mean test accuracy |
|---|---|
| Simulated events | 80.5% |
| Frame-difference baseline | 81.1% |

- Accuracy varies a lot between splits (events: 22.5 points between best and worst seed).
- Errors concentrate on specific held-out recordings, which points to scene, lighting and head-motion differences between recordings rather than the input type.
- The simulated events performed about the same as frame differences, which is expected because the simulator is built from the same frames.

Figures are in `figures/`:
- `fig1_accuracy_by_seed.png` - test accuracy per seed and mean
- `fig2_confusion_matrices.png` - confusion matrices per input and seed
- `fig3_training_curves.png` - training loss and validation accuracy

## Limitations
- Only 4 original recordings per class, so test sets are small and results are noisy.
- Events are simulated from 30 fps video, with no interpolation, so fine timing is not real event-camera timing.
- Two classes only; the recordings differ in more than the action (scene, lighting).

## Files
| File | Purpose |
|---|---|
| `sim_events.py` | video -> simulated events -> voxel grid, plus a preview image |
| `build_dataset.py` | builds cached voxel grids for all clips (`--mode events` or `framediff`) |
| `train.py` | trains and tests the CNN, splitting by original recording |
| `plot_results.py` | makes the figures from a saved training log |
| `run_all.bat` | runs every experiment (Windows) and then the plots |

## How to run (Windows Command Prompt)
Clips go in `data\food\` and `data\iron\`, named like `food_v01_s01.mp4` (`v01` = original recording, `s01` = clip number).

```
pip install numpy opencv-python torch matplotlib
python build_dataset.py --data data --out cache_events --C 0.2
python build_dataset.py --data data --out cache_frames --mode framediff
run_all.bat
```
## How to run (Linux Command Prompt)

```
# 1. Install the required Python packages
pip install numpy opencv-python torch matplotlib

# 2. Extract event data from your original recordings and clips
python build_dataset.py --data data --out cache_events --C 0.2

# 3. Compute frame differences for the video clips
python build_dataset.py --data data --out cache_frames --mode framediff

# 4. Make the execution script executable and run it
chmod +x run_all.sh
./run_all.sh
```


The video data is not included in this repository.
