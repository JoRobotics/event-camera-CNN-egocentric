"""Make result figures from the text copied out of your Command Prompt runs.

1. Paste the output of your train.py runs into one text file, e.g. results_log.txt.
   Include the line with the command ("python train.py --cache ... --seed N") for each run,
   because the script uses it to know which run each block of numbers belongs to.
   If you ran the same command twice, the later run is used.
2. Run:
       python plot_results.py results_log.txt
   Three figures are saved in the folder  figures/  and a summary table is printed.

Options:
       --events cache_events   name of the events cache (default shown)
       --frames cache_frames   name of the frame-difference cache (default shown)
       --out figures           output folder
"""
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Colours: first two slots of a validated categorical palette, on a light surface.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

RUN = re.compile(r"train\.py\b.*?--cache\s+(\S+)")
SEED = re.compile(r"--seed\s+(\d+)")
EPOCH = re.compile(r"epoch\s+(\d+)\s+train loss\s+([\d.]+)\s+val acc\s+([\d.]+)")
TEST = re.compile(r"TEST accuracy:\s*([\d.]+)\s+\((\d+) clips\)")
ROW1 = re.compile(r"\[\[\s*(\d+)\s+(\d+)\s*\]")
ROW2 = re.compile(r"^\s*\[\s*(\d+)\s+(\d+)\s*\]\]")
CLASSES = re.compile(r"classes:\s*\[(.*)\]")


def parse(path):
    """Return {(cache, seed): run} for every train.py run found in the log."""
    runs, cur = {}, None
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = RUN.search(line)
        if m:
            s = SEED.search(line)
            cur = dict(cache=m.group(1), seed=int(s.group(1)) if s else 0,
                       loss=[], val=[], acc=None, n=None, cm=[], classes=None)
            runs[(cur["cache"], cur["seed"])] = cur  # a re-run replaces the earlier one
            continue
        if cur is None:
            continue
        m = EPOCH.search(line)
        if m:
            cur["loss"].append(float(m.group(2)))
            cur["val"].append(float(m.group(3)))
            continue
        m = TEST.search(line)
        if m:
            cur["acc"], cur["n"] = float(m.group(1)), int(m.group(2))
            continue
        m = CLASSES.search(line)
        if m:
            cur["classes"] = re.findall(r"'([^']+)'", m.group(1))
            continue
        m = ROW1.search(line) or ROW2.search(line)
        if m:
            cur["cm"].append([int(m.group(1)), int(m.group(2))])
    return {k: r for k, r in runs.items() if r["acc"] is not None and len(r["cm"]) == 2}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def new_fig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE)
    return fig


def fig_accuracy(runs, series, seeds, out):
    """Grouped bars: test accuracy per seed, plus the mean."""
    fig = new_fig(8.2, 5.0)
    ax = fig.add_axes([0.08, 0.14, 0.9, 0.62])
    style(ax)
    groups = list(seeds) + ["mean"]
    x = np.arange(len(groups))
    width = 0.36 if len(series) == 2 else 0.5
    means = {}
    for k, (label, cache, colour) in enumerate(series):
        vals = [runs[(cache, s)]["acc"] * 100 for s in seeds]
        means[label] = float(np.mean(vals))
        vals.append(means[label])
        offs = x + (k - (len(series) - 1) / 2) * width
        bars = ax.bar(offs, vals, width * 0.92, color=colour, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center",
                    va="bottom", fontsize=9, color=INK)
    ax.axhline(50, color=INK2, linewidth=1, linestyle=(0, (4, 4)))
    ax.set_xlim(-0.6, len(groups) - 0.4 + 0.6)
    ax.text(len(groups) - 0.4 + 0.55, 51, "chance", ha="right", va="bottom", fontsize=9, color=INK2)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{t}%" for t in (0, 25, 50, 75, 100)])
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}\n{runs[(series[0][1], s)]['n']} test clips" for s in seeds] + ["Mean"])
    if len(series) == 2:
        names = [s[0] for s in series]
        title = (f"Mean test accuracy: {names[0].lower()} {means[names[0]]:.1f}%, "
                 f"{names[1].lower()} {means[names[1]]:.1f}%")
    else:
        title = f"Mean test accuracy: {means[series[0][0]]:.1f}%"
    fig.text(0.08, 0.93, title, fontsize=14, color=INK, weight="bold", ha="left")
    fig.text(0.08, 0.875, "Test accuracy on held-out recordings, one random split per seed",
             fontsize=10, color=INK2, ha="left")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=len(series),
              fontsize=10, labelcolor=INK)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_confusion(runs, series, seeds, out):
    """One confusion matrix per input type and seed, shaded by the share of each true class."""
    rows, cols = len(series), len(seeds)
    fig, axs = plt.subplots(rows, cols, figsize=(3.4 * cols + 0.8, 3.3 * rows + 1.0),
                            facecolor=SURFACE, squeeze=False)
    for r, (label, cache, colour) in enumerate(series):
        cmap = LinearSegmentedColormap.from_list("c", [SURFACE, colour])
        for c, s in enumerate(seeds):
            run = runs[(cache, s)]
            cm = np.array(run["cm"])
            frac = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax = axs[r][c]
            ax.set_facecolor(SURFACE)
            ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
            names = run["classes"] or ["class 0", "class 1"]
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15,
                            color="white" if frac[i, j] > 0.55 else INK)
            ax.set_xticks([0, 1]); ax.set_xticklabels(names, fontsize=9, color=INK2)
            ax.set_yticks([0, 1]); ax.set_yticklabels(names, fontsize=9, color=INK2)
            ax.tick_params(length=0)
            for side in ax.spines.values():
                side.set_visible(False)
            ax.set_title(f"{label}, seed {s}: {run['acc'] * 100:.1f}%", fontsize=10.5,
                         color=INK, loc="left")
            if c == 0:
                ax.set_ylabel("True class", fontsize=9, color=INK2)
            if r == rows - 1:
                ax.set_xlabel("Predicted class", fontsize=9, color=INK2)
    fig.suptitle("Both inputs mostly make the same kind of error on each split",
                 x=0.01, ha="left", fontsize=14, color=INK, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_curves(runs, series, seeds, out):
    """Validation accuracy (top) and training loss (bottom) per epoch, one column per seed."""
    cols = len(seeds)
    fig, axs = plt.subplots(2, cols, figsize=(3.9 * cols + 0.6, 6.6), facecolor=SURFACE,
                            squeeze=False, sharex="col")
    for c, s in enumerate(seeds):
        for label, cache, colour in series:
            run = runs[(cache, s)]
            ep = np.arange(1, len(run["val"]) + 1)
            axs[0][c].plot(ep, np.array(run["val"]) * 100, color=colour, linewidth=1.8, label=label)
            axs[1][c].plot(ep, run["loss"], color=colour, linewidth=1.8)
        for r in range(2):
            style(axs[r][c])
        axs[0][c].axhline(50, color=INK2, linewidth=1, linestyle=(0, (4, 4)))
        axs[0][c].set_ylim(30, 102)
        axs[0][c].set_yticks([40, 60, 80, 100])
        axs[0][c].set_yticklabels(["40%", "60%", "80%", "100%"])
        axs[0][c].set_title(f"Seed {s}", fontsize=10.5, color=INK, loc="left")
        axs[1][c].set_ylim(0, 0.75)
        axs[1][c].set_xlabel("Epoch", fontsize=9, color=INK2)
        if c == 0:
            axs[0][c].set_ylabel("Validation accuracy", fontsize=9, color=INK2)
            axs[1][c].set_ylabel("Training loss", fontsize=9, color=INK2)
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(series), loc="upper right",
               bbox_to_anchor=(0.99, 0.985), fontsize=10, labelcolor=INK)
    fig.suptitle("Training loss falls steadily; validation accuracy is erratic",
                 x=0.01, ha="left", fontsize=14, color=INK, weight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Make result figures from a pasted training log.")
    ap.add_argument("log")
    ap.add_argument("--events", default="cache_events")
    ap.add_argument("--frames", default="cache_frames")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()

    runs = parse(a.log)
    if not runs:
        raise SystemExit("No complete train.py runs found in the log. Check that it includes "
                         "the 'python train.py --cache ...' lines and the TEST accuracy output.")
    series = [(label, cache, colour) for label, cache, colour in
              (("Events", a.events, BLUE), ("Frame differences", a.frames, ORANGE))
              if any(k[0] == cache for k in runs)]
    if not series:
        raise SystemExit(f"No runs found for caches '{a.events}' or '{a.frames}'. "
                         f"Found: {sorted({k[0] for k in runs})}")
    seeds = sorted(set.intersection(*[{k[1] for k in runs if k[0] == c} for _, c, _ in series]))
    if not seeds:
        raise SystemExit("The input types have no seed in common, so they cannot be compared.")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_accuracy(runs, series, seeds, out / "fig1_accuracy_by_seed.png")
    fig_confusion(runs, series, seeds, out / "fig2_confusion_matrices.png")
    fig_curves(runs, series, seeds, out / "fig3_training_curves.png")

    print(f"Seeds used: {seeds}")
    print(f"{'Input':<20}" + "".join(f"seed {s:<6}" for s in seeds) + "mean")
    for label, cache, _ in series:
        accs = [runs[(cache, s)]["acc"] * 100 for s in seeds]
        print(f"{label:<20}" + "".join(f"{v:<11.1f}" for v in accs) + f"{np.mean(accs):.1f}")
    print(f"Figures saved in {out.resolve()}")


if __name__ == "__main__":
    main()
