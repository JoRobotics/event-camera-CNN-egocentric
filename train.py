"""Train a small CNN on cached voxel grids. Splits by ORIGINAL VIDEO to avoid leakage.

Usage:  python train.py --cache cache_events --epochs 25
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def split_by_video(rows, val=0.15, test=0.15, seed=0):
    """Split whole videos into train/val/test, per class so every class appears in each split."""
    rng = np.random.default_rng(seed)
    by_cls = defaultdict(set)
    for r in rows:
        by_cls[r["label"]].add(r["video"])
    assign = {}
    for vids in by_cls.values():
        vids = sorted(vids)
        rng.shuffle(vids)
        n = len(vids)
        n_test = max(1, round(n * test)) if n >= 3 else 0
        n_val = max(1, round(n * val)) if n >= 3 else 0
        for i, v in enumerate(vids):
            assign[v] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return assign


class SmallCNN(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(in_ch, 16), block(16, 32), block(32, 64))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def evaluate(model, loader, n_classes, device):
    model.eval()
    cm = np.zeros((n_classes, n_classes), dtype=int)
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            for t, p in zip(y.numpy(), pred):
                cm[t, p] += 1
    acc = np.trace(cm) / max(cm.sum(), 1)
    return acc, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_events")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cache = Path(a.cache)
    with open(cache / "labels.csv") as f:
        rows = list(csv.DictReader(f))
    classes = sorted({r["class"] for r in rows}, key=lambda c: int(next(r["label"] for r in rows if r["class"] == c)))
    assign = split_by_video(rows, seed=a.seed)

    data = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for r in rows:
        x = np.load(cache / r["file"])
        s = assign[r["video"]]
        data[s][0].append(x); data[s][1].append(int(r["label"]))
    for s in data:
        print(s, len(data[s][1]), "clips")
    if not data["train"][1]:
        raise SystemExit("No training data - check your clips and filenames.")

    # normalise with training-set statistics
    Xtr = np.stack(data["train"][0])
    scale = float(np.abs(Xtr).std() + 1e-6)

    def loader(split, shuffle):
        if not data[split][1]:
            return None
        X = torch.tensor(np.stack(data[split][0]) / scale, dtype=torch.float32)
        y = torch.tensor(data[split][1])
        return DataLoader(TensorDataset(X, y), batch_size=a.bs, shuffle=shuffle)

    tr, va, te = loader("train", True), loader("val", False), loader("test", False)
    model = SmallCNN(Xtr.shape[1], len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.CrossEntropyLoss()

    best, best_state = -1, None
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0.0
        for x, y in tr:
            x, y = x.to(device), y.to(device)
            # light augmentation: random horizontal flip
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[3])
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward(); opt.step()
            tot += loss.item() * len(y)
        msg = f"epoch {ep:2d}  train loss {tot / len(tr.dataset):.4f}"
        if va:
            vacc, _ = evaluate(model, va, len(classes), device)
            msg += f"  val acc {vacc:.3f}"
            if vacc >= best:
                best, best_state = vacc, {k: v.clone() for k, v in model.state_dict().items()}
        print(msg)

    if best_state:
        model.load_state_dict(best_state)
    if te:
        acc, cm = evaluate(model, te, len(classes), device)
        print(f"\nTEST accuracy: {acc:.3f}  ({cm.sum()} clips)")
        print("Confusion matrix (rows = true, cols = predicted):")
        print("classes:", classes)
        print(cm)
    else:
        print("Too few videos for a test split - add more original videos.")


if __name__ == "__main__":
    main()
