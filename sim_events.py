"""Video -> simulated event stream -> voxel grid. Also a visual check for one clip.

Usage:  python sim_events.py data/iron/iron_v01_s01.mp4 --C 0.2
"""
import argparse
import cv2
import numpy as np


def load_clip(path, size=96, max_seconds=3.0):
    """Return (frames[T,H,W] float in 0..1, fps)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < int(max_seconds * fps) + 1:
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(g.astype(np.float32) / 255.0)
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"Could not read enough frames from {path}")
    return np.stack(frames), fps


def video_to_events(frames, fps, C=0.2, sigma_C=0.03, noise_rate=0.0, seed=0):
    """Log-intensity threshold model. Returns dict of x, y, t (seconds), p (+1/-1)."""
    rng = np.random.default_rng(seed)
    T, H, W = frames.shape
    L = np.log(frames + 1e-3)
    ref = L[0].copy()
    thr = np.clip(C + sigma_C * rng.standard_normal((H, W)), 0.05, None)  # per-pixel mismatch
    xs, ys, ts, ps = [], [], [], []
    dt = 1.0 / fps
    for i in range(1, T):
        diff = L[i] - ref
        n = np.floor(np.abs(diff) / thr).astype(np.int32)
        yy, xx = np.nonzero(n)
        if len(yy):
            cnt = n[yy, xx]
            sign = np.sign(diff[yy, xx])
            ref[yy, xx] += sign * cnt * thr[yy, xx]
            yy, xx = np.repeat(yy, cnt), np.repeat(xx, cnt)
            pp = np.repeat(sign, cnt)
            tt = (i - 1) * dt + dt * rng.random(len(yy))  # spread within the frame interval
            xs.append(xx); ys.append(yy); ts.append(tt); ps.append(pp)
        if noise_rate > 0:  # random background-activity events
            k = rng.poisson(noise_rate * H * W * dt)
            if k:
                xs.append(rng.integers(0, W, k)); ys.append(rng.integers(0, H, k))
                ts.append((i - 1) * dt + dt * rng.random(k))
                ps.append(rng.choice([-1.0, 1.0], k))
    if not xs:
        e = np.zeros(0)
        return dict(x=e.astype(int), y=e.astype(int), t=e, p=e, H=H, W=W, duration=(T - 1) * dt)
    x, y = np.concatenate(xs), np.concatenate(ys)
    t, p = np.concatenate(ts), np.concatenate(ps)
    o = np.argsort(t)
    return dict(x=x[o], y=y[o], t=t[o], p=p[o], H=H, W=W, duration=(T - 1) * dt)


def events_to_voxel(ev, bins=5):
    """Signed event counts in `bins` time slices -> array (bins, H, W)."""
    v = np.zeros((bins, ev["H"], ev["W"]), dtype=np.float32)
    if len(ev["t"]) == 0:
        return v
    b = np.minimum((ev["t"] / ev["duration"] * bins).astype(int), bins - 1)
    np.add.at(v, (b, ev["y"], ev["x"]), ev["p"])
    return v


def visualise(ev, bins=4, out="events_preview.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = events_to_voxel(ev, bins)
    fig, axs = plt.subplots(1, bins, figsize=(3 * bins, 3))
    for k, ax in enumerate(np.atleast_1d(axs)):
        img = np.ones((ev["H"], ev["W"], 3))
        img[v[k] > 0] = [1, 0, 0]   # positive = red
        img[v[k] < 0] = [0, 0, 1]   # negative = blue
        ax.imshow(img); ax.set_title(f"slice {k + 1}"); ax.axis("off")
    plt.tight_layout(); plt.savefig(out, dpi=120); plt.close()
    print(f"Saved {out}  |  {len(ev['t'])} events over {ev['duration']:.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--C", type=float, default=0.2)
    ap.add_argument("--size", type=int, default=96)
    a = ap.parse_args()
    frames, fps = load_clip(a.clip, a.size)
    visualise(video_to_events(frames, fps, C=a.C))
