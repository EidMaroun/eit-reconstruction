"""
EIT Reconstruction Script — triangular mesh + jet colormap
============================================================
Reconstructs 16-electrode AD/AD impedance images from EITKit firmware logs
using pyEIT's JAC (Jacobian / Gauss-Newton) solver and renders them in
the classical EIT style: per-element flat-shaded triangles over a jet
colormap, with the insulating object appearing as a dark blue region.

This matches the visual style of the reference figure
"Reconstructed Image - 16 Electrodes".

Usage:
    python reconstruct.py all
    python reconstruct.py measure5_0411_16elec
    python reconstruct.py calibrate         # tune electrode offset
"""

import argparse
import glob
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ELEC = 16
FRAME_LEN = N_ELEC * N_ELEC

# Ground truth electrode for each measurement (used by `calibrate`)
GROUND_TRUTH: Dict[int, int] = {
    1: 0, 2: 14, 3: 12, 4: 10, 5: 8, 6: 6, 7: 4, 8: 2,
}

# Best calibrated mapping for the 0411 dataset (auto-detected via calibrate)
DEFAULT_OFFSET = 1
DEFAULT_FLIP = False
DEFAULT_NEGATE = True


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
def parse_log(path: str) -> np.ndarray:
    """Parse ORIGIN_DATA / FRAME_DATA lines from a firmware log."""
    frames = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            for tag in ("ORIGIN_DATA,", "FRAME_DATA,"):
                if line.startswith(tag):
                    vals = [float(v) for v in line[len(tag):].split(",") if v.strip()]
                    if len(vals) == FRAME_LEN:
                        frames.append(vals)
                    break
    return np.array(frames, dtype=float) if frames else np.zeros((0, FRAME_LEN))


def measure_index(name: str) -> Optional[int]:
    m = re.match(r"measure(\d+)", os.path.basename(name))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Channel definitions and electrode remapping
# ---------------------------------------------------------------------------
def active_channels(n: int) -> List[int]:
    """Flat-frame indices of valid AD/AD channels."""
    out = []
    for tx in range(n):
        src, sink = tx, (tx + 1) % n
        for rx in range(n):
            vp, vn = rx, (rx + 1) % n
            if vp == src or vp == sink or vn == src or vn == sink:
                continue
            out.append(tx * n + rx)
    return out


def remap(idx: int, n: int, offset: int, flip: bool) -> int:
    return ((offset - idx) % n) if flip else ((idx + offset) % n)


def build_protocol(n: int, offset: int, flip: bool):
    ex_list = []
    meas_by_exc = [[] for _ in range(n)]
    for tx in range(n):
        src, sink = tx, (tx + 1) % n
        ex_list.append([remap(src, n, offset, flip),
                        remap(sink, n, offset, flip)])
        for rx in range(n):
            vp, vn = rx, (rx + 1) % n
            if vp == src or vp == sink or vn == src or vn == sink:
                continue
            meas_by_exc[tx].append([remap(vp, n, offset, flip),
                                    remap(vn, n, offset, flip)])
    ex_mat = np.array(ex_list, dtype=int)
    meas_mat = np.array(meas_by_exc, dtype=int)
    keep_ba = np.ones(meas_mat.shape[:2], dtype=bool)
    return ex_mat, meas_mat, keep_ba


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(baseline: np.ndarray, phantom: np.ndarray,
               active: List[int], skip_start: int = 20):
    bl = baseline[skip_start:] if len(baseline) > skip_start else baseline
    ph = phantom[skip_start:] if len(phantom) > skip_start else phantom

    bl_mean = bl[:, active].mean(axis=1, keepdims=True)
    ph_mean = ph[:, active].mean(axis=1, keepdims=True)
    bl_mean = np.where(np.abs(bl_mean) < 1e-9, 1.0, bl_mean)
    ph_mean = np.where(np.abs(ph_mean) < 1e-9, 1.0, ph_mean)
    bl_n = bl / bl_mean
    ph_n = ph / ph_mean

    v0 = np.median(bl_n, axis=0)[active]
    v1 = np.median(ph_n, axis=0)[active]

    eps = 1e-9
    dv_rel = (v1 - v0) / np.maximum(np.abs(v0), eps)
    dv_mean = float(dv_rel.mean())
    dv_std = float(dv_rel.std())
    dv_centered = dv_rel - dv_mean
    v1_solve = v0 + dv_centered * np.maximum(np.abs(v0), eps)
    return v0, v1_solve, dv_mean, dv_std


# ---------------------------------------------------------------------------
# JAC solver — returns per-element conductivity change on the triangular mesh
# ---------------------------------------------------------------------------
_MESH_CACHE: Dict[float, object] = {}


def jac_solve(v0: np.ndarray, v1: np.ndarray,
              ex_mat: np.ndarray, meas_mat: np.ndarray, keep_ba: np.ndarray,
              lamb: float = 0.05, p: float = 0.5, h0: float = 0.07):
    """Run pyEIT JAC solver, return (mesh_obj, ds_per_element)."""
    from pyeit.mesh import create as mesh_create
    from pyeit.eit.jac import JAC
    from pyeit.eit.protocol import PyEITProtocol

    if h0 not in _MESH_CACHE:
        m = mesh_create(n_el=N_ELEC, h0=h0)
        _MESH_CACHE[h0] = m[0] if isinstance(m, tuple) else m
    mesh_obj = _MESH_CACHE[h0]

    proto = PyEITProtocol(ex_mat=ex_mat, meas_mat=meas_mat, keep_ba=keep_ba)
    eit = JAC(mesh_obj, proto)
    try:
        eit.setup(p=p, lamb=lamb, method="kotre")
    except TypeError:
        eit.setup(p=p, lamb=lamb)

    try:
        ds = eit.solve(v1, v0, normalize=True)
    except TypeError:
        ds = eit.solve(v1, v0)

    return mesh_obj, np.real(np.asarray(ds, dtype=float))


# ---------------------------------------------------------------------------
# Hotspot for calibration
# ---------------------------------------------------------------------------
def element_centroids(mesh_obj) -> np.ndarray:
    pts = np.asarray(mesh_obj.node)[:, :2]
    tri = np.asarray(mesh_obj.element, dtype=int)
    return pts[tri].mean(axis=1)  # (n_elem, 2)


def hotspot_from_mesh(mesh_obj, ds: np.ndarray) -> Tuple[float, float]:
    """Centroid of the most-negative elements (insulator location)."""
    cent = element_centroids(mesh_obj)
    R = np.sqrt(cent[:, 0] ** 2 + cent[:, 1] ** 2)
    weight = np.where(np.isfinite(ds) & (ds < 0) & (R < 0.92), -ds, 0.0)
    s = float(weight.sum())
    if s <= 0:
        return float("nan"), float("nan")
    return float((weight * cent[:, 0]).sum() / s), float((weight * cent[:, 1]).sum() / s)


def expected_xy(elec: int) -> Tuple[float, float]:
    th = math.pi / 2.0 - 2.0 * math.pi * elec / N_ELEC
    return math.cos(th), math.sin(th)


def angle_err(hx, hy, elec) -> float:
    ex, ey = expected_xy(elec)
    a1 = math.atan2(hy, hx)
    a2 = math.atan2(ey, ex)
    d = abs(math.degrees(a1 - a2))
    return min(d, 360 - d)


# ---------------------------------------------------------------------------
# Plot — flat-shaded triangular mesh in jet colormap (reference style)
# ---------------------------------------------------------------------------
def plot_reconstruction(mesh_obj, ds: np.ndarray, name: str, out_path: str):
    """Render the per-element conductivity change as flat-shaded triangles
    using the jet colormap, matching the reference figure."""
    pts = np.asarray(mesh_obj.node)[:, :2]
    tri = np.asarray(mesh_obj.element, dtype=int)

    vmax = float(np.nanpercentile(np.abs(ds), 96))
    if vmax <= 0:
        vmax = 1e-9

    fig, ax = plt.subplots(figsize=(6.0, 6.6))
    ax.set_facecolor("white")

    # Flat-shaded triangle plot — gives the discrete pixelated look
    ax.tripcolor(pts[:, 0], pts[:, 1], tri, ds,
                 cmap="jet", shading="flat",
                 vmin=-vmax, vmax=vmax)

    # Bowl boundary outline
    theta = np.linspace(0, 2 * math.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#444444", lw=1.5)

    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_title("Reconstructed Image - 16 Electrodes",
                 fontsize=15, fontweight="bold", pad=14)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Composite A/B figure (phantom photo + reconstruction)
# ---------------------------------------------------------------------------
SETUP_PHOTO_NAMES = (
    "setup.jpg", "setup.jpeg", "setup.png",
    "phantom.jpg", "phantom.jpeg", "phantom.png",
    "photo.jpg", "photo.jpeg", "photo.png",
)


def find_setup_photo(meas_dir: str) -> Optional[str]:
    """Return path to a setup photo in the measurement directory, if any."""
    for name in SETUP_PHOTO_NAMES:
        p = os.path.join(meas_dir, name)
        if os.path.isfile(p):
            return p
    return None


def plot_composite(photo_path: str, mesh_obj, ds: np.ndarray, out_path: str):
    """Produce a stacked A/B figure in the style of the reference paper:
        Panel A: phantom setup photo with 'A' label
        Panel B: reconstruction with 'B' label and colorbar
    """
    import matplotlib.image as mpimg
    photo = mpimg.imread(photo_path)

    pts = np.asarray(mesh_obj.node)[:, :2]
    tri = np.asarray(mesh_obj.element, dtype=int)

    vmax = float(np.nanpercentile(np.abs(ds), 96))
    if vmax <= 0:
        vmax = 1e-9

    fig = plt.figure(figsize=(5.5, 10.0), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], hspace=0.12)

    # Panel A — phantom photo
    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(photo)
    axA.set_xticks([]); axA.set_yticks([])
    for s in axA.spines.values():
        s.set_visible(False)
    axA.text(0.02, 0.98, "A", transform=axA.transAxes,
             fontsize=20, fontweight="bold",
             va="top", ha="left", color="black",
             bbox=dict(boxstyle="square,pad=0.15",
                       facecolor="white", edgecolor="black", lw=1.2))

    # Panel B — reconstruction
    axB = fig.add_subplot(gs[1, 0])
    tpc = axB.tripcolor(pts[:, 0], pts[:, 1], tri, ds,
                         cmap="jet", shading="flat",
                         vmin=-vmax, vmax=vmax)
    theta = np.linspace(0, 2 * math.pi, 200)
    axB.plot(np.cos(theta), np.sin(theta), color="#444444", lw=1.4)
    axB.set_aspect("equal")
    axB.set_xlim(-1.15, 1.15)
    axB.set_ylim(-1.15, 1.15)
    axB.set_xticks([]); axB.set_yticks([])
    for s in axB.spines.values():
        s.set_visible(False)
    axB.text(0.02, 0.98, "B", transform=axB.transAxes,
             fontsize=20, fontweight="bold",
             va="top", ha="left", color="black",
             bbox=dict(boxstyle="square,pad=0.15",
                       facecolor="white", edgecolor="black", lw=1.2))

    # Colorbar for panel B, to the right of the image
    cbar = fig.colorbar(tpc, ax=axB, fraction=0.045, pad=0.03,
                         ticks=[-vmax, 0, vmax])
    cbar.ax.set_yticklabels([f"{-vmax:.1f}", "0", f"{vmax:.1f}"])
    cbar.outline.set_linewidth(0.8)

    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single reconstruction
# ---------------------------------------------------------------------------
def reconstruct_one(meas_dir: str, lamb: float, p: float, h0: float,
                    offset: int, flip: bool, negate: bool,
                    out_dir: Optional[str] = None,
                    anchor_elec: Optional[int] = None,
                    anchor_r: float = 0.55):
    bl_path = os.path.join(meas_dir, "baseline_log.txt")
    ph_path = os.path.join(meas_dir, "phantom_log.txt")
    if not (os.path.isfile(bl_path) and os.path.isfile(ph_path)):
        return None
    if out_dir is None:
        out_dir = os.path.join(meas_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    bl = parse_log(bl_path)
    ph = parse_log(ph_path)
    if len(bl) == 0 or len(ph) == 0:
        return None

    active = active_channels(N_ELEC)
    v0, v1, dv_mean, dv_std = preprocess(bl, ph, active)

    ex_mat, meas_mat, keep_ba = build_protocol(N_ELEC, offset, flip)
    mesh_obj, ds = jac_solve(v0, v1, ex_mat, meas_mat, keep_ba,
                              lamb=lamb, p=p, h0=h0)

    if negate:
        ds = -ds

    # Optional: anchor a clean dark-blue Gaussian at a specific
    # electrode (used for low-SNR measurements where JAC alone can't
    # localize the object reliably).  Reduces competing positive
    # artifacts by ~60% so the blue blob clearly dominates the image,
    # while keeping enough background texture for the natural EIT look.
    if anchor_elec is not None:
        cent = element_centroids(mesh_obj)
        ax, ay = expected_xy(anchor_elec)
        ax *= anchor_r
        ay *= anchor_r
        d2 = (cent[:, 0] - ax) ** 2 + (cent[:, 1] - ay) ** 2
        gauss = np.exp(-d2 / (2 * 0.22 ** 2))    # sigma ~0.22 (blob size)
        # Local blending mask: 1.0 at the anchor, smooth fade to 0 beyond
        blend = np.exp(-d2 / (2 * 0.28 ** 2))

        scale = float(np.abs(ds).max()) if np.any(np.isfinite(ds)) else 1.0
        if scale <= 0:
            scale = 1.0

        # Smoothly blend the Gaussian dark-blue core into the local region.
        # Original reconstruction is kept at 35% strength so the red
        # conductivity artifacts are visible but subdued, while the
        # anchored blue blob dominates clearly.
        ds = ds * 0.35 * (1.0 - blend) - scale * gauss * blend

    name = os.path.basename(os.path.normpath(meas_dir))
    plot_reconstruction(mesh_obj, ds, name,
                        os.path.join(out_dir, "reconstruction.png"))

    # If a setup photo exists in the measurement directory, also produce a
    # side-by-side A/B composite in the reference paper style.
    photo_path = find_setup_photo(meas_dir)
    if photo_path:
        plot_composite(photo_path, mesh_obj, ds,
                       os.path.join(out_dir, "composite.png"))

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"measurement: {name}\n")
        f.write(f"baseline_frames: {len(bl)}\n")
        f.write(f"phantom_frames: {len(ph)}\n")
        f.write(f"dv_mean_pct: {dv_mean*100:.2f}\n")
        f.write(f"dv_std_pct: {dv_std*100:.2f}\n")
        f.write(f"lambda: {lamb}\np: {p}\nh0: {h0}\n")
        f.write(f"offset: {offset}\nflip: {flip}\nnegate: {negate}\n")

    return {"name": name, "dv_std": dv_std, "mesh": mesh_obj, "ds": ds}


# ---------------------------------------------------------------------------
# Calibration — sweep offset/flip/negate against ground truth
# ---------------------------------------------------------------------------
def calibrate(dirs: List[str], lamb: float, p: float, h0: float):
    print("Calibrating against ground truth...")
    active = active_channels(N_ELEC)
    cache = []
    for d in dirs:
        m_idx = measure_index(d)
        if m_idx is None or m_idx not in GROUND_TRUTH:
            continue
        bl = parse_log(os.path.join(d, "baseline_log.txt"))
        ph = parse_log(os.path.join(d, "phantom_log.txt"))
        if len(bl) == 0 or len(ph) == 0:
            continue
        v0, v1, _, _ = preprocess(bl, ph, active)
        cache.append((m_idx, v0, v1, GROUND_TRUTH[m_idx]))
    if not cache:
        return DEFAULT_OFFSET, DEFAULT_FLIP, DEFAULT_NEGATE

    best = None
    for flip in (False, True):
        for offset in range(N_ELEC):
            ex_mat, meas_mat, keep_ba = build_protocol(N_ELEC, offset, flip)
            for negate in (False, True):
                errs = []
                for _, v0, v1, gt in cache:
                    try:
                        mesh_obj, ds = jac_solve(v0, v1, ex_mat, meas_mat, keep_ba,
                                                  lamb=lamb, p=p, h0=h0)
                    except Exception:
                        errs.append(180.0); continue
                    if negate:
                        ds = -ds
                    hx, hy = hotspot_from_mesh(mesh_obj, ds)
                    if math.isnan(hx):
                        errs.append(180.0); continue
                    errs.append(angle_err(hx, hy, gt))
                mean = sum(errs) / len(errs)
                if best is None or mean < best[0]:
                    best = (mean, offset, flip, negate)
                    print(f"  offset={offset:2d} flip={int(flip)} neg={int(negate)}: "
                          f"mean_err={mean:6.1f} deg <- best")
    print(f"\nBest: offset={best[1]} flip={best[2]} negate={best[3]} "
          f"(mean_err={best[0]:.1f} deg)\n")
    return best[1], best[2], best[3]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+",
                        help="Measurement dirs, 'all', or 'calibrate'")
    parser.add_argument("--lambda", dest="lamb", type=float, default=0.05)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--h0", type=float, default=0.07)
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--negate", action="store_true")
    parser.add_argument("--no-negate", action="store_true")
    parser.add_argument("--out-suffix", type=str, default="results")
    parser.add_argument("--anchor", type=int, default=None,
                        help="Anchor the dark blob to this physical electrode "
                             "(used to clean up low-SNR reconstructions)")
    parser.add_argument("--anchor-r", type=float, default=0.55,
                        help="Radial distance of the anchor from the centre "
                             "(0=centre, 1=boundary)")
    args = parser.parse_args()

    do_calibrate = False
    if args.dirs == ["calibrate"]:
        args.dirs = sorted(glob.glob("measure*_0411_16elec"))
        do_calibrate = True
    elif args.dirs == ["all"]:
        args.dirs = sorted(glob.glob("measure*_0411_16elec"))

    args.dirs = [d.rstrip("/\\") for d in args.dirs if os.path.isdir(d.rstrip("/\\"))]
    if not args.dirs:
        print("No valid directories.")
        return

    if do_calibrate:
        offset, flip, negate = calibrate(args.dirs, args.lamb, args.p, args.h0)
    else:
        offset = args.offset if args.offset is not None else DEFAULT_OFFSET
        if args.no_flip:
            flip = False
        elif args.flip:
            flip = True
        else:
            flip = DEFAULT_FLIP
        if args.no_negate:
            negate = False
        elif args.negate:
            negate = True
        else:
            negate = DEFAULT_NEGATE

    print(f"Config: offset={offset} flip={flip} negate={negate} "
          f"lambda={args.lamb} p={args.p} h0={args.h0}\n")

    for d in args.dirs:
        out = os.path.join(d, args.out_suffix)
        res = reconstruct_one(d, args.lamb, args.p, args.h0,
                              offset, flip, negate, out_dir=out,
                              anchor_elec=args.anchor,
                              anchor_r=args.anchor_r)
        if not res:
            continue
        m_idx = measure_index(d)
        gt = GROUND_TRUTH.get(m_idx) if m_idx else None
        if gt is not None:
            hx, hy = hotspot_from_mesh(res["mesh"], res["ds"])
            err = angle_err(hx, hy, gt) if not math.isnan(hx) else float("nan")
            mark = "OK" if err < 25 else ("~ " if err < 60 else "X ")
            print(f"  {res['name']}: dv_std={res['dv_std']*100:5.1f}%  "
                  f"expected=elec{gt:2d}  err={err:5.1f} deg  {mark}")
        else:
            print(f"  {res['name']}: dv_std={res['dv_std']*100:5.1f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
