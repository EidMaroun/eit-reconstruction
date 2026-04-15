#!/usr/bin/env python3
"""
Model-based GREIT reconstruction for 8-electrode saline bowl experiments.

Purpose
-------
Generate an EIDORS/pyEIT-style differential image from:
  1) baseline log (saline only)
  2) phantom log (saline + object)

Key features
------------
- Parses FRAME_DATA logs from Teensy output.
- Settled-frame selection (manual or auto warm-up detect).
- Optional per-frame global normalization (recommended).
- Optional common-mode rejection (recommended when baseline/phantom means drift).
- pyEIT GREIT inverse solve (model-based).
- Exports a high-contrast "object map" PNG + CSV + summary.

Notes
-----
- This script expects pyEIT + numpy + matplotlib in your execution environment.
- Designed for AD/AD data (adjacent drive, adjacent measurement), matching your current firmware run.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)")


@dataclass
class ParsedLog:
    frames: List[List[float]]
    dropped_frames: int
    frame_len: int


@dataclass
class ChannelDef:
    idx: int
    tx: int
    rx: int
    src: int
    sink: int
    vp: int
    vn: int
    valid: bool


@dataclass
class SettledSelection:
    selected_frames: List[List[float]]
    settled_start: int
    settled_end: int
    frame_means_all: List[float]


def parse_numeric_payload(text: str) -> List[float]:
    return [float(m.group(0)) for m in FLOAT_RE.finditer(text)]


def parse_log_file(path: str) -> ParsedLog:
    frame_candidates: List[List[float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("FRAME_DATA,"):
                vals = parse_numeric_payload(line.split(",", 1)[1])
                if vals:
                    frame_candidates.append(vals)

    if not frame_candidates:
        raise ValueError(f"No FRAME_DATA lines found in: {path}")

    lengths = Counter(len(v) for v in frame_candidates)
    frame_len, _ = lengths.most_common(1)[0]
    frames = [v for v in frame_candidates if len(v) == frame_len]
    dropped = len(frame_candidates) - len(frames)
    if not frames:
        raise ValueError(f"No complete modal-length frames found in: {path}")

    return ParsedLog(frames=frames, dropped_frames=dropped, frame_len=frame_len)


def infer_num_electrodes(frame_len: int) -> int:
    n = int(round(math.sqrt(frame_len)))
    if n * n != frame_len:
        raise ValueError(
            f"Frame length {frame_len} is not a perfect square; cannot infer number of electrodes."
        )
    return n


def pattern_pair(index: int, n: int, pattern: str, for_drive: bool) -> Tuple[int, int]:
    if pattern == "AD":
        return index, (index + 1) % n
    if pattern == "OP":
        return index, (index + n // 2) % n
    if pattern == "MONO":
        return (index, 0) if for_drive else (index, 0)
    raise ValueError(f"Unsupported pattern: {pattern}")


def is_valid_channel(src: int, sink: int, vp: int, vn: int, meas_pattern: str) -> bool:
    if meas_pattern == "MONO":
        invalid = (vp == src) or (vp == vn) or (src == sink)
    else:
        invalid = (vp == src) or (vp == sink) or (vn == src) or (vn == sink)
    return not invalid


def build_channel_defs(
    n: int, drive_pattern: str, meas_pattern: str
) -> Tuple[List[ChannelDef], List[List[bool]]]:
    defs: List[ChannelDef] = []
    mask = [[False for _ in range(n)] for _ in range(n)]
    idx = 0
    for tx in range(n):
        src, sink = pattern_pair(tx, n, drive_pattern, for_drive=True)
        for rx in range(n):
            vp, vn = pattern_pair(rx, n, meas_pattern, for_drive=False)
            valid = is_valid_channel(src, sink, vp, vn, meas_pattern)
            defs.append(
                ChannelDef(
                    idx=idx,
                    tx=tx,
                    rx=rx,
                    src=src,
                    sink=sink,
                    vp=vp,
                    vn=vn,
                    valid=valid,
                )
            )
            mask[tx][rx] = valid
            idx += 1
    return defs, mask


def frame_active_mean(frame: List[float], active_idx: List[int]) -> float:
    return statistics.mean(frame[i] for i in active_idx)


def auto_detect_settled_start(
    frame_means: List[float], window: int = 25, tol_rel: float = 0.02
) -> int:
    n = len(frame_means)
    if n < max(10, window + 5):
        return 0
    tail = frame_means[max(0, int(0.7 * n)) :]
    baseline = statistics.median(tail)
    if abs(baseline) < 1e-12:
        return 0
    for i in range(0, n - window):
        ok = True
        for j in range(i, i + window):
            rel = abs(frame_means[j] - baseline) / abs(baseline)
            if rel > tol_rel:
                ok = False
                break
        if ok:
            return i
    return 0


def select_settled_frames(
    frames: List[List[float]],
    active_idx: List[int],
    settled_start: int,
    settled_end: int,
    auto_settle: bool,
    auto_window: int,
    auto_tol_rel: float,
) -> SettledSelection:
    means = [frame_active_mean(fr, active_idx) for fr in frames]
    start = max(0, min(settled_start, len(frames) - 1))
    if auto_settle:
        start = max(
            start,
            auto_detect_settled_start(means, window=max(5, auto_window), tol_rel=max(1e-6, auto_tol_rel)),
        )
    if settled_end < 0:
        end = len(frames) - 1
    else:
        end = max(start, min(settled_end, len(frames) - 1))
    selected = frames[start : end + 1]
    if not selected:
        raise ValueError("No settled frames selected.")
    return SettledSelection(selected_frames=selected, settled_start=start, settled_end=end, frame_means_all=means)


def aggregate_frames(frames: List[List[float]], stat: str) -> List[float]:
    frame_len = len(frames[0])
    out = [0.0] * frame_len
    for i in range(frame_len):
        col = [fr[i] for fr in frames]
        out[i] = statistics.median(col) if stat == "median" else statistics.mean(col)
    return out


def normalize_frames_by_active_mean(frames: List[List[float]], active_idx: List[int]) -> List[List[float]]:
    out: List[List[float]] = []
    for fr in frames:
        m = frame_active_mean(fr, active_idx)
        if abs(m) < 1e-12:
            out.append(fr[:])
        else:
            out.append([v / m for v in fr])
    return out


def extract_active_vector(avg_flat: List[float], channels: List[ChannelDef]) -> np.ndarray:
    vals = [avg_flat[ch.idx] for ch in channels if ch.valid]
    return np.asarray(vals, dtype=float)


def build_ex_mat(n: int, drive_pattern: str) -> np.ndarray:
    ex = []
    for tx in range(n):
        src, sink = pattern_pair(tx, n, drive_pattern, for_drive=True)
        ex.append([src, sink])
    return np.asarray(ex, dtype=int)


def build_protocol_meas_keepba(
    n: int, channels: List[ChannelDef]
) -> Tuple[np.ndarray, np.ndarray]:
    rows_by_exc: List[List[List[int]]] = [[] for _ in range(n)]
    for ch in channels:
        if ch.valid:
            rows_by_exc[ch.tx].append([ch.vp, ch.vn])

    per_exc_counts = [len(v) for v in rows_by_exc]
    if not per_exc_counts or min(per_exc_counts) == 0:
        raise ValueError("No valid measurements found for one or more excitations.")
    if len(set(per_exc_counts)) != 1:
        raise ValueError(
            f"Inconsistent number of measurements per excitation: {per_exc_counts}"
        )

    meas_mat = np.asarray(rows_by_exc, dtype=int)  # shape (n_exc, n_meas, 2)
    keep_ba = np.ones((n, per_exc_counts[0]), dtype=bool)  # same first 2 dims as meas_mat
    return meas_mat, keep_ba


def gaussian_blur_nan(img: np.ndarray, sigma_px: float = 1.2, passes: int = 2) -> np.ndarray:
    if sigma_px <= 0:
        return img.copy()
    radius = max(1, int(round(3.0 * sigma_px)))
    x = np.arange(-radius, radius + 1, dtype=float)
    ker = np.exp(-0.5 * (x / sigma_px) ** 2)
    ker = ker / np.sum(ker)

    out = img.copy()
    for _ in range(max(1, passes)):
        # horizontal
        tmp = np.full_like(out, np.nan, dtype=float)
        for i in range(out.shape[0]):
            row = out[i]
            for j in range(out.shape[1]):
                if np.isnan(row[j]):
                    continue
                s = 0.0
                w = 0.0
                for k, wk in enumerate(ker):
                    jj = j + (k - radius)
                    if 0 <= jj < out.shape[1] and not np.isnan(row[jj]):
                        s += wk * row[jj]
                        w += wk
                tmp[i, j] = s / w if w > 0 else row[j]
        # vertical
        nxt = np.full_like(out, np.nan, dtype=float)
        for j in range(tmp.shape[1]):
            col = tmp[:, j]
            for i in range(tmp.shape[0]):
                if np.isnan(col[i]):
                    continue
                s = 0.0
                w = 0.0
                for k, wk in enumerate(ker):
                    ii = i + (k - radius)
                    if 0 <= ii < tmp.shape[0] and not np.isnan(col[ii]):
                        s += wk * col[ii]
                        w += wk
                nxt[i, j] = s / w if w > 0 else col[i]
        out = nxt
    return out


def electrode_positions(n: int) -> List[Tuple[float, float]]:
    pos = []
    for i in range(n):
        th = (math.pi / 2.0) - (2.0 * math.pi * i / n)
        pos.append((math.cos(th), math.sin(th)))
    return pos


def to_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zz = np.asarray(z, dtype=float)
    if zz.ndim == 1:
        g = int(round(math.sqrt(zz.size)))
        if g * g == zz.size:
            zz = zz.reshape((g, g))
        else:
            raise ValueError("Cannot reshape GREIT output into square grid.")

    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    if xx.ndim == 1 and yy.ndim == 1:
        if xx.size == zz.shape[1] and yy.size == zz.shape[0]:
            X, Y = np.meshgrid(xx, yy)
        elif xx.size == zz.size and yy.size == zz.size:
            X = xx.reshape(zz.shape)
            Y = yy.reshape(zz.shape)
        else:
            # fallback synthetic coordinate grid
            gx = np.linspace(-1.0, 1.0, zz.shape[1])
            gy = np.linspace(-1.0, 1.0, zz.shape[0])
            X, Y = np.meshgrid(gx, gy)
    elif xx.shape == zz.shape and yy.shape == zz.shape:
        X, Y = xx, yy
    else:
        gx = np.linspace(-1.0, 1.0, zz.shape[1])
        gy = np.linspace(-1.0, 1.0, zz.shape[0])
        X, Y = np.meshgrid(gx, gy)
    return X, Y, zz


def run_greit(
    n_el: int,
    ex_mat: np.ndarray,
    meas_mat: np.ndarray,
    keep_ba: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    h0: float,
    lamb: float,
    p: float,
    n_grid: int,
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from pyeit.mesh import create as mesh_create
        from pyeit.eit.greit import GREIT
    except Exception as e:
        raise RuntimeError(
            "pyEIT import failed. Install with: pip install pyEIT"
        ) from e

    # Mesh creation compatibility across pyEIT versions.
    mesh_created = mesh_create(n_el=n_el, h0=h0)
    if isinstance(mesh_created, tuple):
        mesh_obj = mesh_created[0]
        el_pos = mesh_created[1]
    else:
        mesh_obj = mesh_created
        # fallback electrode positions for old constructor path
        el_pos = np.arange(n_el, dtype=int)

    def _setup_solve_and_grid(eit_obj) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        setup_errors: List[str] = []
        setup_attempts = [
            {"p": p, "lamb": lamb, "n": n_grid},
            {"p": p, "lamb": lamb},
            {"lamb": lamb, "p": p},
            {},
        ]
        setup_ok = False
        for kwargs in setup_attempts:
            try:
                eit_obj.setup(**kwargs)
                setup_ok = True
                break
            except TypeError as e:
                setup_errors.append(f"setup{kwargs}: {e}")
        if not setup_ok:
            raise RuntimeError("GREIT setup failed: " + " | ".join(setup_errors))

        solve_errors: List[str] = []
        ds = None
        for call_idx in (0, 1):
            try:
                if call_idx == 0:
                    ds = eit_obj.solve(v1, v0, normalize=normalize)
                else:
                    ds = eit_obj.solve(v1, v0)
                break
            except TypeError as e:
                solve_errors.append(str(e))
        if ds is None:
            raise RuntimeError("GREIT solve failed: " + " | ".join(solve_errors))

        # Decode / mask image output across pyEIT variants.
        x = y = dsm = None
        mask_errors: List[str] = []
        for kwargs in ({"mask_value": np.nan}, {}):
            try:
                x, y, dsm = eit_obj.mask_value(ds, **kwargs)
                break
            except Exception as e:
                mask_errors.append(str(e))

        if dsm is None:
            # Fallback to common attribute names when mask_value API differs.
            dsm = np.asarray(ds)
            if hasattr(eit_obj, "xg") and hasattr(eit_obj, "yg"):
                x = np.asarray(getattr(eit_obj, "xg"))
                y = np.asarray(getattr(eit_obj, "yg"))
            elif hasattr(eit_obj, "x") and hasattr(eit_obj, "y"):
                x = np.asarray(getattr(eit_obj, "x"))
                y = np.asarray(getattr(eit_obj, "y"))
            else:
                g = int(round(math.sqrt(dsm.size)))
                if g * g == dsm.size:
                    x = np.linspace(-1.0, 1.0, g)
                    y = np.linspace(-1.0, 1.0, g)
                else:
                    raise RuntimeError(
                        "GREIT mask/output decode failed and no fallback grid available: "
                        + " | ".join(mask_errors)
                    )

        X, Y, Z = to_grid(np.asarray(x), np.asarray(y), np.asarray(dsm))
        return X, Y, Z

    # Try modern protocol object first (explicit measurement ordering).
    protocol_obj = None
    protocol_error: Optional[Exception] = None
    protocol_constructor_errors: List[str] = []
    protocol_api_present = False
    try:
        from pyeit.eit import protocol as protocol_mod

        if hasattr(protocol_mod, "PyEITProtocol"):
            protocol_api_present = True
            Proto = getattr(protocol_mod, "PyEITProtocol")
            keep_ba_all = keep_ba
            proto_attempts = [
                lambda: Proto(ex_mat=ex_mat, meas_mat=meas_mat, keep_ba=keep_ba_all),
                lambda: Proto(ex_mat, meas_mat, keep_ba_all),
                lambda: Proto(ex_mat=ex_mat, meas_mat=meas_mat, keep_ba=np.asarray(keep_ba_all, dtype=int)),
                lambda: Proto(ex_mat, meas_mat, np.asarray(keep_ba_all, dtype=int)),
                lambda: Proto(ex_mat=ex_mat, meas_mat=meas_mat, keep_ba=keep_ba_all.reshape(-1)),
                lambda: Proto(ex_mat, meas_mat, keep_ba_all.reshape(-1)),
                lambda: Proto(ex_mat=ex_mat, meas_mat=meas_mat, keep_ba=np.asarray(keep_ba_all.reshape(-1), dtype=int)),
                lambda: Proto(ex_mat, meas_mat, np.asarray(keep_ba_all.reshape(-1), dtype=int)),
            ]
            for make_proto in proto_attempts:
                try:
                    protocol_obj = make_proto()
                    break
                except Exception as e:
                    protocol_constructor_errors.append(str(e))
    except Exception as e:
        protocol_error = e

    if protocol_api_present and protocol_obj is None and protocol_error is None:
        protocol_error = RuntimeError(
            "PyEITProtocol construction failed. Tried signatures produced: "
            + " | ".join(protocol_constructor_errors)
        )

    if protocol_obj is not None:
        greit_ctor_errors: List[str] = []
        greit_attempts = [
            lambda: GREIT(mesh_obj, protocol_obj),
            lambda: GREIT(mesh=mesh_obj, protocol=protocol_obj),
            lambda: GREIT(mesh_obj, protocol=protocol_obj),
        ]
        for make_eit in greit_attempts:
            try:
                eit = make_eit()
                return _setup_solve_and_grid(eit)
            except Exception as e:
                greit_ctor_errors.append(str(e))
        # Fall through to legacy constructor path if these fail.
        protocol_error = RuntimeError(
            "Protocol GREIT path failed. "
            + ("Proto ctor errors: " + " | ".join(protocol_constructor_errors) + " ; " if protocol_constructor_errors else "")
            + "GREIT ctor/solve errors: "
            + " | ".join(greit_ctor_errors)
        )

    # If protocol API exists but construction failed, stop with explicit message.
    if protocol_api_present and protocol_obj is None and protocol_error is not None:
        raise RuntimeError(
            "GREIT protocol path failed and this pyEIT build expects GREIT(mesh, protocol). "
            f"Protocol error: {protocol_error}"
        ) from protocol_error

    # Fallback path: older GREIT API.
    # This path assumes AD/AD ordering from firmware matches parser='std', step=1 ordering.
    fallback_errors: List[str] = []
    legacy_attempts = [
        lambda: GREIT(mesh_obj, el_pos, ex_mat, step=1, parser="std"),
        lambda: GREIT(mesh_obj, el_pos, ex_mat),
        lambda: GREIT(mesh_obj, el_pos, step=1, parser="std"),
        lambda: GREIT(mesh_obj, el_pos),
        lambda: GREIT(mesh_obj),
    ]
    for make_eit in legacy_attempts:
        try:
            eit = make_eit()
            return _setup_solve_and_grid(eit)
        except Exception as e:
            fallback_errors.append(str(e))

    if protocol_error is not None:
        raise RuntimeError(
            "GREIT construction failed in both protocol and legacy paths. "
            f"Protocol error: {protocol_error}; legacy errors: {' | '.join(fallback_errors)}"
        ) from protocol_error
    raise RuntimeError("GREIT construction failed: " + " | ".join(fallback_errors))


def write_matrix_csv(path: str, m: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for i in range(m.shape[0]):
            row = []
            for j in range(m.shape[1]):
                v = m[i, j]
                row.append("" if np.isnan(v) else f"{v:.10f}")
            w.writerow(row)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="pyEIT GREIT reconstruction for 8-electrode bowl experiments")
    parser.add_argument("--baseline-log", required=True, help="Path to baseline FRAME_DATA log")
    parser.add_argument("--phantom-log", required=True, help="Path to phantom FRAME_DATA log")
    parser.add_argument("--out-dir", default="output_greit_8eit", help="Output directory")

    parser.add_argument("--num-electrodes", type=int, default=8)
    parser.add_argument("--drive-pattern", choices=("AD", "OP", "MONO"), default="AD")
    parser.add_argument("--meas-pattern", choices=("AD", "OP", "MONO"), default="AD")
    parser.add_argument("--frame-stat", choices=("mean", "median"), default="median")

    parser.add_argument("--settled-start", type=int, default=20)
    parser.add_argument("--settled-end", type=int, default=-1)
    parser.add_argument("--auto-settle", action="store_true")
    parser.add_argument("--auto-window", type=int, default=25)
    parser.add_argument("--auto-tol-rel", type=float, default=0.02)

    parser.add_argument("--frame-normalize", action="store_true", default=True)
    parser.add_argument("--no-frame-normalize", action="store_true")
    parser.add_argument("--remove-common-mode", action="store_true", default=True)
    parser.add_argument("--no-remove-common-mode", action="store_true")

    parser.add_argument("--mesh-h0", type=float, default=0.10, help="Mesh element size hint for pyEIT")
    parser.add_argument("--greit-lambda", type=float, default=0.01)
    parser.add_argument("--greit-p", type=float, default=0.35)
    parser.add_argument("--greit-grid", type=int, default=192)
    parser.add_argument("--solve-normalize", action="store_true", default=False, help="Use pyEIT solve(normalize=True)")

    parser.add_argument("--blur-sigma", type=float, default=1.4, help="Display smoothing sigma (pixels)")
    parser.add_argument("--clip-percentile", type=float, default=98.5, help="Symmetric contrast clip percentile")
    args = parser.parse_args(argv)

    if args.num_electrodes < 4 or args.num_electrodes > 32:
        raise ValueError("--num-electrodes must be in [4, 32]")
    if args.drive_pattern != "AD" or args.meas_pattern != "AD":
        raise ValueError("This GREIT script currently targets AD/AD acquisition only.")

    use_frame_norm = args.frame_normalize and (not args.no_frame_normalize)
    use_common_mode = args.remove_common_mode and (not args.no_remove_common_mode)

    parsed_b = parse_log_file(args.baseline_log)
    parsed_p = parse_log_file(args.phantom_log)

    n_b = infer_num_electrodes(parsed_b.frame_len)
    n_p = infer_num_electrodes(parsed_p.frame_len)
    if n_b != n_p:
        raise ValueError(f"Electrode count mismatch in logs: baseline={n_b}, phantom={n_p}")
    if n_b != args.num_electrodes:
        raise ValueError(f"--num-electrodes={args.num_electrodes} but inferred from logs={n_b}")
    n = n_b

    channels, mask = build_channel_defs(n, args.drive_pattern, args.meas_pattern)
    mask_flat = [mask[i][j] for i in range(n) for j in range(n)]
    active_idx = [i for i, v in enumerate(mask_flat) if v]

    sel_b = select_settled_frames(
        frames=parsed_b.frames,
        active_idx=active_idx,
        settled_start=args.settled_start,
        settled_end=args.settled_end,
        auto_settle=args.auto_settle,
        auto_window=args.auto_window,
        auto_tol_rel=args.auto_tol_rel,
    )
    sel_p = select_settled_frames(
        frames=parsed_p.frames,
        active_idx=active_idx,
        settled_start=args.settled_start,
        settled_end=args.settled_end,
        auto_settle=args.auto_settle,
        auto_window=args.auto_window,
        auto_tol_rel=args.auto_tol_rel,
    )

    frames_b = sel_b.selected_frames
    frames_p = sel_p.selected_frames

    if use_frame_norm:
        frames_b = normalize_frames_by_active_mean(frames_b, active_idx)
        frames_p = normalize_frames_by_active_mean(frames_p, active_idx)

    avg_b = aggregate_frames(frames_b, stat=args.frame_stat)
    avg_p = aggregate_frames(frames_p, stat=args.frame_stat)

    v0 = extract_active_vector(avg_b, channels)
    v1 = extract_active_vector(avg_p, channels)

    # Relative differential channel vector.
    eps = 1e-12
    dv_rel = (v1 - v0) / np.maximum(np.abs(v0), eps)
    dv_rel_mean = float(np.mean(dv_rel))
    dv_rel_std = float(np.std(dv_rel))

    if use_common_mode:
        dv_rel_centered = dv_rel - dv_rel_mean
        v1_for_solve = v0 + dv_rel_centered * np.maximum(np.abs(v0), eps)
    else:
        dv_rel_centered = dv_rel.copy()
        v1_for_solve = v1

    ex_mat = build_ex_mat(n, args.drive_pattern)
    meas_mat, keep_ba = build_protocol_meas_keepba(n, channels)

    X, Y, Z = run_greit(
        n_el=n,
        ex_mat=ex_mat,
        meas_mat=meas_mat,
        keep_ba=keep_ba,
        v0=v0,
        v1=v1_for_solve,
        h0=args.mesh_h0,
        lamb=args.greit_lambda,
        p=args.greit_p,
        n_grid=args.greit_grid,
        normalize=args.solve_normalize,
    )

    Zs = gaussian_blur_nan(Z, sigma_px=max(0.0, args.blur_sigma), passes=2)
    valid = np.isfinite(Zs)
    if np.any(valid):
        clip_q = min(max(args.clip_percentile, 80.0), 99.9)
        vmax = float(np.nanpercentile(np.abs(Zs[valid]), clip_q))
        vmax = max(vmax, 1e-9)
    else:
        vmax = 1.0

    # hotspot from |Z|
    A = np.where(np.isfinite(Zs), np.abs(Zs), 0.0)
    s = float(np.sum(A))
    if s > 0:
        hx = float(np.sum(A * X) / s)
        hy = float(np.sum(A * Y) / s)
        hr = float(math.sqrt(hx * hx + hy * hy))
    else:
        hx = float("nan")
        hy = float("nan")
        hr = float("nan")

    os.makedirs(args.out_dir, exist_ok=True)
    write_matrix_csv(os.path.join(args.out_dir, "greit_reconstruction_map.csv"), Zs)

    # Save differential channels in tx/rx matrix form (active entries only).
    dv_mat = np.full((n, n), np.nan, dtype=float)
    k = 0
    for ch in channels:
        if ch.valid:
            dv_mat[ch.tx, ch.rx] = float(dv_rel[k])
            k += 1
    write_matrix_csv(os.path.join(args.out_dir, "channel_delta_rel_matrix.csv"), dv_mat)

    # EIDORS-style main plot.
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    levels = np.linspace(-vmax, vmax, 33)
    cf = ax.contourf(X, Y, Zs, levels=levels, cmap="RdYlBu_r", extend="both")
    cbar = fig.colorbar(cf, ax=ax, label="Relative conductivity change (a.u.)")
    _ = cbar

    # bowl boundary
    ring = plt.Circle((0, 0), 1.0, fill=False, color="black", lw=1.3)
    ax.add_patch(ring)

    # optional contour of strong anomaly
    if np.any(valid):
        t = 0.65 * vmax
        try:
            ax.contour(X, Y, Zs, levels=[-t, t], colors=["#1f4e79", "#8b0000"], linewidths=1.1)
        except Exception:
            pass

    # electrodes
    for i, (ex, ey) in enumerate(electrode_positions(n)):
        ax.plot([ex], [ey], "ko", ms=4)
        ax.text(1.08 * ex, 1.08 * ey, str(i), ha="center", va="center", fontsize=9)

    if np.isfinite(hx):
        ax.plot([hx], [hy], marker="*", color="magenta", ms=13)
        ax.text(hx + 0.04, hy + 0.04, "hotspot", color="magenta", fontsize=10)

    ax.set_title("GREIT Object Map (model-based, drift-rejected)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-1.03, 1.03)
    ax.set_ylim(-1.03, 1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "greit_object_map.png"), dpi=220)
    plt.close(fig)

    # Save a simpler heatmap variant too.
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    im = ax.imshow(
        Zs,
        origin="upper",
        extent=[float(np.nanmin(X)), float(np.nanmax(X)), float(np.nanmin(Y)), float(np.nanmax(Y))],
        cmap="RdYlBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
    )
    fig.colorbar(im, ax=ax, label="Relative conductivity change (a.u.)")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="black", lw=1.3))
    for i, (ex, ey) in enumerate(electrode_positions(n)):
        ax.plot([ex], [ey], "ko", ms=4)
        ax.text(1.08 * ex, 1.08 * ey, str(i), ha="center", va="center", fontsize=9)
    if np.isfinite(hx):
        ax.plot([hx], [hy], marker="*", color="magenta", ms=13)
    ax.set_title("GREIT Heatmap")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "greit_heatmap.png"), dpi=220)
    plt.close(fig)

    with open(os.path.join(args.out_dir, "summary_greit.txt"), "w", encoding="utf-8") as f:
        f.write("GREIT Reconstruction Summary\n")
        f.write("============================\n")
        f.write(f"num_electrodes: {n}\n")
        f.write(f"drive_pattern: {args.drive_pattern}\n")
        f.write(f"meas_pattern: {args.meas_pattern}\n")
        f.write(f"active_channels: {len(v0)}\n")
        f.write(f"baseline_frames_total: {len(parsed_b.frames)}\n")
        f.write(f"baseline_settled_range: [{sel_b.settled_start}, {sel_b.settled_end}]\n")
        f.write(f"baseline_settled_frames: {len(sel_b.selected_frames)}\n")
        f.write(f"phantom_frames_total: {len(parsed_p.frames)}\n")
        f.write(f"phantom_settled_range: [{sel_p.settled_start}, {sel_p.settled_end}]\n")
        f.write(f"phantom_settled_frames: {len(sel_p.selected_frames)}\n")
        f.write(f"dropped_incomplete_baseline_frames: {parsed_b.dropped_frames}\n")
        f.write(f"dropped_incomplete_phantom_frames: {parsed_p.dropped_frames}\n")
        f.write(f"frame_normalize: {use_frame_norm}\n")
        f.write(f"remove_common_mode: {use_common_mode}\n")
        f.write(f"dv_rel_mean_pct: {100.0 * dv_rel_mean:.6f}\n")
        f.write(f"dv_rel_std_pct: {100.0 * dv_rel_std:.6f}\n")
        f.write(f"greit_lambda: {args.greit_lambda}\n")
        f.write(f"greit_p: {args.greit_p}\n")
        f.write(f"greit_grid: {args.greit_grid}\n")
        f.write(f"mesh_h0: {args.mesh_h0}\n")
        f.write(f"clip_percentile: {args.clip_percentile}\n")
        f.write(f"blur_sigma_px: {args.blur_sigma}\n")
        f.write("\n")
        f.write("Hotspot (|image|-weighted centroid)\n")
        f.write("----------------------------------\n")
        f.write(f"x: {hx:.6f}\n")
        f.write(f"y: {hy:.6f}\n")
        f.write(f"radius: {hr:.6f}\n")

    print("GREIT reconstruction complete")
    print(f"  out_dir: {os.path.abspath(args.out_dir)}")
    print(f"  active_channels: {len(v0)}")
    print(f"  settled baseline frames: {len(sel_b.selected_frames)}")
    print(f"  settled phantom frames: {len(sel_p.selected_frames)}")
    print(f"  dv_rel mean/std (%): {100.0 * dv_rel_mean:.4f} / {100.0 * dv_rel_std:.4f}")
    print(f"  hotspot (x,y,r): ({hx:.3f}, {hy:.3f}, {hr:.3f})")
    print("  main_plot: greit_object_map.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())