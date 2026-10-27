"""
analyze_breathing.py - breathing rate from a recording made with imu_logger.py.

Group 2, ACIT4035 - clip-on dual-IMU breathing-rate monitor.

Method A from the proposal:
  1. Orientation of each sensor from its quaternions.
  2. Rotation of the FRONT sensor relative to the BACK sensor. Movement of the
     whole trunk turns both sensors together and largely cancels out; breathing
     moves the abdomen relative to the spine and remains.
  3. Band-pass filter (default 0.1-0.7 Hz = 6-42 breaths/min).
  4. Sliding 32 s windows, moved 8 s at a time. In each window the frequency
     spectrum (FFT) is computed and its highest peak is the breathing frequency.
     breaths/min = frequency (Hz) x 60.
     The three rotation axes are combined by adding their spectra, so the result
     does not depend on how the sensors are turned on the body.
  5. Per test phase (from the marker file): median of the window estimates,
     plus a simple breath count (peaks in the waveform) as a cross-check.

The same is done for the front sensor alone and the back sensor alone, so the
benefit of the relative signal can be compared.

Usage:
    python analyze_breathing.py recordings/20261023_153040_breathing01.csv
    python analyze_breathing.py rec.csv --marker 170 5      (add a missed marker: time in s, key)
    python analyze_breathing.py rec.csv --band 0.15 0.7 --window 32 --step 8

Output (next to the recording):
    <name>_analysis.png       figure
    <name>_phases.csv         one row per test phase
    <name>_windows.csv        one row per 32 s window

Needs: python -m pip install numpy scipy pandas matplotlib
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.spatial.transform import Rotation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 50.0  # Hz, the common time grid both sensors are resampled onto

# Same labels as imu_logger.py, used for markers added with --marker
MARKER_LABELS = {
    "1": "sync lean", "2": "sit, normal breathing", "3": "breath hold",
    "4": "paced 12/min", "5": "paced 18/min", "6": "paced 24/min",
    "7": "movement / rocking", "8": "walking", "9": "other",
}
# Phases that are not used to estimate breathing (and are not scored)
NOT_BREATHING = ("sync", "hold", "other")
# Phases with large body movement - left out when choosing the main breathing axis
MOVEMENT = ("movement", "rocking", "walking", "sync")

SIGNALS = ["relative", "front", "back"]
COLORS = {"relative": "#2A7F62", "front": "#4C72B0", "back": "#DD8452"}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_recording(path):
    """Read the logger CSV and resample both sensors onto one 50 Hz time grid."""
    df = pd.read_csv(path)
    names = set(df["sensor"])
    if not {"front", "back"} <= names:
        sys.exit(f"Expected sensors 'front' and 'back' in {path}, found {sorted(names)}")
    front = df[df["sensor"] == "front"]
    back = df[df["sensor"] == "back"]
    t0 = max(front["time_s"].min(), back["time_s"].min())
    t1 = min(front["time_s"].max(), back["time_s"].max())
    t = np.arange(np.ceil(t0 * FS) / FS, t1, 1 / FS)

    def rotations(d):
        # scipy wants quaternions as (x, y, z, w); WitMotion gives q0 = w
        q = np.column_stack([np.interp(t, d["time_s"], d[c]) for c in ("q1", "q2", "q3", "q0")])
        return Rotation.from_quat(q)   # from_quat normalises after interpolation

    return t, rotations(front), rotations(back)


def load_markers(rec_path, extra):
    """Read <name>_markers.csv if it exists, add markers given on the command line."""
    path = rec_path[:-4] + "_markers.csv"
    rows = []
    if os.path.exists(path):
        m = pd.read_csv(path, dtype={"key": str})
        rows = list(m[["time_s", "key", "label"]].itertuples(index=False, name=None))
    else:
        print(f"No marker file found ({path}) - the whole recording is treated as one phase.")
    for time_s, key in extra:
        rows.append((float(time_s), key, MARKER_LABELS.get(key, "other")))
        print(f"Added marker {key} ({MARKER_LABELS.get(key, 'other')}) at {float(time_s):.1f} s")
    rows.sort()
    return rows


def make_phases(markers, t_end):
    """Each marker starts a phase that lasts until the next marker."""
    phases = []
    for i, (start, key, label) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else t_end
        m = re.search(r"(\d+)\s*/\s*min", label)
        phases.append({"phase": i + 1, "label": label, "start_s": start, "end_s": end,
                       "expected_bpm": float(m.group(1)) if m else np.nan})
    if not phases:
        phases.append({"phase": 1, "label": "whole recording", "start_s": 0.0, "end_s": t_end,
                       "expected_bpm": np.nan})
    return phases


# ----------------------------------------------------------------------------
# Signal processing
# ----------------------------------------------------------------------------

def tilt_signals(t, r_front, r_back, ref_mask):
    """Small rotations (degrees, 3 axes) of front, back and front-relative-to-back,
    measured from their average orientation during the reference period."""
    r_rel = r_back.inv() * r_front          # front orientation seen from the back sensor
    out = {}
    for name, r in (("relative", r_rel), ("front", r_front), ("back", r_back)):
        ref = r[ref_mask].mean()
        out[name] = (ref.inv() * r).as_rotvec(degrees=True)   # in the sensor's own axes
    return out


def bandpass(x, low, high):
    sos = butter(4, [low, high], btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def main_axis(x, mask):
    """Direction in which the (filtered) 3-axis signal moves most = principal component.
    Used for plotting one breathing waveform and for the breath count."""
    xm = x[mask] - x[mask].mean(axis=0)
    _, _, vt = np.linalg.svd(xm, full_matrices=False)
    w = x @ vt[0]
    return w, vt[0]


def spectral_rate(seg, low, high, nfft=8192):
    """Breathing rate (breaths/min) from the highest spectral peak in one window.
    Returns rate and peak quality (share of the band's power near the peak)."""
    seg = seg - seg.mean(axis=0)
    win = np.hanning(len(seg))[:, None]
    spec = np.abs(np.fft.rfft(seg * win, n=nfft, axis=0)) ** 2
    power = spec.sum(axis=1)                      # add the three axes together
    f = np.fft.rfftfreq(nfft, 1 / FS)
    band = (f >= low) & (f <= high)
    fb, pb = f[band], power[band]
    i = int(np.argmax(pb))
    # parabolic interpolation between FFT bins for a finer estimate
    if 0 < i < len(pb) - 1:
        a, b, c = pb[i - 1], pb[i], pb[i + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        f_peak = fb[i] + shift * (fb[1] - fb[0])
    else:
        f_peak = fb[i]
    near = np.abs(fb - fb[i]) <= 0.03
    quality = pb[near].sum() / pb.sum()
    return f_peak * 60, quality


def count_breaths(w, t, start, end, high):
    """Breath count: peaks in the main-axis waveform. Rate from the mean peak spacing."""
    m = (t >= start) & (t < end)
    x = w[m]
    if len(x) < FS * 10:
        return np.nan, 0
    min_gap = int(FS / high)                       # no two breaths closer than 1/high s
    peaks, _ = find_peaks(x, distance=min_gap, prominence=0.5 * np.std(x))
    if len(peaks) < 3:
        return np.nan, len(peaks)
    tp = t[m][peaks]
    return 60 * (len(tp) - 1) / (tp[-1] - tp[0]), len(peaks)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Breathing rate from an imu_logger.py recording.")
    ap.add_argument("recording", help="CSV file written by imu_logger.py")
    ap.add_argument("--band", nargs=2, type=float, default=[0.1, 0.7], metavar=("LOW", "HIGH"),
                    help="band-pass filter in Hz (default 0.1 0.7)")
    ap.add_argument("--window", type=float, default=32, help="window length in s (default 32)")
    ap.add_argument("--step", type=float, default=8, help="window step in s (default 8)")
    ap.add_argument("--marker", nargs=2, action="append", default=[], metavar=("TIME_S", "KEY"),
                    help="add a missed marker, e.g. --marker 170 5 (can be repeated)")
    args = ap.parse_args()
    low, high = args.band

    t, r_front, r_back = load_recording(args.recording)
    markers = load_markers(args.recording, args.marker)
    phases = make_phases(markers, t[-1])

    # Reference orientation and main axis: from the quiet (non-movement) phases
    quiet = np.zeros_like(t, dtype=bool)
    for p in phases:
        if not any(k in p["label"] for k in MOVEMENT):
            quiet |= (t >= p["start_s"]) & (t < p["end_s"])
    if quiet.sum() < FS * 20:
        quiet[:] = True

    raw = tilt_signals(t, r_front, r_back, quiet)
    filt = {k: bandpass(v, low, high) for k, v in raw.items()}
    wave = {}
    for k in SIGNALS:
        wave[k], axis = main_axis(filt[k], quiet)
        print(f"{k:8s}: main breathing axis in sensor coordinates (x, y, z) = "
              f"({axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f}), "
              f"amplitude (std) = {np.std(wave[k][quiet]):.2f} deg")

    # Sliding-window spectral estimate (Method A)
    n_win, n_step = int(args.window * FS), int(args.step * FS)
    rows = []
    for i0 in range(0, len(t) - n_win + 1, n_step):
        row = {"start_s": t[i0], "center_s": t[i0] + args.window / 2, "end_s": t[i0 + n_win - 1]}
        for k in SIGNALS:
            row[f"{k}_bpm"], row[f"{k}_quality"] = spectral_rate(filt[k][i0:i0 + n_win], low, high)
        rows.append(row)
    win = pd.DataFrame(rows)

    # Per-phase summary: windows that lie completely inside the phase
    summary = []
    for p in phases:
        inside = win[(win["start_s"] >= p["start_s"]) & (win["end_s"] <= p["end_s"])]
        s = dict(p, duration_s=p["end_s"] - p["start_s"], n_windows=len(inside))
        if any(k in p["label"] for k in NOT_BREATHING):
            summary.append(s)                       # no rate for sync leans / breath holds
            continue
        for k in SIGNALS:
            s[f"{k}_bpm"] = inside[f"{k}_bpm"].median() if len(inside) else np.nan
            s[f"{k}_count_bpm"], _ = count_breaths(wave[k], t, p["start_s"], p["end_s"], high)
            if not np.isnan(p["expected_bpm"]) and len(inside):
                s[f"{k}_mae"] = (inside[f"{k}_bpm"] - p["expected_bpm"]).abs().mean()
        summary.append(s)
    summ = pd.DataFrame(summary)

    base = args.recording[:-4]
    win.round(3).to_csv(base + "_windows.csv", index=False)
    summ.round(2).to_csv(base + "_phases.csv", index=False)

    # Console table
    print(f"\nBreathing rate per phase (breaths/min). Spectral = median of {args.window:g} s windows "
          f"fully inside the phase; count = peaks in the waveform.")
    print(f"{'phase':28s} {'time (s)':>13s} {'expect':>6s} | {'relative':>8s} {'count':>5s} | "
          f"{'front':>6s} {'count':>5s} | {'back':>6s}")
    for _, s in summ.iterrows():
        f = lambda v: "   -" if pd.isna(v) else f"{v:5.1f}"
        print(f"{s['label'][:28]:28s} {s['start_s']:6.0f}-{s['end_s']:<6.0f} {f(s['expected_bpm']):>6s} | "
              f"{f(s['relative_bpm']):>8s} {f(s['relative_count_bpm']):>5s} | "
              f"{f(s['front_bpm']):>6s} {f(s['front_count_bpm']):>5s} | {f(s['back_bpm']):>6s}")
    if not phases or all(np.isnan(p["expected_bpm"]) for p in phases):
        pass
    else:
        scored = summ.dropna(subset=["expected_bpm"])
        if len(scored):
            print("\nMean absolute error over windows in paced phases: " + ", ".join(
                f"{k} {scored[f'{k}_mae'].mean():.2f}" for k in SIGNALS if f"{k}_mae" in scored))

    # Figure
    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1.2, 1.1]})
    lim = 4 * max(np.std(wave[k][quiet]) for k in SIGNALS)
    ax[0].plot(t, wave["front"], color=COLORS["front"], lw=0.8, label="front alone")
    ax[0].plot(t, wave["back"], color=COLORS["back"], lw=0.8, label="back alone")
    ax[0].set_ylabel("Tilt, filtered (°)")
    ax[0].set_ylim(-lim, lim)
    ax[0].legend(loc="lower left", frameon=False, ncol=2, fontsize=9)
    ax[1].plot(t, wave["relative"], color=COLORS["relative"], lw=0.9)
    ax[1].set_ylabel("Front relative to back,\nfiltered (°)")
    ax[1].set_ylim(-lim, lim)
    for k in SIGNALS:
        ax[2].plot(win["center_s"], win[f"{k}_bpm"], "o-", ms=3, lw=1, color=COLORS[k],
                   label=k if k != "relative" else "relative (front − back)")
    for p in phases:
        if not np.isnan(p["expected_bpm"]):
            ax[2].hlines(p["expected_bpm"], p["start_s"], p["end_s"], color="k", lw=2, alpha=0.6)
    ax[2].plot([], [], color="k", lw=2, alpha=0.6, label="paced rate")
    ax[2].set_ylabel("Breathing rate\n(breaths/min)")
    ax[2].set_ylim(0, high * 60)
    ax[2].legend(loc="upper left", frameon=False, fontsize=9, ncol=4)
    ax[2].set_xlabel("Time since start of recording (s)")
    for a in ax:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
        for i, p in enumerate(phases):
            if i % 2 == 0:
                a.axvspan(p["start_s"], p["end_s"], color="grey", alpha=0.08, lw=0)
    for p in phases:
        ax[0].text((p["start_s"] + p["end_s"]) / 2, lim * 0.98, p["label"].replace(", ", "\n"),
                   ha="center", va="top", fontsize=8, color="#444")
    ax[0].set_title(f"{os.path.basename(args.recording)} — band {low:g}–{high:g} Hz, "
                    f"{args.window:g} s windows every {args.step:g} s", loc="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(base + "_analysis.png", dpi=120)
    print(f"\nSaved {base}_analysis.png, {base}_phases.csv and {base}_windows.csv")


if __name__ == "__main__":
    main()
