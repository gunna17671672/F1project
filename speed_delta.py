"""
Where on track is one driver actually faster than the other?

lap_times.py answers "who was faster overall." This answers "faster where."
The racing line plots show speed but you can't compare two side-by-side maps
point by point. This module lines both drivers up on a shared axis so they
can be compared directly.

The trick: GPS samples land at different points on track for each driver (4Hz
sampling, slightly different lines, slightly different speeds), so there's no
shared index to compare on - same problem as the location/car_data merge, one
level up. The fix is similar in spirit: reduce both laps to "fraction of the
way around the track" (0 to 1, via cumulative distance along each driver's own
path) and interpolate both onto one common grid of that fraction. Once both
speeds are sampled at the same 400 points around the lap, they're directly
comparable.
"""

import numpy as np
import pandas as pd

import f1data
from racing_line import lap_telemetry, specific_lap

N_GRID = 400          # points around the lap to compare at
N_SEGMENTS = 40        # chunks the track gets split into for the dominance map


def arc_length_fraction(df):
    """
    Turn a driver's (x_m, y_m) trace into 'fraction of the lap completed'
    at each sample, by summing the straight-line distance between
    consecutive GPS points. Returns (fraction array, total lap length in m).
    """
    xy = df[["x_m", "y_m"]].to_numpy()
    step = np.sqrt(((xy[1:] - xy[:-1]) ** 2).sum(axis=1))
    dist = np.concatenate([[0.0], np.cumsum(step)])
    total = dist[-1]
    return dist / total, total


def build_comparison(session_key, drivers, lap_number):
    """
    Fetch + align both drivers' telemetry for one lap onto a common
    0..1 "fraction around the track" axis.

    Returns a dict with the common grid, each driver's interpolated speed
    and (x, y) reference path, plus summary numbers.
    """
    if len(drivers) != 2:
        raise ValueError("speed delta needs exactly two drivers")
    a, b = drivers

    raw, lengths = {}, {}
    for num in drivers:
        lap = specific_lap(num, lap_number, session_key)
        if lap is None:
            raise ValueError(f"{f1data.driver_code(session_key, num)} has "
                             f"no lap {lap_number}")
        t = lap_telemetry(num, lap, session_key)
        if len(t) < 20:
            raise ValueError(f"{f1data.driver_code(session_key, num)} has "
                             f"no telemetry for lap {lap_number}")
        frac, total = arc_length_fraction(t)
        raw[num] = (frac, t)
        lengths[num] = total

    grid = np.linspace(0, 1, N_GRID)
    speed = {n: np.interp(grid, raw[n][0], raw[n][1]["speed"].to_numpy())
             for n in drivers}
    # Use driver A's path as the reference line to draw the track shape -
    # doesn't matter much which one, the two lines are near-identical.
    xs = np.interp(grid, raw[a][0], raw[a][1]["x_m"].to_numpy())
    ys = np.interp(grid, raw[a][0], raw[a][1]["y_m"].to_numpy())

    delta = speed[b] - speed[a]   # positive = b faster at that point

    # Split into N_SEGMENTS chunks and decide, per chunk, who's faster on
    # average. Chunking rather than coloring point-by-point avoids a
    # flickery map from sample noise.
    edges = np.linspace(0, N_GRID, N_SEGMENTS + 1).astype(int)
    seg_winner, seg_margin = [], []
    for i in range(N_SEGMENTS):
        chunk = delta[edges[i]:edges[i + 1]]
        m = chunk.mean()
        seg_winner.append(b if m > 0 else a)
        seg_margin.append(abs(m))

    return {
        "grid": grid, "xs": xs, "ys": ys, "speed": speed, "delta": delta,
        "lengths": lengths, "edges": edges,
        "seg_winner": seg_winner, "seg_margin": seg_margin,
        "driver_a": a, "driver_b": b,
        "pct_b_faster": float((delta > 0).mean() * 100),
    }


def build_figure(session_key=None, drivers=None, lap_number=None):
    """
    Two-panel figure: a track map colored by who was faster in each chunk,
    and a speed-delta trace showing the margin around the lap.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    session_key = session_key or f1data.DEFAULT_SESSION
    drivers = drivers or f1data.DEFAULT_DRIVERS
    if lap_number is None:
        raise ValueError("speed delta needs a specific lap number for both "
                         "drivers - use the same-lap comparison")

    r = build_comparison(session_key, drivers, lap_number)
    a, b = r["driver_a"], r["driver_b"]
    colors = f1data.driver_colors(session_key, drivers)
    code_a, code_b = f1data.driver_code(session_key, a), f1data.driver_code(session_key, b)

    fig, (ax_map, ax_delta) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.3]})

    # --- left: dominance map ---
    points = np.column_stack([r["xs"], r["ys"]]).reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = []
    edges = r["edges"]
    for i in range(N_SEGMENTS):
        c = colors[r["seg_winner"][i]]
        seg_colors += [c] * (edges[i + 1] - edges[i])
    seg_colors = seg_colors[:len(segs)]

    lc = LineCollection(segs, colors=seg_colors, linewidth=4)
    ax_map.add_collection(lc)
    ax_map.set_xlim(r["xs"].min() - 200, r["xs"].max() + 200)
    ax_map.set_ylim(r["ys"].min() - 200, r["ys"].max() + 200)
    ax_map.set_aspect("equal")
    ax_map.axis("off")
    ax_map.set_title(f"Faster where - lap {lap_number}", fontsize=11)
    ax_map.plot([], [], color=colors[a], linewidth=4, label=code_a)
    ax_map.plot([], [], color=colors[b], linewidth=4, label=code_b)
    ax_map.legend(loc="upper right", fontsize=9)

    # --- right: delta trace, filled toward whoever's ahead ---
    ax_delta.axhline(0, color="grey", linewidth=1)
    ax_delta.fill_between(r["grid"], r["delta"], 0,
                          where=r["delta"] >= 0, color=colors[b], alpha=0.6,
                          interpolate=True)
    ax_delta.fill_between(r["grid"], r["delta"], 0,
                          where=r["delta"] <= 0, color=colors[a], alpha=0.6,
                          interpolate=True)
    ax_delta.plot(r["grid"], r["delta"], color="white", linewidth=0.6, alpha=0.4)
    ax_delta.set_xlim(0, 1)
    ax_delta.set_xlabel("Fraction around the lap")
    ax_delta.set_ylabel(f"Speed delta (km/h)  [{code_b} - {code_a}]")
    ax_delta.set_title(f"{code_b} faster above the line, "
                       f"{code_a} faster below", fontsize=11)
    ax_delta.grid(alpha=0.2)

    fig.suptitle(f"{code_a} vs {code_b} - speed delta, lap {lap_number}",
                fontsize=13)
    fig.tight_layout()
    return fig, r


def main():
    fig, r = build_figure(lap_number=25)
    code_a = f1data.driver_code(f1data.DEFAULT_SESSION, r["driver_a"])
    code_b = f1data.driver_code(f1data.DEFAULT_SESSION, r["driver_b"])
    print(f"{code_b} faster on {r['pct_b_faster']:.0f}% of the lap")
    fig.savefig("plots/speed_delta.png", dpi=130, bbox_inches="tight")
    print("saved plots/speed_delta.png")


if __name__ == "__main__":
    main()
