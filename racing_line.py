"""
Step 3: draw the path a car took around the track, colored by speed.

The hard part isn't the drawing, it's getting the data into one table:
  location  -> x, y, z  (where the car is)     ~4 Hz
  car_data  -> speed    (what the car is doing) ~4 Hz

These are two different endpoints, sampled independently, and their timestamps
never match exactly (I saw .083 vs .057 in the same moment). There's no shared
key to join on. pandas.merge_asof does a "nearest timestamp" join, which is
exactly the tool for this.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import LineCollection

import f1data
from lap_times import load_laps


def fastest_clean_lap(driver_number):
    """Return the row for this driver's quickest green-flag lap."""
    df = load_laps(driver_number)
    clean = df[df["clean"]]
    return clean.loc[clean["lap_duration"].idxmin()]


def specific_lap(driver_number, lap_number):
    """
    Return a given lap number for this driver.

    Comparing each driver's personal best is misleading: their fastest laps
    come from different points in the race, on different fuel loads and
    different tires. Picking the SAME lap number for both fixes fuel load and
    track evolution, so what's left is closer to a fair comparison.

    It does NOT equalise tire compound or tire age - the two cars are on
    different strategies, so that difference is real and stays in the data.
    """
    df = load_laps(driver_number)
    match = df[df["lap_number"] == lap_number]
    if match.empty:
        raise SystemExit(f"driver {driver_number} has no lap {lap_number}")
    row = match.iloc[0]
    if not row["clean"]:
        print(f"  warning: lap {lap_number} for driver {driver_number} is "
              f"not a clean lap ({row['lap_duration']:.1f}s) - "
              f"safety car or pit lap?")
    return row


def lap_telemetry(driver_number, lap):
    """
    Fetch location + car_data for a single lap and merge them into one table.

    `lap` is a row from the laps endpoint: it gives me date_start and
    lap_duration, which is how I turn "lap 44" into a time window I can
    query the telemetry endpoints with.
    """
    start = pd.to_datetime(lap["date_start"])
    end = start + pd.Timedelta(seconds=lap["lap_duration"])

    # OpenF1 wants ISO strings. I pad the window slightly on each end so I
    # don't lose the first/last sample to rounding.
    params = {
        "session_key": f1data.SESSION_KEY,
        "driver_number": driver_number,
        "date>": (start - pd.Timedelta(seconds=1)).isoformat(),
        "date<": (end + pd.Timedelta(seconds=1)).isoformat(),
    }

    loc = pd.DataFrame(f1data.get("location", **params))
    car = pd.DataFrame(f1data.get("car_data", **params))

    # merge_asof needs both sides sorted by the join key.
    loc["date"] = pd.to_datetime(loc["date"])
    car["date"] = pd.to_datetime(car["date"])
    loc = loc.sort_values("date")
    car = car.sort_values("date")

    merged = pd.merge_asof(
        loc,
        car[["date", "speed", "throttle", "brake", "n_gear"]],
        on="date",
        direction="nearest",              # closest sample either side
        tolerance=pd.Timedelta("0.5s"),   # refuse to match if nothing is close
    )

    # If a location sample had no car_data within 0.5s, speed is NaN. Drop
    # those rather than plotting a line segment with no color.
    before = len(merged)
    merged = merged.dropna(subset=["speed"])
    if before != len(merged):
        print(f"  dropped {before - len(merged)} rows with no speed match")

    # x/y are in decimetres (I checked: ~9.9 units per metre against Spa's
    # official 7004 m lap). Convert to metres so the axes mean something.
    merged["x_m"] = merged["x"] / 10.0
    merged["y_m"] = merged["y"] / 10.0

    return merged


def draw_line(ax, df, title, vmin, vmax):
    """Draw one racing line as segments colored by speed."""
    # A LineCollection lets each little segment have its own color, which is
    # what makes the speed gradient work. A normal plot() can only do one color.
    points = df[["x_m", "y_m"]].to_numpy().reshape(-1, 1, 2)
    segments = [[points[i][0], points[i + 1][0]] for i in range(len(points) - 1)]

    lc = LineCollection(segments, cmap="plasma", linewidth=3)
    # Color each segment by the speed at its starting point.
    lc.set_array(df["speed"].to_numpy()[:-1])
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    return lc


def main(lap_number=None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    laps, telem = {}, {}
    for num in f1data.DRIVERS:
        if lap_number is None:
            lap = fastest_clean_lap(num)
            print(f"{f1data.DRIVERS[num]}: fastest clean lap = "
                  f"lap {int(lap['lap_number'])} ({lap['lap_duration']:.3f}s)")
        else:
            lap = specific_lap(num, lap_number)
            print(f"{f1data.DRIVERS[num]}: lap {lap_number} "
                  f"({lap['lap_duration']:.3f}s)")
        laps[num] = lap
        telem[num] = lap_telemetry(num, lap)
        print(f"  {len(telem[num])} merged samples, "
              f"speed {telem[num]['speed'].min():.0f}-"
              f"{telem[num]['speed'].max():.0f} km/h")

    # Both plots share one color scale, otherwise the colors aren't comparable
    # between drivers - which is the whole point of putting them side by side.
    vmin = min(t["speed"].min() for t in telem.values())
    vmax = max(t["speed"].max() for t in telem.values())

    for ax, num in zip(axes, f1data.DRIVERS):
        lap = laps[num]
        n = int(lap["lap_number"])
        compound, age = f1data.stint_at_lap(num, n)
        subtitle = f"{compound}, {age} laps old" if compound else ""
        lc = draw_line(
            ax, telem[num],
            f"{f1data.DRIVERS[num]} - lap {n} "
            f"({lap['lap_duration']:.3f}s)\n{subtitle}",
            vmin, vmax,
        )

    cbar = fig.colorbar(lc, ax=axes, orientation="horizontal",
                        fraction=0.05, pad=0.06)
    cbar.set_label("Speed (km/h)")

    if lap_number is None:
        mode = "each driver's fastest clean lap (different fuel loads - see README)"
        outfile = "plots/racing_line.png"
    else:
        mode = f"both drivers on lap {lap_number} (same fuel load, same track state)"
        outfile = f"plots/racing_line_lap{lap_number}.png"

    fig.suptitle(f"Racing line colored by speed - Spa-Francorchamps 2026\n{mode}",
                 fontsize=13)
    plt.savefig(outfile, dpi=130, bbox_inches="tight")
    print(f"saved {outfile}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lap", type=int, default=None,
                    help="compare both drivers on this lap number "
                         "(default: each driver's own fastest lap)")
    args = ap.parse_args()
    main(args.lap)
