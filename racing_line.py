"""
Draw the path a car took around the track, colored by speed.

The hard part isn't the drawing, it's getting the data into one table:
  location  -> x, y, z  (where the car is)      ~4 Hz
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


def fastest_clean_lap(driver_number, session_key=None):
    """Return the row for this driver's quickest green-flag lap."""
    df = load_laps(driver_number, session_key)
    clean = df[df["clean"]]
    if clean.empty:
        return None
    return clean.loc[clean["lap_duration"].idxmin()]


def specific_lap(driver_number, lap_number, session_key=None):
    """
    Return a given lap number for this driver.

    Comparing each driver's personal best is misleading: their fastest laps
    come from different points in the race, on different fuel loads and
    different tires. Picking the SAME lap number for both fixes fuel load and
    track evolution, so what's left is closer to a fair comparison.

    It does NOT equalise tire compound or tire age - the two cars are on
    different strategies, so that difference is real and stays in the data.
    """
    df = load_laps(driver_number, session_key)
    match = df[df["lap_number"] == lap_number]
    if match.empty:
        return None
    return match.iloc[0]


def lap_telemetry(driver_number, lap, session_key=None):
    """
    Fetch location + car_data for a single lap and merge them into one table.

    `lap` is a row from the laps endpoint: it gives me date_start and
    lap_duration, which is how I turn "lap 44" into a time window I can
    query the telemetry endpoints with.
    """
    session_key = session_key or f1data.DEFAULT_SESSION
    start = pd.to_datetime(lap["date_start"])
    end = start + pd.Timedelta(seconds=lap["lap_duration"])

    # Pad the window slightly so I don't lose the first/last sample to
    # rounding.
    params = {
        "session_key": session_key,
        "driver_number": driver_number,
        "date>": (start - pd.Timedelta(seconds=1)).isoformat(),
        "date<": (end + pd.Timedelta(seconds=1)).isoformat(),
    }

    loc = pd.DataFrame(f1data.get("location", **params))
    car = pd.DataFrame(f1data.get("car_data", **params))
    if loc.empty or car.empty:
        return pd.DataFrame()

    # merge_asof needs both sides sorted by the join key.
    #
    # format="ISO8601" matters: some sessions return timestamps with
    # microseconds ("...T14:27:10.083000+00:00") and others without
    # ("...T13:02:11+00:00"), sometimes in the same response. Without this,
    # pandas infers the format from the first row and then throws on any row
    # that doesn't match.
    loc["date"] = pd.to_datetime(loc["date"], format="ISO8601")
    car["date"] = pd.to_datetime(car["date"], format="ISO8601")
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
    # those rather than plotting a segment with no color.
    merged = merged.dropna(subset=["speed"])

    # x/y are in decimetres (I checked: ~9.9 units per metre against Spa's
    # official 7004 m lap). Convert to metres so the axes mean something.
    merged["x_m"] = merged["x"] / 10.0
    merged["y_m"] = merged["y"] / 10.0

    # OpenF1 sometimes returns a few (0,0) rows when the GPS drops out. Those
    # would draw a huge spike across the middle of the track map.
    merged = merged[(merged["x"] != 0) | (merged["y"] != 0)]

    return merged


def draw_line(ax, df, title, vmin, vmax):
    """Draw one racing line as segments colored by speed."""
    # A LineCollection lets each little segment have its own color, which is
    # what makes the gradient work. A normal plot() can only do one color.
    points = df[["x_m", "y_m"]].to_numpy().reshape(-1, 1, 2)
    segments = [[points[i][0], points[i + 1][0]] for i in range(len(points) - 1)]

    lc = LineCollection(segments, cmap="plasma", linewidth=3)
    lc.set_array(df["speed"].to_numpy()[:-1])
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")
    return lc


def build_figure(session_key=None, drivers=None, lap_number=None):
    """Returns (figure, list of per-driver info dicts)."""
    session_key = session_key or f1data.DEFAULT_SESSION
    drivers = drivers or f1data.DEFAULT_DRIVERS

    fig, axes = plt.subplots(1, len(drivers), figsize=(7.5 * len(drivers), 8))
    if len(drivers) == 1:
        axes = [axes]

    laps, telem, info = {}, {}, []
    missing = []
    for num in drivers:
        code = f1data.driver_code(session_key, num)
        lap = (fastest_clean_lap(num, session_key) if lap_number is None
               else specific_lap(num, lap_number, session_key))
        if lap is None:
            missing.append(f"{code} has no lap {lap_number}")
            continue
        t = lap_telemetry(num, lap, session_key)
        if len(t) < 20:
            # Not a crash - some sessions have real holes in their telemetry.
            # Monaco 2026, for example, is missing about 50 minutes of
            # location data in the middle of the race.
            missing.append(f"{code} has no telemetry for lap "
                           f"{int(lap['lap_number'])}")
            continue
        laps[num], telem[num] = lap, t

    if not telem:
        raise ValueError(
            "No telemetry for this lap. " + "; ".join(missing) +
            ". OpenF1's coverage has gaps in some sessions - try another lap "
            "or another race.")

    # All plots share one color scale, otherwise the colors aren't comparable
    # between drivers - which is the whole point of putting them side by side.
    vmin = min(t["speed"].min() for t in telem.values())
    vmax = max(t["speed"].max() for t in telem.values())

    lc = None
    for ax, num in zip(axes, drivers):
        if num not in telem:
            ax.axis("off")
            continue
        lap = laps[num]
        n = int(lap["lap_number"])
        code = f1data.driver_code(session_key, num)
        compound, age = f1data.stint_at_lap(session_key, num, n)
        subtitle = f"{compound}, {age} laps old" if compound else ""
        lc = draw_line(ax, telem[num],
                       f"{code} - lap {n} ({lap['lap_duration']:.3f}s)\n{subtitle}",
                       vmin, vmax)
        info.append({
            "driver": code, "lap": n,
            "lap_time": round(float(lap["lap_duration"]), 3),
            "compound": compound, "tyre_age": age,
            "samples": len(telem[num]),
            "min_speed": int(telem[num]["speed"].min()),
            "max_speed": int(telem[num]["speed"].max()),
        })

    cbar = fig.colorbar(lc, ax=axes, orientation="horizontal",
                        fraction=0.05, pad=0.06)
    cbar.set_label("Speed (km/h)")

    mode = ("each driver's fastest clean lap (different fuel loads)"
            if lap_number is None
            else f"both drivers on lap {lap_number} "
                 f"(same fuel load, same track state)")
    fig.suptitle(f"Racing line colored by speed\n{mode}", fontsize=13)
    return fig, info


def main(lap_number=None):
    fig, info = build_figure(lap_number=lap_number)
    for i in info:
        print(f"{i['driver']}: lap {i['lap']} ({i['lap_time']}s), "
              f"{i['samples']} merged samples, "
              f"speed {i['min_speed']}-{i['max_speed']} km/h")
    out = ("plots/racing_line.png" if lap_number is None
           else f"plots/racing_line_lap{lap_number}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lap", type=int, default=None,
                    help="compare both drivers on this lap number "
                         "(default: each driver's own fastest lap)")
    args = ap.parse_args()
    main(args.lap)
