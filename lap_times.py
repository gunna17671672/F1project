"""
Lap time comparison over a race.

The raw lap times include safety car laps (160s+) and pit out-laps, which are
30-50s slower than a normal lap. If I plot those as-is, the y-axis stretches
to fit them and the actual racing pace turns into a flat line. So I flag them
and plot them differently instead of silently dropping them.

build_figure() returns a matplotlib Figure so both this script and the
Streamlit app can use it.
"""

import matplotlib.pyplot as plt
import pandas as pd

import f1data


def load_laps(driver_number, session_key=None):
    """Get laps for one driver as a DataFrame, with a 'clean' flag."""
    session_key = session_key or f1data.DEFAULT_SESSION
    data = f1data.get("laps", session_key=session_key,
                      driver_number=driver_number)
    df = pd.DataFrame(data)
    if df.empty:
        return df

    # Lap 1 sometimes has no duration (timing starts at the line), and any
    # lap with a null duration is useless to me.
    df = df.dropna(subset=["lap_duration"])

    # A "clean" lap = green flag, not an out-lap. I use the driver's own
    # median as the yardstick rather than a hardcoded number, so this still
    # works at circuits with very different lap lengths.
    median = df["lap_duration"].median()
    df["clean"] = (
        (~df["is_pit_out_lap"])
        & (df["lap_duration"] < median * 1.07)   # 7% slower = not racing
    )
    return df


def load_pit_laps(driver_number, session_key=None):
    """Lap numbers where the driver pitted, from the stints endpoint."""
    session_key = session_key or f1data.DEFAULT_SESSION
    stints = f1data.get("stints", session_key=session_key,
                        driver_number=driver_number)
    # A pit stop happens at the end of every stint except the last one.
    return [s["lap_end"] for s in stints[:-1]]


def build_figure(session_key=None, drivers=None):
    """Returns (figure, list of per-driver summary dicts)."""
    session_key = session_key or f1data.DEFAULT_SESSION
    drivers = drivers or f1data.DEFAULT_DRIVERS
    colors = f1data.driver_colors(session_key, drivers)

    fig, ax = plt.subplots(figsize=(12, 6))
    summary, all_clean = [], []

    for num in drivers:
        code = f1data.driver_code(session_key, num)
        df = load_laps(num, session_key)
        if df.empty:
            continue

        color = colors[num]
        clean = df[df["clean"]]
        dirty = df[~df["clean"]]

        ax.plot(clean["lap_number"], clean["lap_duration"],
                "o-", color=color, label=code, markersize=4, linewidth=1.5)
        # Show excluded laps as hollow markers so it's obvious they exist
        # and I'm not hiding data.
        ax.plot(dirty["lap_number"], dirty["lap_duration"],
                "o", color=color, markerfacecolor="none",
                markersize=5, alpha=0.5)

        for lap in load_pit_laps(num, session_key):
            ax.axvline(lap, color=color, linestyle=":", alpha=0.6)

        all_clean.append(clean)
        summary.append({
            "driver": code,
            "clean_laps": len(clean),
            "best": round(clean["lap_duration"].min(), 3),
            "median": round(clean["lap_duration"].median(), 3),
        })

    if all_clean:
        # Zoom the y-axis to racing laps. Excluded laps sit above the top of
        # the axis on purpose.
        fastest = min(c["lap_duration"].min() for c in all_clean)
        ax.set_ylim(fastest - 1, fastest + 12)

    ax.set_xlabel("Lap number")
    ax.set_ylabel("Lap time (seconds)")
    ax.set_title("Lap time comparison\n"
                 "(dotted lines = pit stops; safety car / out-laps excluded, "
                 "off-scale above)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, summary


def main():
    fig, summary = build_figure()
    for s in summary:
        print(f"{s['driver']}: {s['clean_laps']} clean laps, "
              f"best {s['best']}s, median {s['median']}s")
    fig.savefig("plots/lap_times.png", dpi=130)
    print("saved plots/lap_times.png")


if __name__ == "__main__":
    main()
