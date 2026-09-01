"""
Step 2: compare lap times between two drivers over the race.

The raw lap times include safety car laps (160s+) and pit out-laps, which are
30-50s slower than a normal lap. If I plot those as-is, the y-axis stretches
to fit them and the actual racing pace turns into a flat line. So I flag them
and plot them differently instead of silently dropping them.
"""

import matplotlib.pyplot as plt
import pandas as pd

import f1data


def load_laps(driver_number):
    """Get laps for one driver as a DataFrame, with a 'clean' flag."""
    data = f1data.get("laps",
                      session_key=f1data.SESSION_KEY,
                      driver_number=driver_number)
    df = pd.DataFrame(data)

    # Lap 1 has no duration in OpenF1 sometimes (timing starts at the line),
    # and any lap with a null duration is useless to me.
    df = df.dropna(subset=["lap_duration"])

    # A "clean" lap = green flag, not an out-lap. I use the driver's own median
    # as the yardstick rather than a hardcoded number, so this still works if I
    # switch to a different circuit later.
    median = df["lap_duration"].median()
    df["clean"] = (
        (~df["is_pit_out_lap"])
        & (df["lap_duration"] < median * 1.07)   # 7% slower = not racing
    )
    return df


def load_pit_laps(driver_number):
    """Lap numbers where the driver pitted, from the stints endpoint."""
    stints = f1data.get("stints",
                        session_key=f1data.SESSION_KEY,
                        driver_number=driver_number)
    # A pit stop happens at the end of every stint except the last one.
    return [s["lap_end"] for s in stints[:-1]]


def main():
    fig, ax = plt.subplots(figsize=(12, 6))

    all_clean = []
    for num, code in f1data.DRIVERS.items():
        df = load_laps(num)
        color = f1data.COLORS[num]
        clean = df[df["clean"]]
        dirty = df[~df["clean"]]

        ax.plot(clean["lap_number"], clean["lap_duration"],
                "o-", color=color, label=code, markersize=4, linewidth=1.5)
        # Show the excluded laps as hollow markers so it's obvious they exist
        # and I'm not hiding data.
        ax.plot(dirty["lap_number"], dirty["lap_duration"],
                "o", color=color, markerfacecolor="none", markersize=5, alpha=0.5)

        for lap in load_pit_laps(num):
            ax.axvline(lap, color=color, linestyle=":", alpha=0.6)

        all_clean.append(clean)
        print(f"{code}: {len(clean)} clean laps, "
              f"best {clean['lap_duration'].min():.3f}s, "
              f"median {clean['lap_duration'].median():.3f}s")

    # Zoom the y-axis to the racing laps, with a little headroom. The excluded
    # laps sit above the top of the axis on purpose.
    fastest = min(c["lap_duration"].min() for c in all_clean)
    ax.set_ylim(fastest - 1, fastest + 12)

    ax.set_xlabel("Lap number")
    ax.set_ylabel("Lap time (seconds)")
    ax.set_title("Lap time comparison - Belgian GP 2026\n"
                 "(dotted lines = pit stops; safety car / out-laps excluded, "
                 "off-scale above)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/lap_times.png", dpi=130)
    print("saved plots/lap_times.png")


if __name__ == "__main__":
    main()
