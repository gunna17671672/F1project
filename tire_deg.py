"""
Step 4: how much does lap time fall off as a set of tires gets older?

The stints endpoint tells me which laps ran on which compound. If I plot lap
time against *tire age* instead of lap number, all the stints line up at zero
and I can compare how fast each compound falls away.

IMPORTANT - the fuel problem:
My first version of this gave NEGATIVE degradation, i.e. tires apparently
getting faster with age. That's not physically possible. The cause is fuel
burn: a car starts the race ~100 kg heavier than it finishes, and a lighter
car is quicker. That improvement runs opposite to tire wear and, over a long
stint, is bigger than it.

So raw lap times measure (tire wear - fuel burn), not tire wear. To separate
them I add the fuel effect back on, normalising every lap to what it would
have been on a full tank.

Caveat I should be upfront about: real fuel loads are not public, and OpenF1
does not publish them. FUEL_EFFECT_S_PER_LAP below is a rule-of-thumb
estimate, not a measurement. Every corrected number depends on it.
"""

import matplotlib.pyplot as plt
import numpy as np

import f1data
from lap_times import load_laps

# Seconds per lap the car gains purely from burning fuel and getting lighter.
# Rule of thumb: ~0.3 s per 10 kg, and ~1.8-2.2 kg burned per lap at Spa,
# which lands around 0.06 s/lap. This is the single biggest assumption in the
# whole project - change it here and every corrected slope moves with it.
FUEL_EFFECT_S_PER_LAP = 0.06

COMPOUND_STYLE = {
    "SOFT": "-",
    "MEDIUM": "--",
    "HARD": ":",
}


def stint_data(driver_number):
    """Yield (stint_info, laps_dataframe) for each stint this driver ran."""
    laps = load_laps(driver_number)
    stints = f1data.get("stints",
                        session_key=f1data.SESSION_KEY,
                        driver_number=driver_number)

    for s in stints:
        # Only green-flag laps, or safety car periods flatten the slope.
        in_stint = laps[
            (laps["lap_number"] >= s["lap_start"])
            & (laps["lap_number"] <= s["lap_end"])
            & (laps["clean"])
        ].copy()

        # Tire age = laps completed on this set. tyre_age_at_start handles the
        # case where a driver started on a used set from qualifying.
        in_stint["tyre_age"] = (
            in_stint["lap_number"] - s["lap_start"] + s["tyre_age_at_start"]
        )

        # Fuel correction. Lap N has burned (N-1) laps of fuel, so it is
        # already (N-1) * effect seconds quicker than it would have been on a
        # full tank. Adding that back puts every lap on the same fuel load.
        in_stint["fuel_corrected"] = (
            in_stint["lap_duration"]
            + (in_stint["lap_number"] - 1) * FUEL_EFFECT_S_PER_LAP
        )

        yield s, in_stint


def main():
    fig, ax = plt.subplots(figsize=(11, 6.5))

    print(f"assuming fuel effect = {FUEL_EFFECT_S_PER_LAP} s/lap\n")
    print(f"{'driver':<7}{'stint':<7}{'compound':<10}{'laps':<7}"
          f"{'raw':<10}{'fuel-corrected':<15}")
    print("-" * 56)

    for num, code in f1data.DRIVERS.items():
        color = f1data.COLORS[num]

        for s, df in stint_data(num):
            if len(df) < 4:
                print(f"{code:<7}{s['stint_number']:<7}{s['compound']:<10}"
                      f"{len(df):<7}{'(too short to fit)':<25}")
                continue

            style = COMPOUND_STYLE.get(s["compound"], "-")
            ax.plot(df["tyre_age"], df["fuel_corrected"],
                    "o", color=color, markersize=4, alpha=0.6)

            # Fit lap_time = slope * age + intercept. slope IS the degradation.
            raw_slope, _ = np.polyfit(df["tyre_age"], df["lap_duration"], 1)
            slope, intercept = np.polyfit(df["tyre_age"], df["fuel_corrected"], 1)

            xs = np.array([df["tyre_age"].min(), df["tyre_age"].max()])
            ax.plot(xs, slope * xs + intercept,
                    style, color=color, linewidth=2,
                    label=f"{code} {s['compound']} ({slope:+.3f} s/lap)")

            print(f"{code:<7}{s['stint_number']:<7}{s['compound']:<10}"
                  f"{len(df):<7}{raw_slope:+.3f}    {slope:+.3f}")

    ax.set_xlabel("Tire age (laps on this set)")
    ax.set_ylabel(f"Fuel-corrected lap time (s, normalised to lap 1 fuel load)")
    ax.set_title("Tire degradation by stint - Belgian GP 2026\n"
                 f"lap times corrected for fuel burn at "
                 f"{FUEL_EFFECT_S_PER_LAP} s/lap (estimated)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/tire_deg.png", dpi=130)
    print("\nsaved plots/tire_deg.png")


if __name__ == "__main__":
    main()
