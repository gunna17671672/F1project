"""
Export the raw and merged data behind this project as plain CSV files.

Everything the plots are built from ultimately comes from OpenF1. This script
pulls it out of the on-disk cache (or fetches fresh if needed) and writes it
as CSVs anyone can open in Excel/Sheets, no Python required - meant for
handing the underlying resources to someone who wants to see the data itself,
not just the plots.

Run with: ./.venv/bin/python export_data.py
"""

import time

import pandas as pd

import f1data
from lap_times import load_laps
from racing_line import lap_telemetry, specific_lap

SESSION_KEY = f1data.SESSION_KEY if hasattr(f1data, "SESSION_KEY") else f1data.DEFAULT_SESSION
DRIVERS = f1data.DEFAULT_DRIVERS

OUT = "data"


def main():
    race = [s for s in f1data.list_races(2026) if s["session_key"] == SESSION_KEY][0]

    # 1. Session info - which race this all is.
    pd.DataFrame([race]).to_csv(f"{OUT}/session_info.csv", index=False)
    print("wrote session_info.csv")

    # 2. Every driver on the grid that race (context: who NOR/VER were racing).
    drivers = f1data.list_drivers(SESSION_KEY)
    pd.DataFrame(drivers).to_csv(f"{OUT}/drivers.csv", index=False)
    print(f"wrote drivers.csv ({len(drivers)} drivers)")

    # 3. Lap-by-lap times for the two compared drivers, with the same
    #    'clean lap' flag the analysis uses.
    all_laps = []
    for num in DRIVERS:
        df = load_laps(num, SESSION_KEY)
        df.insert(0, "driver_code", f1data.driver_code(SESSION_KEY, num))
        all_laps.append(df)
    pd.concat(all_laps).to_csv(f"{OUT}/laps.csv", index=False)
    print(f"wrote laps.csv ({sum(len(d) for d in all_laps)} rows)")

    # 4. Stint / tire compound history for both drivers.
    all_stints = []
    for num in DRIVERS:
        s = pd.DataFrame(f1data.get("stints", session_key=SESSION_KEY, driver_number=num))
        s.insert(0, "driver_code", f1data.driver_code(SESSION_KEY, num))
        all_stints.append(s)
    pd.concat(all_stints).to_csv(f"{OUT}/stints.csv", index=False)
    print(f"wrote stints.csv ({sum(len(d) for d in all_stints)} rows)")

    # 5. Full-race merged telemetry (location + car_data, joined by
    #    merge_asof) for both drivers, every lap - this is the actual
    #    row-level data the racing line and speed delta plots are built
    #    from, not just the summary.
    all_telem = []
    for num in DRIVERS:
        laps_df = load_laps(num, SESSION_KEY)
        code = f1data.driver_code(SESSION_KEY, num)
        got = 0
        for _, lap in laps_df.iterrows():
            # Throttled and individually guarded: pulling 44 laps x 2
            # telemetry endpoints in a burst trips OpenF1's rate limit even
            # with retries. A short pause between laps keeps it under that,
            # and a skipped lap (rare) doesn't take the whole export down.
            try:
                t = lap_telemetry(num, lap, SESSION_KEY)
            except Exception as e:
                print(f"  skipped {code} lap {int(lap['lap_number'])}: {e}")
                time.sleep(3)
                continue
            if t.empty:
                continue
            t = t.copy()
            t.insert(0, "driver_code", code)
            t.insert(1, "lap_number", int(lap["lap_number"]))
            all_telem.append(t[["driver_code", "lap_number", "date", "x", "y", "z",
                                "x_m", "y_m", "speed", "throttle", "brake", "n_gear"]])
            got += 1
            time.sleep(0.4)
        print(f"  merged telemetry for {code}: {got} laps fetched")
    telem = pd.concat(all_telem, ignore_index=True)
    telem.to_csv(f"{OUT}/telemetry_full_race.csv", index=False)
    print(f"wrote telemetry_full_race.csv ({len(telem)} rows)")


if __name__ == "__main__":
    main()
