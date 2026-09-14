# predicting where a car will be ~1 sec from now based on its recent
# gps position + speed. first real ML step, everything before this was
# just plots
#
# wrapped in a function so app.py can reuse it for the website tab
# without copy-pasting all this - same pattern as the other analysis
# files (lap_times.py, tire_deg.py, etc all have a build_figure()-style
# function the app calls)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def run_pipeline(verbose=True):
    def log(*a):
        if verbose:
            print(*a)

    df = pd.read_csv("data/telemetry_full_race.csv")
    df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    df = df.sort_values(["driver_code", "lap_number", "date"]).reset_index(drop=True)

    log(f"loaded {len(df)} rows, columns: {list(df.columns)}")

    # lag features = what the car was doing a moment ago. gotta group by
    # driver+lap or the shift jumps across a lap boundary / between cars
    group = df.groupby(["driver_code", "lap_number"])
    df["x_prev1"] = group["x_m"].shift(1)
    df["y_prev1"] = group["y_m"].shift(1)
    df["x_prev2"] = group["x_m"].shift(2)
    df["y_prev2"] = group["y_m"].shift(2)
    df["dt_prev1"] = (df["date"] - group["date"].shift(1)).dt.total_seconds()

    before = len(df)
    df = df.dropna(subset=["x_prev1", "y_prev1", "x_prev2", "y_prev2"]).reset_index(drop=True)
    log(f"dropped {before - len(df)} rows with no lag history "
        f"({df.groupby(['driver_code','lap_number']).ngroups} driver-lap groups)")
    log(f"{len(df)} rows remain")

    # target = position a few rows ahead. 4 rows ~ 1 sec at our sample rate
    HORIZON_STEPS = 4

    group = df.groupby(["driver_code", "lap_number"])
    df["x_future"] = group["x_m"].shift(-HORIZON_STEPS)
    df["y_future"] = group["y_m"].shift(-HORIZON_STEPS)
    df["date_future"] = group["date"].shift(-HORIZON_STEPS)

    df["dx_future"] = df["x_future"] - df["x_m"]
    df["dy_future"] = df["y_future"] - df["y_m"]
    df["dt_future"] = (df["date_future"] - df["date"]).dt.total_seconds()

    before = len(df)
    df = df.dropna(subset=["dx_future", "dy_future"]).reset_index(drop=True)
    log(f"\ndropped {before - len(df)} rows with no future to predict "
        f"(last {HORIZON_STEPS} samples of each lap)")
    log(f"{len(df)} rows remain, average horizon = {df.dt_future.mean():.2f}s")

    df["vx"] = (df["x_m"] - df["x_prev1"]) / df["dt_prev1"]
    df["vy"] = (df["y_m"] - df["y_prev1"]) / df["dt_prev1"]

    # some rows have an implied speed of literally thousands of m/s, which
    # is impossible (F1 tops out ~97 m/s). bad gps samples, not my math -
    # checked the time gaps and they look normal. had to filter these or
    # they wrecked the model's fit
    TOP_SPEED_MS = 100

    cur_speed = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    fut_speed = np.sqrt(df["dx_future"] ** 2 + df["dy_future"] ** 2) / df["dt_future"]
    bad = ~np.isfinite(cur_speed) | (cur_speed > TOP_SPEED_MS) | (fut_speed > TOP_SPEED_MS)
    n_glitch = int(bad.sum())
    glitch_pct = 100 * bad.mean()
    if bad.any():
        log(f"dropping {n_glitch} rows ({glitch_pct:.1f}%) with an "
            f"impossible implied speed (GPS glitch)")
        df = df[~bad].reset_index(drop=True)

    # baseline: just assume the car keeps doing what it's doing right now
    df["baseline_dx"] = df["vx"] * df["dt_future"]
    df["baseline_dy"] = df["vy"] * df["dt_future"]

    # giving the model the baseline guess as a feature instead of raw
    # velocity/time separately - linear regression can't multiply its own
    # inputs together so it can't reconstruct v*t on its own
    FEATURES = ["x_m", "y_m", "speed", "baseline_dx", "baseline_dy"]
    TARGETS = ["dx_future", "dy_future"]

    # split by lap, not random rows, since rows 0.25s apart are basically
    # identical and a random split would let the model cheat
    test_mask = (df["lap_number"] % 4 == 0)
    train, test = df[~test_mask], df[test_mask]
    log(f"\ntrain: {len(train)} rows, "
        f"{train.groupby(['driver_code','lap_number']).ngroups} laps")
    log(f"test:  {len(test)} rows, "
        f"{test.groupby(['driver_code','lap_number']).ngroups} laps "
        f"(lap numbers: {sorted(test.lap_number.unique())})")

    model = LinearRegression()
    model.fit(train[FEATURES], train[TARGETS])
    pred = model.predict(test[FEATURES])

    # weight the model put on the baseline guess itself - should land near
    # 1.0 if the baseline's trustworthy. it was 0.57 before I filtered the
    # bad GPS rows out, which is what tipped me off to them in the first place
    coef = pd.DataFrame(model.coef_, index=TARGETS, columns=FEATURES).round(3)
    log(f"\nweight on baseline_dx/baseline_dy (should be ~1.0):")
    log(coef[["baseline_dx", "baseline_dy"]])

    # straight-line error in meters between predicted and real future spot.
    # basically a simplified ADE (the standard metric for this stuff)
    def displacement_error(dx_pred, dy_pred, dx_true, dy_true):
        return np.sqrt((dx_pred - dx_true) ** 2 + (dy_pred - dy_true) ** 2)

    model_err = displacement_error(pred[:, 0], pred[:, 1],
                                   test["dx_future"], test["dy_future"])
    baseline_err = displacement_error(test["baseline_dx"], test["baseline_dy"],
                                      test["dx_future"], test["dy_future"])

    improvement = 100 * (1 - model_err.mean() / baseline_err.mean())

    log(f"\nmean displacement error - baseline (constant velocity): "
        f"{baseline_err.mean():.2f} m")
    log(f"mean displacement error - linear regression model:      "
        f"{model_err.mean():.2f} m")
    log(f"model beats the baseline by {improvement:.1f}%")

    df.to_csv("data/telemetry_with_lags.csv", index=False)

    return {
        "n_loaded": 39430,
        "n_glitch": n_glitch,
        "glitch_pct": glitch_pct,
        "train_rows": len(train),
        "train_laps": train.groupby(["driver_code", "lap_number"]).ngroups,
        "test_rows": len(test),
        "test_laps": test.groupby(["driver_code", "lap_number"]).ngroups,
        "avg_horizon": df["dt_future"].mean(),
        "coef": coef,
        "baseline_err": baseline_err,
        "model_err": model_err,
        "baseline_mean": baseline_err.mean(),
        "model_mean": model_err.mean(),
        "improvement": improvement,
    }


def build_error_chart(result):
    """Simple bar chart: baseline error vs model error. Used by the app tab."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    names = ["Constant velocity\n(baseline)", "Linear regression\n(model)"]
    values = [result["baseline_mean"], result["model_mean"]]
    bars = ax.bar(names, values, color=["#888888", "#E10600"])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f} m",
                ha="center", fontsize=11)
    ax.set_ylabel("Mean displacement error (m)")
    ax.set_title("Predicting position ~1s ahead")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    run_pipeline()
