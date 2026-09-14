# predicting where a car will be ~1 sec from now based on its recent
# gps position + speed. first real ML step, everything before this was
# just plots

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/telemetry_full_race.csv")
df["date"] = pd.to_datetime(df["date"], format="ISO8601")
df = df.sort_values(["driver_code", "lap_number", "date"]).reset_index(drop=True)

print(f"loaded {len(df)} rows, columns: {list(df.columns)}")

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
print(f"dropped {before - len(df)} rows with no lag history "
      f"({df.groupby(['driver_code','lap_number']).ngroups} driver-lap groups)")
print(f"{len(df)} rows remain")

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
print(f"\ndropped {before - len(df)} rows with no future to predict "
      f"(last {HORIZON_STEPS} samples of each lap)")
print(f"{len(df)} rows remain, average horizon = {df.dt_future.mean():.2f}s")

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
if bad.any():
    print(f"dropping {bad.sum()} rows ({100*bad.mean():.1f}%) with an "
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
print(f"\ntrain: {len(train)} rows, "
      f"{train.groupby(['driver_code','lap_number']).ngroups} laps")
print(f"test:  {len(test)} rows, "
      f"{test.groupby(['driver_code','lap_number']).ngroups} laps "
      f"(lap numbers: {sorted(test.lap_number.unique())})")

model = LinearRegression()
model.fit(train[FEATURES], train[TARGETS])
pred = model.predict(test[FEATURES])

# straight-line error in meters between predicted and real future spot.
# basically a simplified ADE (the standard metric for this stuff)
def displacement_error(dx_pred, dy_pred, dx_true, dy_true):
    return np.sqrt((dx_pred - dx_true) ** 2 + (dy_pred - dy_true) ** 2)

model_err = displacement_error(pred[:, 0], pred[:, 1],
                               test["dx_future"], test["dy_future"])
baseline_err = displacement_error(test["baseline_dx"], test["baseline_dy"],
                                  test["dx_future"], test["dy_future"])

print(f"\nmean displacement error - baseline (constant velocity): "
      f"{baseline_err.mean():.2f} m")
print(f"mean displacement error - linear regression model:      "
      f"{model_err.mean():.2f} m")
improvement = 100 * (1 - model_err.mean() / baseline_err.mean())
print(f"model beats the baseline by {improvement:.1f}%")

df.to_csv("data/telemetry_with_lags.csv", index=False)
