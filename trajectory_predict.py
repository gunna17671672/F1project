# Predicting where a car will be ~1 second from now, using its recent
# GPS position + speed. First real ML part of this project (everything
# before this was just plots).

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/telemetry_full_race.csv")
df["date"] = pd.to_datetime(df["date"], format="ISO8601")

# sort first or "previous row" doesn't mean "a moment ago"
df = df.sort_values(["driver_code", "lap_number", "date"]).reset_index(drop=True)

print(f"loaded {len(df)} rows, columns: {list(df.columns)}")

# --- lag features: what the car was doing a moment ago ---
# sklearn just sees a table of numbers, it has no idea about time order.
# so to let the model use "recent history," I have to copy that history
# into each row myself as its own column. grouping by driver+lap so a lag
# never jumps across a lap boundary or between two different cars.
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


# --- the target: where is the car a few samples later? ---
# 4 rows ahead ~= 1 second at our sample rate (not exact, samples aren't
# perfectly evenly spaced - fine for now, could tighten this up later)
HORIZON_STEPS = 4

group = df.groupby(["driver_code", "lap_number"])
df["x_future"] = group["x_m"].shift(-HORIZON_STEPS)
df["y_future"] = group["y_m"].shift(-HORIZON_STEPS)
df["date_future"] = group["date"].shift(-HORIZON_STEPS)

# predicting the CHANGE in position, not the absolute future x/y - makes
# more sense next to a velocity-based baseline
df["dx_future"] = df["x_future"] - df["x_m"]
df["dy_future"] = df["y_future"] - df["y_m"]
df["dt_future"] = (df["date_future"] - df["date"]).dt.total_seconds()

before = len(df)
df = df.dropna(subset=["dx_future", "dy_future"]).reset_index(drop=True)
print(f"\ndropped {before - len(df)} rows with no future to predict "
      f"(last {HORIZON_STEPS} samples of each lap)")
print(f"{len(df)} rows remain, average horizon = {df.dt_future.mean():.2f}s")

# rough velocity from our two GPS points (real distance / real time, not
# assuming every sample gap is identical)
df["vx"] = (df["x_m"] - df["x_prev1"]) / df["dt_prev1"]
df["vy"] = (df["y_m"] - df["y_prev1"]) / df["dt_prev1"]

# --- catching a data problem ---
# my first version of this model LOST to a dumb baseline. dug into the
# fitted coefficients to see why, and found some rows had an implied
# speed in the thousands of m/s - impossible, F1 tops out around 97 m/s.
# bad GPS samples, not a bug in my math (checked - the time gaps for
# these rows look totally normal, so it's not a divide-by-tiny-number
# thing either). least squares is sensitive to outliers, so a handful of
# these were messing up the whole fit. filtering them out on a realistic
# top speed fixed it.
TOP_SPEED_MS = 100  # ~360 km/h, well above any real F1 speed

cur_speed = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
fut_speed = np.sqrt(df["dx_future"] ** 2 + df["dy_future"] ** 2) / df["dt_future"]
bad = ~np.isfinite(cur_speed) | (cur_speed > TOP_SPEED_MS) | (fut_speed > TOP_SPEED_MS)
if bad.any():
    print(f"dropping {bad.sum()} rows ({100*bad.mean():.1f}%) with an "
          f"impossible implied speed (GPS glitch)")
    df = df[~bad].reset_index(drop=True)

# --- baseline: assume the car just keeps doing what it's doing ---
df["baseline_dx"] = df["vx"] * df["dt_future"]
df["baseline_dy"] = df["vy"] * df["dt_future"]

# --- features for the actual model ---
# gave it the baseline's own guess as an input (a linear model can't
# multiply velocity x time itself - that's a product, linear regression
# only does weighted sums - so handing it the baseline directly and
# letting it learn a correction on top works better than handing it the
# raw pieces separately)
FEATURES = ["x_m", "y_m", "speed", "baseline_dx", "baseline_dy"]
TARGETS = ["dx_future", "dy_future"]

# --- split: hold out whole laps, not random rows ---
# rows are ~0.25s apart and almost identical to their neighbors. a random
# split would put near-duplicate rows on both sides and the model could
# basically cheat. holding out entire laps means the test laps are ones
# it's never seen anything from.
test_mask = (df["lap_number"] % 4 == 0)
train, test = df[~test_mask], df[test_mask]
print(f"\ntrain: {len(train)} rows, "
      f"{train.groupby(['driver_code','lap_number']).ngroups} laps")
print(f"test:  {len(test)} rows, "
      f"{test.groupby(['driver_code','lap_number']).ngroups} laps "
      f"(lap numbers: {sorted(test.lap_number.unique())})")

# --- fit the model ---
# .fit() finds the weights that make the model's predictions match the
# real targets as closely as possible on the training rows. for plain
# linear regression there's an exact formula for the best weights - no
# randomness, nothing to tune, it just solves it.
model = LinearRegression()
model.fit(train[FEATURES], train[TARGETS])
pred = model.predict(test[FEATURES])

# --- score it ---
# displacement error = straight-line distance in metres between predicted
# and actual future position. this is basically ADE (average displacement
# error, the standard metric for this kind of thing) but simplified -
# real ADE averages over a whole predicted path, this is just one horizon.
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
