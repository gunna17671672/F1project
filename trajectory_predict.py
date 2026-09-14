"""
Step 1: turn the telemetry table into something a model can learn from.

sklearn doesn't know anything about "time." A model just sees a table:
each row is one example, some columns are inputs (features), one column
is the answer we want (target). It has no idea row 5 came after row 4.

Our data IS a time series though - each row is where a car was at some
instant. So if we want the model to use "recent history" to predict
what happens next, we have to build that history INTO each row ourselves.
That's what a lag feature is: for row i, "x_m_prev1" just means "whatever
x_m was one sample earlier." Once that's a column, sklearn can use it like
any other number, and time order stops mattering to the model - all the
temporal reasoning got done by us, in this feature-building step, before
the model ever sees the data.

Two things to be careful about, or the lags come out wrong:
1. If we don't sort by time first, "one row earlier" is meaningless.
2. If we don't compute lags SEPARATELY per driver and per lap, the last
   row of one lap gets "lag" from a completely different car or a
   different point in the race - a huge, fake jump in position.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/telemetry_full_race.csv")
df["date"] = pd.to_datetime(df["date"], format="ISO8601")

# Sort so "the row before this one" actually means "a moment earlier."
df = df.sort_values(["driver_code", "lap_number", "date"]).reset_index(drop=True)

print(f"loaded {len(df)} rows, columns: {list(df.columns)}")

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


# ============================================================
# Step 2: build the target, a baseline, and fit the first model
# ============================================================

# --- the target: where does the car end up a few samples from now? ---
# HORIZON_STEPS rows ahead, computed the same grouped way as the lags, so
# we never predict "off the end of the lap" into a different lap or car.
# At our ~0.25s sample gap, 4 rows is roughly 1 second - not exact, since
# sampling isn't perfectly steady (that's why we kept dt_prev1 in step 1).
# Using an exact time horizon instead of a fixed row-count is a fair
# thing to fix in week 2.
HORIZON_STEPS = 4

group = df.groupby(["driver_code", "lap_number"])
df["x_future"] = group["x_m"].shift(-HORIZON_STEPS)
df["y_future"] = group["y_m"].shift(-HORIZON_STEPS)
df["date_future"] = group["date"].shift(-HORIZON_STEPS)

# We predict the CHANGE in position, not the absolute future position -
# same information, but it lines up naturally with the baseline below,
# since a velocity is a statement about how far you move, not where
# you end up.
df["dx_future"] = df["x_future"] - df["x_m"]
df["dy_future"] = df["y_future"] - df["y_m"]
df["dt_future"] = (df["date_future"] - df["date"]).dt.total_seconds()

before = len(df)
df = df.dropna(subset=["dx_future", "dy_future"]).reset_index(drop=True)
print(f"\ndropped {before - len(df)} rows with no future to predict "
      f"(last {HORIZON_STEPS} samples of each lap)")
print(f"{len(df)} rows remain, average horizon = {df.dt_future.mean():.2f}s")

# --- features: what the model gets to see "right now" ---
# velocity estimated from our own two GPS points - real distance moved
# over real time elapsed (dt_prev1 from step 1), not an assumed fixed step.
df["vx"] = (df["x_m"] - df["x_prev1"]) / df["dt_prev1"]
df["vy"] = (df["y_m"] - df["y_prev1"]) / df["dt_prev1"]

# GPS glitch filter. Checking why the FIRST version of this model lost
# to the baseline, the model's own fitted coefficient on the baseline's
# answer came out as 0.57 instead of ~1.0 - a sign of outliers dragging
# the least-squares fit off course. The cause: a chunk of rows have an
# implied velocity in the thousands of m/s, physically impossible for a
# car (F1 tops out around 97 m/s / 350 km/h). These aren't a side-effect
# of tiny time gaps (their dt_prev1 looks completely normal) - they're
# genuine bad samples in the location feed. Squared-error fitting is very
# sensitive to a few extreme points, so a small fraction of glitched rows
# was distorting the fit for every row, not just their own. Checked the
# same thing on the FUTURE side too (a glitched future position would
# poison the target we're evaluating against, not just an input) and
# found it there as well.
TOP_SPEED_MS = 100  # ~360 km/h, comfortably above any real F1 speed

cur_speed = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
fut_speed = np.sqrt(df["dx_future"] ** 2 + df["dy_future"] ** 2) / df["dt_future"]
bad = ~np.isfinite(cur_speed) | (cur_speed > TOP_SPEED_MS) | (fut_speed > TOP_SPEED_MS)
if bad.any():
    print(f"dropping {bad.sum()} rows ({100*bad.mean():.1f}%) with a physically "
          f"impossible implied speed (GPS glitch in the location feed, "
          f"current or future side)")
    df = df[~bad].reset_index(drop=True)

# First attempt fed the model x_m, y_m, vx, vy, speed, dt_future
# separately and it LOST to the baseline. Checking the fitted coefficients
# showed why: the true relationship is displacement = velocity x time - a
# PRODUCT. Linear regression can only build a weighted SUM of whatever
# columns you hand it; it can't multiply two of its own inputs together
# unless that product is already a column. Handing it vx and dt_future
# side by side was never going to let it reconstruct vx * dt_future.
#
# The fix: hand the model the baseline's own answer (which already IS
# that product, computed correctly) as an input, and ask it to CORRECT
# that answer using where the car is on track. This is a standard pattern
# - learn a correction on top of a physics baseline, rather than asking a
# linear model to rediscover physics it structurally can't express.
FEATURES = ["x_m", "y_m", "speed", "baseline_dx", "baseline_dy"]
# (baseline_dx/dy computed a few lines below, from the now-cleaned vx/vy)
TARGETS = ["dx_future", "dy_future"]

# --- constant-velocity baseline ---
# the simplest possible guess: "the car keeps doing exactly what it's
# doing right now." predicted displacement = current velocity x how far
# ahead we're looking - using the REAL dt_future for this exact row, so
# the comparison to the baseline below is apples to apples.
df["baseline_dx"] = df["vx"] * df["dt_future"]
df["baseline_dy"] = df["vy"] * df["dt_future"]

# --- split: hold out whole laps, not random rows ---
# consecutive rows are ~0.25s apart and nearly identical. shuffling rows
# randomly would put a row in the test set right next to its near-twin
# in the training set - the model could basically look up the answer
# instead of generalizing. holding out entire laps means the test laps
# are laps the model has genuinely never seen anything from.
test_mask = (df["lap_number"] % 4 == 0)
train, test = df[~test_mask], df[test_mask]
print(f"\ntrain: {len(train)} rows, "
      f"{train.groupby(['driver_code','lap_number']).ngroups} laps")
print(f"test:  {len(test)} rows, "
      f"{test.groupby(['driver_code','lap_number']).ngroups} laps "
      f"(lap numbers: {sorted(test.lap_number.unique())})")

# --- fit one model ---
# .fit() searches for the coefficients that make (weighted sum of the
# features) match the target as closely as possible across every training
# row, by minimizing squared error. For plain linear regression this has
# an exact closed-form answer - no randomness, no iterating, nothing to
# tune. It runs once and it's done.
model = LinearRegression()
model.fit(train[FEATURES], train[TARGETS])
pred = model.predict(test[FEATURES])

# --- score the model and the baseline the same way ---
# displacement error = straight-line distance (metres) between the
# predicted future position and the real one. Averaging this across the
# test set is a simplified version of what the trajectory-prediction
# literature calls ADE (average displacement error) - simplified because
# we're checking one horizon per row, not a whole predicted future path.
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
