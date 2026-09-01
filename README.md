# F1 Telemetry Analysis

Compare any two drivers in any race using free telemetry from the
[OpenF1 API](https://openf1.org). Draws the racing line from GPS coordinates
colored by speed, compares lap times, and measures tire degradation.

Comes with a Streamlit UI for picking the race and drivers, or you can run the
scripts directly.

The worked example throughout is the **Belgian GP 2026** at Spa-Francorchamps
(`session_key=11334`), Norris vs Verstappen.

## What's here

| Script | What it does |
| --- | --- |
| `app.py` | Streamlit UI - pick a season, race and two drivers, see all three plots |
| `explore_sessions.py` | Lists races for a season and drivers in a session, to find a `session_key` |
| `f1data.py` | Shared API helper with an on-disk cache, plus session/driver config |
| `lap_times.py` | Lap time comparison over the race |
| `racing_line.py` | Track map drawn from GPS coordinates, colored by speed |
| `tire_deg.py` | Lap time vs tire age per stint, with a fuel-burn correction |
| `speed_delta.py` | Where on track one driver gains or loses time on the other |

Plots are written to `plots/`.

## Running it

Set up once:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The UI, which is the easiest way to explore:

```bash
streamlit run app.py
```

Or the individual scripts, which use the defaults in `f1data.py`:

```bash
python lap_times.py
python racing_line.py            # each driver's fastest lap
python racing_line.py --lap 25   # both drivers on the same lap
python tire_deg.py
```

Responses are cached in `cache/`, so only the first run hits the network.
Delete that folder to force fresh API calls.

## The OpenF1 endpoints I used

OpenF1 is free and needs no authentication. Data is organised as
`meeting` (a race weekend) -> `session` (race, qualifying, practice). Almost
every endpoint is filtered by `session_key`.

| Endpoint | Rate | Columns I used |
| --- | --- | --- |
| `sessions` | - | `session_key`, `circuit_short_name`, `date_start` |
| `drivers` | - | `driver_number`, `name_acronym`, `team_name` |
| `laps` | 1/lap | `lap_number`, `lap_duration`, `date_start`, `is_pit_out_lap` |
| `location` | ~4 Hz | `x`, `y`, `z`, `date` |
| `car_data` | ~4 Hz | `speed`, `throttle`, `brake`, `n_gear`, `rpm`, `drs`, `date` |
| `stints` | 1/stint | `compound`, `lap_start`, `lap_end`, `tyre_age_at_start` |

Four things that cost me time and are worth knowing:

**Sprints are `session_type="Race"` too.** Filtering sessions by type gives you
sprint races mixed in with grands prix. Filter on `session_name` instead.

**A query that matches nothing returns HTTP 404**, with body
`{"detail": "No results found."}`, rather than an empty list. My first version
called `raise_for_status()` and crashed on what is really a normal outcome.
`f1data.get()` now turns a 404 into an empty list.

**Timestamp formats are inconsistent between sessions**, and sometimes within
one response. Spa returns `2026-07-19T14:27:10.083000+00:00` (microseconds),
Monaco returns `2026-06-07T13:02:11+00:00` (none). `pd.to_datetime` infers the
format from the first row and then throws on any row that differs, so every
call passes `format="ISO8601"`.

**`location` x/y/z are in decimetres, not metres.** OpenF1 doesn't document the
units, so I measured them: I summed the distance between consecutive points
around one lap and compared to Spa's official 7004 m. That gave 9.89 units per
metre, so the unit is 1/10 m. It comes in just under 10 because straight lines
between 4 Hz samples cut the corners slightly.

## Speed delta: where on track is a driver actually faster?

The racing line and lap-time plots compare drivers in aggregate. This asks a
more specific question: at *this* point on the track, who's faster?

GPS samples from two different cars never land at the same spot on track, so
there's no shared index to compare speed at "the same point" directly - the
same kind of problem as merging `location` and `car_data`, one level up. The
fix: turn each driver's path into "fraction of distance completed" (via
cumulative distance along their own GPS trace) and interpolate both drivers'
speed onto one common 400-point grid of that fraction. Once both are sampled
at the same 400 positions around the lap, they're directly comparable, and
the difference at each point can be drawn as a track map (colored by who's
faster in each chunk) or a delta trace.

One artifact worth knowing: sharp narrow spikes in the delta, usually right
at a braking zone, are typically a few metres of difference in *where* each
driver started braking, not a sustained speed advantage. A car still at full
speed 5m before its brake point reads as "faster" than one already on the
brakes, even if the braking itself is identical.

## The interesting technical problem: joining two telemetry streams

`location` (where the car is) and `car_data` (what the car is doing) are
separate endpoints, sampled independently. Their timestamps never match:

```
location: 14:27:10.083   x=-497, y=1401
car_data: 14:27:10.057   speed=229
```

There is no shared key - no lap number, no sample index - so a normal join on
`date` returns almost nothing. The fix is `pandas.merge_asof`, a nearest-key
join:

```python
merged = pd.merge_asof(
    loc, car[["date", "speed", "throttle", "brake", "n_gear"]],
    on="date",
    direction="nearest",
    tolerance=pd.Timedelta("0.5s"),
)
```

`direction="nearest"` takes the closest sample either side; the default
(`"backward"`) always takes the previous one, which biases every speed reading
slightly stale. `tolerance` matters: without it, a dropout in `car_data` would
silently match a corner to a straight-line speed reading seconds away. With it
those rows come back `NaN` and get dropped. In this session nothing was
dropped, but the guard is what makes the result trustworthy.

The gradient itself needs a `LineCollection` - `plot()` draws one color per
call, so each segment has to be its own colored object.

## Findings

### 1. The race was decided by strategy inversion, not raw pace

The two ran opposite strategies:

| | Stint 1 | Stint 2 |
| --- | --- | --- |
| NOR | HARD, laps 1-30 | MEDIUM, laps 31-44 |
| VER | MEDIUM, laps 1-17 | HARD, laps 18-44 |

Whoever was on the fresher or softer tire at a given moment was faster.
Verstappen led on pace from lap 18 to 30 by roughly a second a lap. Norris
pitted on lap 30 onto MEDIUMs and from lap 32 ran 109.1s against Verstappen's
110.2s, holding it to the flag. Taking the softer tire last, on the lightest
fuel load, was worth more than taking it first.

Norris's fastest lap of the race was his final one: 108.890s.

### 2. Raw lap times say tires get faster with age. They don't.

Fitting a line to lap time vs tire age gave negative slopes - tires apparently
improving as they wore out. The cause is fuel burn. A car starts ~100 kg
heavier than it finishes and burns 1.8-2.2 kg per lap at Spa; at roughly 0.3 s
per 10 kg that's about **0.06 s/lap gained just from getting lighter**. That
runs opposite to tire wear and, over a long stint, is bigger than it.

So raw lap times measure `(tire wear - fuel burn)`. Adding the fuel effect back
in makes every slope physically sensible:

| Driver | Compound | Laps | Raw slope | Fuel-corrected |
| --- | --- | --- | --- | --- |
| NOR | HARD | 23 | -0.012 | **+0.048** |
| NOR | MEDIUM | 13 | -0.040 | **+0.020** |
| VER | MEDIUM | 13 | +0.171 | **+0.231** |
| VER | HARD | 24 | +0.005 | **+0.065** |

The ordering matches intuition: MEDIUM on a heavy car degrades hardest
(+0.231 s/lap), and Norris's short late MEDIUM stint on a light car degrades
least (+0.020 s/lap).

### 3. Comparing fastest laps is misleading; comparing the same lap isn't

`racing_line.py` with no arguments compares each driver's personal best. But
Norris's came on lap 44 (fresh MEDIUM, near-empty tank) and Verstappen's on
lap 23 (HARD, half tank). Much of that 0.7s gap is fuel and compound, not
driving.

`--lap 25` fixes this. On lap 25 both were on HARD tires with the same fuel
load and the same track state - the only large difference left is tire age:
Norris's set was 24 laps old, Verstappen's 7. Verstappen was 1.1s faster
(109.801s vs 110.903s). That is close to a clean read on what 17 laps of tire
wear costs at Spa.

## What broke when I tested beyond the one race I built this against

Running the same code against other races surfaced real bugs, not just
missing data:

- **OpenF1 returns HTTP 429 (rate limited) under repeated requests.** It's a
  free API with no key, so this is expected under load - a first page load
  with an uncached driver pair can trip it. `f1data.get()` retries with
  exponential backoff (up to ~30s total) before giving up.
- **A driver can be listed in a session's driver list with almost no lap
  data**, for two different reasons that need different messages: the whole
  session has no data at all yet (Jeddah and Sakhir 2026, as of testing -
  OpenF1 hadn't backfilled them), versus one driver individually retired
  early (Verstappen, Zandvoort 2026, one lap row with a null duration). The
  UI checks the session first and tells you to pick a different race, then
  checks individual drivers and tells you to pick a different driver.

## Data coverage is uneven, and the code has to expect that

Testing the app on races other than Spa turned up two failure modes that
aren't bugs in my code, they're just what the data looks like:

- **Two of the 14 completed 2026 races (Jeddah, Sakhir) have no `location` or
  `car_data` at all.** Lap times and stints still work there; the racing line
  can't.
- **Monaco has a ~50 minute hole in the middle of the race.** Sampling is
  normal (about 4 Hz) either side of it, so a lap early in the race plots
  fine while a lap in the middle returns nothing.
- **A driver can be in the driver list with no usable laps.** Verstappen at
  Zandvoort 2026 has a single lap row with a null duration, because he
  retired on lap 1. The UI checks for this and says so rather than rendering
  an empty plot.

## Limitations

- **The fuel correction is an estimate, not a measurement.** Real fuel loads
  aren't public and OpenF1 doesn't publish them. `FUEL_EFFECT_S_PER_LAP = 0.06`
  in `tire_deg.py` is a rule of thumb; every corrected number moves with it.
- **No traffic data.** OpenF1 doesn't say whether a driver was in clean air or
  stuck behind someone. Some lap-to-lap variation is dirty air, not tire state.
  This is a pace comparison, not a clean-air comparison.
- **Linear degradation is a simplification.** Real tire wear tends to be flat
  and then fall off a cliff. Over a 13-27 lap stint a straight line is a
  reasonable summary, but it will understate the end of a long stint.
- **The clean-lap filter is a judgment call.** I exclude out-laps and any lap
  more than 7% slower than the driver's own median. That catches safety car and
  traffic laps, but 7% is tunable and the results shift a little with it. I use
  a median-relative threshold rather than a hardcoded time so the code still
  works at circuits with very different lap lengths.
- **One race, two drivers.** Nothing here generalises to other tracks without
  rerunning it.

## Possible next steps

- Speed delta around the lap: interpolate both drivers onto a common distance
  axis and plot where time is actually gained and lost, corner by corner.
- Use `throttle` / `brake` / `n_gear` (already merged in, currently unused) to
  compare braking points.
- Segment the track into corners and compare minimum speeds.
