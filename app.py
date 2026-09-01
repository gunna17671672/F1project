"""
Streamlit front end for the F1 telemetry analysis.

Everything here is a thin wrapper: the plots come from the same build_figure()
functions the command line scripts use, so there's no duplicated analysis
code. This file only handles picking a race, picking drivers, and laying the
results out.

Run it with:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import f1data
import lap_times
import racing_line
import tire_deg

st.set_page_config(page_title="F1 Telemetry Explorer",
                   page_icon="🏎️", layout="wide")

# A little CSS to tighten up Streamlit's default spacing and give the header
# an F1-ish look. Nothing functional depends on this.
st.markdown("""
<style>
  .block-container { padding-top: 2.2rem; max-width: 1500px; }
  h1 { font-weight: 800; letter-spacing: -0.5px; }
  .accent { color: #E10600; }
  .subtle { color: #8B92A5; font-size: 0.9rem; }
  div[data-testid="stMetricValue"] { font-size: 1.5rem; }
  .stTabs [data-baseweb="tab"] { font-size: 1rem; padding: 0.4rem 1.1rem; }
</style>
""", unsafe_allow_html=True)


# --- cached data loaders -------------------------------------------------
# st.cache_data stops Streamlit re-fetching every time a widget changes.
# f1data already caches to disk; this caches in memory on top of that.

@st.cache_data(show_spinner=False)
def races(year):
    return f1data.list_races(year)


@st.cache_data(show_spinner=False)
def drivers(session_key):
    return f1data.list_drivers(session_key)


@st.cache_data(show_spinner=False)
def max_lap(session_key, driver_number):
    """
    Highest lap number this driver has usable timing for.

    Returns 0 when there's nothing. A driver can appear in the driver list
    but have almost no data - Verstappen at Zandvoort 2026 has a single lap
    row with a null duration because he retired on lap 1.
    """
    df = lap_times.load_laps(driver_number, session_key)
    return int(df["lap_number"].max()) if not df.empty else 0


# --- sidebar: what to analyse -------------------------------------------

st.sidebar.title("Session")

year = st.sidebar.selectbox("Season", [2026, 2025, 2024], index=0)

race_list = races(year)
if not race_list:
    st.error(f"No completed races found for {year}.")
    st.stop()

race = st.sidebar.selectbox(
    "Race", race_list,
    format_func=f1data.race_label,
    index=next((i for i, r in enumerate(race_list)
                if r["session_key"] == f1data.DEFAULT_SESSION), 0),
)
session_key = race["session_key"]

driver_list = drivers(session_key)
if len(driver_list) < 2:
    st.error("Not enough drivers in this session.")
    st.stop()

labels = {d["driver_number"]: f"{d['name_acronym']} - {d['team_name']}"
          for d in driver_list}
numbers = [d["driver_number"] for d in driver_list]

st.sidebar.markdown("---")
st.sidebar.title("Drivers")

def default_index(preferred, fallback):
    return numbers.index(preferred) if preferred in numbers else fallback

d1 = st.sidebar.selectbox("Driver A", numbers,
                          format_func=lambda n: labels[n],
                          index=default_index(1, 0))
d2 = st.sidebar.selectbox("Driver B", numbers,
                          format_func=lambda n: labels[n],
                          index=default_index(3, 1))

if d1 == d2:
    st.sidebar.warning("Pick two different drivers.")
    st.stop()

selected = (d1, d2)

# A driver can be entered in a race but have no usable laps - a lap 1 retirement
# still shows up in the drivers endpoint. Catch that here rather than letting
# every plot below fail in its own way.
too_short = [f1data.driver_code(session_key, n) for n in selected
             if max_lap(session_key, n) < 5]
if too_short:
    st.warning(
        f"**{', '.join(too_short)}** has almost no lap data in this race - "
        f"most likely an early retirement. Pick another driver.", icon="🚧")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.title("Options")

same_lap = st.sidebar.toggle(
    "Compare the same lap", value=True,
    help="On: both drivers on the same lap number, so fuel load and track "
         "state are equal. Off: each driver's own fastest lap, which is "
         "faster but not a fair comparison.",
)

lap_number = None
if same_lap:
    # Only offer laps both drivers actually completed.
    upper = min(max_lap(session_key, d1), max_lap(session_key, d2))
    lap_number = st.sidebar.slider("Lap", 1, upper,
                                   value=min(25, upper))

fuel_effect = st.sidebar.slider(
    "Fuel effect (s/lap)", 0.0, 0.12, tire_deg.FUEL_EFFECT_S_PER_LAP, 0.01,
    help="Seconds per lap the car gains from burning fuel. Used to correct "
         "the degradation slopes. This is an estimate, not measured data - "
         "slide it to zero to see the raw (physically impossible) result.",
)


# --- header --------------------------------------------------------------

code1 = f1data.driver_code(session_key, d1)
code2 = f1data.driver_code(session_key, d2)

st.markdown(f"# <span class='accent'>{code1}</span> vs "
            f"<span class='accent'>{code2}</span>", unsafe_allow_html=True)
st.markdown(
    f"<p class='subtle'>{race['country_name']} Grand Prix &nbsp;·&nbsp; "
    f"{race['circuit_short_name']} &nbsp;·&nbsp; {race['date_start'][:10]} "
    f"&nbsp;·&nbsp; session_key {session_key}</p>",
    unsafe_allow_html=True,
)


def dark(fn, *args, **kwargs):
    """Build a matplotlib figure using a dark style, to match the UI."""
    with plt.style.context("dark_background"):
        result = fn(*args, **kwargs)
    return result


tab1, tab2, tab3, tab4 = st.tabs(
    ["Racing line", "Lap times", "Tire degradation", "About the data"])


# --- racing line ---------------------------------------------------------

with tab1:
    try:
        with st.spinner("Fetching GPS and speed telemetry..."):
            fig, info = dark(racing_line.build_figure,
                             session_key, selected, lap_number)
        cols = st.columns(len(info))
        for col, i in zip(cols, info):
            with col:
                st.metric(f"{i['driver']} - lap {i['lap']}",
                          f"{i['lap_time']}s",
                          help=f"{i['compound']}, {i['tyre_age']} laps old")
                st.caption(f"{i['samples']} merged samples · "
                           f"{i['min_speed']}-{i['max_speed']} km/h")

        if len(info) == 2:
            gap = info[0]["lap_time"] - info[1]["lap_time"]
            faster = info[1]["driver"] if gap > 0 else info[0]["driver"]
            st.caption(f"**{faster} quicker by {abs(gap):.3f}s** on this lap.")

        st.pyplot(fig, width="stretch")
        plt.close(fig)

        if not same_lap:
            st.warning(
                "These are each driver's personal best laps, which came at "
                "different points in the race on different fuel loads and "
                "tires. Turn on **Compare the same lap** for a fair "
                "comparison.", icon="⚠️")
    except Exception as e:
        st.error(f"Couldn't build the racing line: {e}")


# --- lap times -----------------------------------------------------------

with tab2:
    try:
        with st.spinner("Loading lap times..."):
            fig, summary = dark(lap_times.build_figure, session_key, selected)
        cols = st.columns(len(summary))
        for col, s in zip(cols, summary):
            with col:
                st.metric(f"{s['driver']} best lap", f"{s['best']}s")
                st.caption(f"{s['clean_laps']} clean laps · "
                           f"median {s['median']}s")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption("Hollow markers are safety car and pit out-laps, excluded "
                   "from the analysis. Dotted vertical lines are pit stops.")
    except Exception as e:
        st.error(f"Couldn't build the lap time plot: {e}")


# --- tire degradation ----------------------------------------------------

with tab3:
    try:
        with st.spinner("Loading stints..."):
            fig, rows = dark(tire_deg.build_figure,
                             session_key, selected, fuel_effect)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        table = pd.DataFrame(rows).rename(columns={
            "raw": "raw slope (s/lap)",
            "corrected": "fuel-corrected (s/lap)",
        })
        st.dataframe(table, width="stretch", hide_index=True)

        st.info(
            "Raw lap times often show **negative** degradation - tires "
            "apparently getting faster. They aren't: the car is burning fuel "
            "and getting lighter, and that gain is bigger than the tire loss. "
            "The fuel-corrected column adds that effect back in. Drag the "
            "fuel slider to zero in the sidebar to see the uncorrected "
            "result.", icon="💡")
    except Exception as e:
        st.error(f"Couldn't build the degradation plot: {e}")


# --- notes ---------------------------------------------------------------

with tab4:
    st.markdown("""
### Where this data comes from

All of it is from the [OpenF1 API](https://openf1.org) - free, no
authentication. Data is organised as `meeting` (a race weekend) →
`session` (race, qualifying, practice).

| Endpoint | Rate | What I use it for |
| --- | --- | --- |
| `sessions` | - | Finding a race |
| `drivers` | - | Names, teams, team colours |
| `laps` | 1/lap | Lap times, pit out-lap flags |
| `location` | ~4 Hz | `x`, `y`, `z` position - the track map |
| `car_data` | ~4 Hz | Speed, throttle, brake, gear |
| `stints` | 1/stint | Tire compound and age |

### The bit that was actually hard

`location` and `car_data` are separate endpoints sampled independently, and
their timestamps never match:

```
location: 14:27:10.083   x=-497, y=1401
car_data: 14:27:10.057   speed=229
```

There is no shared key to join on, so a normal join returns nothing.
`pandas.merge_asof` does a nearest-timestamp join instead, with a 0.5s
tolerance so a dropout in one stream can't silently match a corner to a
straight-line speed reading.

### Limitations

- **The fuel correction is an estimate**, not a measurement. Real fuel loads
  aren't public. Every corrected number moves with that slider.
- **No traffic data.** OpenF1 doesn't say whether a driver was in clean air.
  Some variation is dirty air, not tire state.
- **Linear degradation is a simplification.** Real wear tends to be flat and
  then fall off a cliff.
- **The clean-lap filter is a judgment call**: out-laps, plus anything more
  than 7% slower than that driver's own median.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Data: OpenF1 · Built with Streamlit")
