"""
Shared helpers for talking to OpenF1.

Everything here takes a session_key and driver numbers as arguments rather
than hardcoding them, so the same functions serve both the command line
scripts and the Streamlit app.

The telemetry endpoints return thousands of rows and can be slow, so I cache
every response to a file on disk. Second run of any query is instant, and I'm
not hammering a free API while I fiddle with plot colors.
"""

import datetime
import hashlib
import json
import os
import time

import requests

BASE = "https://api.openf1.org/v1"
CACHE_DIR = "cache"

# Defaults used by the command line scripts: Belgian GP 2026, Norris vs
# Verstappen. The Streamlit app overrides these from its dropdowns.
DEFAULT_SESSION = 11334
DEFAULT_DRIVERS = (1, 3)


def get(endpoint, **params):
    """GET an OpenF1 endpoint, with a dumb file cache keyed on the request."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    key = endpoint + json.dumps(params, sort_keys=True)
    fname = os.path.join(
        CACHE_DIR,
        f"{endpoint}_{hashlib.md5(key.encode()).hexdigest()[:10]}.json",
    )

    if os.path.exists(fname):
        with open(fname) as f:
            return json.load(f)

    # OpenF1 is free and rate-limits fairly aggressively. Retry with
    # backoff on a 429 rather than letting the whole page crash - this
    # matters in the UI, where switching drivers can fire several requests
    # at once.
    for attempt in range(5):
        r = requests.get(f"{BASE}/{endpoint}", params=params)
        if r.status_code == 429:
            time.sleep(2 ** attempt)   # 1s, 2s, 4s, 8s, 16s
            continue
        break

    # OpenF1 returns 404 with {"detail": "No results found."} when a query
    # matches nothing, rather than an empty list. That's a normal outcome for
    # me (e.g. a lap window where telemetry is missing), not an error, so I
    # turn it into an empty list instead of letting it raise.
    if r.status_code == 404:
        data = []
    else:
        r.raise_for_status()
        data = r.json()

    with open(fname, "w") as f:
        json.dump(data, f)
    return data


# --- lookups -------------------------------------------------------------

def list_races(year):
    """
    Every race in a season, newest first.

    Filtering on session_name rather than session_type matters: sprint races
    are also session_type="Race", so filtering by type mixes them in with
    grands prix.
    """
    races = get("sessions", year=year, session_name="Race")

    # The schedule includes races that haven't happened yet. Those return
    # empty telemetry, so drop anything in the future or I'd be offering
    # the user broken options in a dropdown.
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    races = [r for r in races if r["date_start"] < now]

    return sorted(races, key=lambda s: s["date_start"], reverse=True)


def race_label(session):
    """Human readable name for a race, for dropdowns."""
    return (f"{session['date_start'][:10]}  -  {session['country_name']} "
            f"({session['circuit_short_name']})")


def list_drivers(session_key):
    """All drivers in a session, sorted by number."""
    drivers = get("drivers", session_key=session_key)
    # Some sessions list a driver more than once; keep the first of each.
    seen, out = set(), []
    for d in sorted(drivers, key=lambda x: x["driver_number"]):
        if d["driver_number"] not in seen:
            seen.add(d["driver_number"])
            out.append(d)
    return out


def driver_info(session_key, driver_number):
    """One driver's entry, or None if they weren't in this session."""
    for d in list_drivers(session_key):
        if d["driver_number"] == driver_number:
            return d
    return None


def driver_code(session_key, driver_number):
    d = driver_info(session_key, driver_number)
    return d["name_acronym"] if d else str(driver_number)


def team_color(session_key, driver_number):
    """
    Team color as a matplotlib-friendly hex string.

    OpenF1 gives team_colour as bare hex like "F47600" with no leading '#',
    so I add one. Falls back to grey if the field is missing.
    """
    d = driver_info(session_key, driver_number)
    if d and d.get("team_colour"):
        return "#" + d["team_colour"].lstrip("#")
    return "#888888"


def driver_colors(session_key, drivers):
    """
    Colors for a set of drivers, nudged apart if they clash.

    Two team mates share a team color, which would make them indistinguishable
    on a plot. If that happens I darken the second one.
    """
    colors = {}
    for num in drivers:
        c = team_color(session_key, num)
        if c in colors.values():
            # Same team: darken so the two lines are still tellable apart.
            r, g, b = (int(c[i:i + 2], 16) for i in (1, 3, 5))
            c = "#%02x%02x%02x" % (int(r * 0.55), int(g * 0.55), int(b * 0.55))
        colors[num] = c
    return colors


def stint_at_lap(session_key, driver_number, lap_number):
    """
    Which tire was this driver on during a given lap?

    Returns (compound, tyre_age) or (None, None) if the lap isn't in any
    stint. Handy for labelling plots so a lap number means something.
    """
    stints = get("stints", session_key=session_key, driver_number=driver_number)
    for s in stints:
        if s["lap_start"] <= lap_number <= s["lap_end"]:
            age = lap_number - s["lap_start"] + s["tyre_age_at_start"]
            return s["compound"], age
    return None, None
