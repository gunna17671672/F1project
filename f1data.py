"""
Shared helpers for talking to OpenF1.

The telemetry endpoints return thousands of rows and can be slow, so I cache
every response to a file on disk. Second run of any script is instant, and I'm
not hammering a free API while I fiddle with plot colors.
"""

import hashlib
import json
import os

import requests

BASE = "https://api.openf1.org/v1"
CACHE_DIR = "cache"


def get(endpoint, **params):
    """GET an OpenF1 endpoint, with a dumb file cache keyed on the request."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Build a stable filename from endpoint + params.
    key = endpoint + json.dumps(params, sort_keys=True)
    fname = os.path.join(
        CACHE_DIR,
        f"{endpoint}_{hashlib.md5(key.encode()).hexdigest()[:10]}.json",
    )

    if os.path.exists(fname):
        with open(fname) as f:
            return json.load(f)

    r = requests.get(f"{BASE}/{endpoint}", params=params)
    r.raise_for_status()
    data = r.json()

    with open(fname, "w") as f:
        json.dump(data, f)
    return data


# The session I picked: Belgian GP race, 19 July 2026.
SESSION_KEY = 11334

# Two drivers to compare. Different teams, and they ran opposite tire
# strategies in this race, which makes the stint comparison interesting.
DRIVERS = {
    1: "NOR",   # Norris, McLaren
    3: "VER",   # Verstappen, Red Bull
}

# Team colors, so plots are readable at a glance.
COLORS = {
    1: "#FF8000",   # McLaren papaya
    3: "#3671C6",   # Red Bull blue
}


def stint_at_lap(driver_number, lap_number):
    """
    Which tire was this driver on during a given lap?

    Returns (compound, tyre_age) or (None, None) if the lap isn't in any
    stint. Handy for labelling plots so a lap number means something.
    """
    stints = get("stints", session_key=SESSION_KEY, driver_number=driver_number)
    for s in stints:
        if s["lap_start"] <= lap_number <= s["lap_end"]:
            age = lap_number - s["lap_start"] + s["tyre_age_at_start"]
            return s["compound"], age
    return None, None
