"""
Step 1: find a session_key to analyze.

OpenF1 organizes everything into a hierarchy:
  meeting  = a whole race weekend (meeting_key)
  session  = one part of that weekend (session_key)
             -> Practice 1/2/3, Qualifying, Sprint, Race
Almost every other endpoint wants a session_key, so this is where we start.
No API key needed, just plain GET requests.
"""

import requests

BASE = "https://api.openf1.org/v1"


def get(endpoint, **params):
    """Small helper so I don't repeat the URL everywhere."""
    r = requests.get(f"{BASE}/{endpoint}", params=params)
    r.raise_for_status()
    return r.json()


def list_races(year):
    # session_name="Race" filters out practice/quali. Note: sprints are also
    # session_type="Race", which is why I filter on session_name instead.
    sessions = get("sessions", year=year, session_name="Race")
    for s in sessions:
        print(f"{s['session_key']}  {s['date_start'][:10]}  "
              f"{s['circuit_short_name']:<20} {s['country_name']}")
    return sessions


def show_drivers(session_key):
    drivers = get("drivers", session_key=session_key)
    for d in drivers:
        print(f"  #{d['driver_number']:<3} {d['name_acronym']}  {d['team_name']}")
    return drivers


if __name__ == "__main__":
    print("=== 2026 races ===")
    list_races(2026)

    print("\n=== drivers at Spa (session 11334) ===")
    show_drivers(11334)
