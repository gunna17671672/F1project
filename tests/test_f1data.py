"""
Tests for the OpenF1 helpers in f1data.py.

None of these touch the network. requests.get gets swapped for a fake so I
can check the caching and error handling without burning real API calls.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import f1data


class FakeResponse:
    """Stands in for a requests.Response. Only has the bits f1data touches."""

    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else []

    def raise_for_status(self):
        # real requests raises on 4xx and 5xx. 404 is handled before this
        # ever gets called, so anything else that's an error should blow up
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clean_cache(tmp_path, monkeypatch):
    # each test runs in its own temp dir so a cache file left by one test
    # can't make the next one pass for the wrong reason
    monkeypatch.chdir(tmp_path)


def test_404_gives_empty_list(monkeypatch):
    monkeypatch.setattr(f1data.requests, "get", lambda *a, **k: FakeResponse(404))
    assert f1data.get("laps", session_key=1) == []


def test_second_call_comes_from_cache(monkeypatch):
    calls = []

    def fake_get(*a, **k):
        calls.append(1)
        return FakeResponse(200, [{"lap_number": 1}])

    monkeypatch.setattr(f1data.requests, "get", fake_get)

    first = f1data.get("laps", session_key=1)
    second = f1data.get("laps", session_key=1)

    assert first == second
    assert len(calls) == 1


def test_cache_file_gets_written(monkeypatch):
    monkeypatch.setattr(
        f1data.requests, "get", lambda *a, **k: FakeResponse(200, [{"x": 1}])
    )
    f1data.get("position", session_key=5)

    files = os.listdir(f1data.CACHE_DIR)
    assert len(files) == 1
    assert files[0].startswith("position_")


def test_retries_when_rate_limited(monkeypatch):
    attempts = []

    def fake_get(*a, **k):
        attempts.append(1)
        if len(attempts) < 3:
            return FakeResponse(429)
        return FakeResponse(200, [{"ok": True}])

    monkeypatch.setattr(f1data.requests, "get", fake_get)
    monkeypatch.setattr(f1data.time, "sleep", lambda s: None)

    out = f1data.get("car_data", session_key=2)

    assert out == [{"ok": True}]
    assert len(attempts) == 3


def test_different_params_dont_share_a_cache_file(monkeypatch):
    monkeypatch.setattr(
        f1data.requests, "get", lambda *a, **k: FakeResponse(200, [{"x": 1}])
    )
    f1data.get("laps", driver_number=1)
    f1data.get("laps", driver_number=3)

    assert len(os.listdir(f1data.CACHE_DIR)) == 2
