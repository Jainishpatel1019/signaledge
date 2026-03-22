"""tests/test_fmp_client.py — FMP client tests (mocked)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.fmp_client import FMPClient

MOCK_TRANSCRIPTS = [
    {"quarter": 4, "year": 2023, "date": "2024-01-25", "content": "Q4 was great."},
    {"quarter": 3, "year": 2023, "date": "2023-10-26", "content": "Q3 results."},
    {"quarter": 2, "year": 2023, "date": "2023-07-27", "content": "Q2 numbers."},
    {"quarter": 1, "year": 2023, "date": "2023-04-27", "content": "Q1 update."},
]

MOCK_PROFILE = [
    {
        "companyName": "Apple Inc.",
        "sector": "Technology",
        "description": "A tech company.",
    }
]


@pytest.fixture
def mock_fmp(monkeypatch):
    client = FMPClient()

    def fake_get(url, params=None, timeout=None):
        resp = MagicMock()
        if "earning_call_transcript" in url:
            resp.status_code = 200
            resp.json.return_value = MOCK_TRANSCRIPTS
        elif "profile" in url:
            resp.status_code = 200
            resp.json.return_value = MOCK_PROFILE
        else:
            resp.status_code = 404
            resp.json.return_value = {}
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr(client._session, "get", fake_get)
    return client


class TestFMPClient:
    def test_get_transcripts_returns_list(self, mock_fmp):
        results = mock_fmp.get_transcripts("AAPL", limit=4)
        assert len(results) == 4
        assert results[0]["ticker"] == "AAPL"
        assert results[0]["quarter"] == 4
        assert "content" in results[0]

    def test_get_company_profile(self, mock_fmp):
        profile = mock_fmp.get_company_profile("AAPL")
        assert profile is not None
        assert profile["name"] == "Apple Inc."
        assert profile["sector"] == "Technology"

    def test_handles_404_gracefully(self, monkeypatch):
        import requests as _req
        client = FMPClient()

        def fake_404(url, params=None, timeout=None):
            raise _req.exceptions.ConnectionError("Not found")

        monkeypatch.setattr(client._session, "get", fake_404)
        result = client.get_transcripts("INVALID")
        assert result == []
