"""tests/test_edgar_client.py — EDGAR client tests (mocked)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.edgar_client import EDGARClient


# ── Fixtures ──────────────────────────────────────────────────────────────

MOCK_TICKERS_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
}

MOCK_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "8-K", "4"],
            "accessionNumber": [
                "0000320193-23-000106",
                "0000320193-23-000077",
                "0000320193-23-000070",
                "0000320193-23-000065",
            ],
            "filingDate": ["2023-11-03", "2023-08-04", "2023-05-05", "2023-04-01"],
            "primaryDocument": ["aapl-20231.htm", "aapl-20232.htm", "aapl-20233.htm", "aapl-4.xml"],
        }
    }
}


@pytest.fixture
def mock_edgar(monkeypatch):
    client = EDGARClient()

    def fake_get(url, *a, **kw):
        resp = MagicMock()
        resp.status_code = 200
        if "company_tickers" in url:
            resp.json.return_value = MOCK_TICKERS_JSON
        else:
            resp.json.return_value = MOCK_SUBMISSIONS
        resp.text = "<html>Filing content</html>"
        return resp

    monkeypatch.setattr(client._session, "get", fake_get)
    # bypass rate limit in tests
    monkeypatch.setattr(client, "_rate_limit", lambda: None)
    return client


# ── Tests ─────────────────────────────────────────────────────────────────

class TestEDGARClient:
    def test_get_cik_returns_padded_string(self, mock_edgar):
        cik = mock_edgar.get_cik("AAPL")
        assert cik == "0000320193"
        assert len(cik) == 10

    def test_get_filings_returns_filtered_list(self, mock_edgar):
        filings = mock_edgar.get_filings("AAPL", form_types=["10-K", "10-Q"])
        assert len(filings) == 2
        forms = {f["form_type"] for f in filings}
        assert forms == {"10-K", "10-Q"}
        for f in filings:
            assert "accession_number" in f
            assert "filed_date" in f

    def test_get_filings_respects_limit(self, mock_edgar):
        filings = mock_edgar.get_filings("AAPL", form_types=["10-K", "10-Q", "8-K"], limit=2)
        assert len(filings) <= 2
