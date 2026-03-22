"""tests/test_pipeline.py — Pipeline processor tests (fully mocked)."""

from __future__ import annotations

import os
import sys
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def use_temp_db(tmp_path, monkeypatch):
    """Point DB_PATH to a temp file so tests don't touch real DB."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr("src.config.DB_PATH", db_path)
    monkeypatch.setattr("src.database.DB_PATH", db_path)
    return db_path


@pytest.fixture
def mock_externals(monkeypatch):
    """Mock all external API calls and heavy model loading."""
    # Mock EDGAR
    mock_edgar = MagicMock()
    mock_edgar.get_cik.return_value = "0000320193"
    mock_edgar.get_filings.return_value = [
        {"form_type": "10-K", "filed_date": "2024-01-15",
         "accession_number": "000-test", "url": "http://test/filing.htm", "cik": "0000320193"}
    ]
    mock_edgar.download_filing_text.return_value = "<html>Risk factors: revenue may decline.</html>"

    # Mock FMP
    mock_fmp = MagicMock()
    mock_fmp.get_company_profile.return_value = {"name": "Apple Inc.", "sector": "Technology"}
    mock_fmp.get_transcripts.return_value = [
        {"ticker": "AAPL", "quarter": 4, "year": 2023, "date": "2024-01-25",
         "content": "Revenue is growing strongly. We expect record results."}
    ]

    # Mock NLI
    mock_nli_cls = MagicMock()
    mock_nli_inst = MagicMock()
    mock_nli_inst.predict.return_value = {"label": "contradiction", "score": 0.85}
    mock_nli_cls.return_value = mock_nli_inst

    # Mock FinBERT
    mock_finbert_cls = MagicMock()
    mock_finbert_inst = MagicMock()
    mock_finbert_inst.sentiment_shift.return_value = -0.3
    mock_finbert_cls.return_value = mock_finbert_inst

    # Mock Groq
    mock_groq_cls = MagicMock()
    mock_groq_inst = MagicMock()
    mock_groq_inst.summarize.return_value = "Contradiction detected. Market impact likely."
    mock_groq_cls.return_value = mock_groq_inst

    # Mock signal generator
    mock_sig = MagicMock()
    mock_sig.compute_car.return_value = {
        "car_1d": -0.01, "car_3d": -0.03, "car_5d": -0.05,
        "price": 185.0, "price_change_pct": -2.5,
    }
    mock_sig.classify_signal.return_value = "BEARISH"
    mock_sig.generate_signal.return_value = {
        "id": "test_signal_001", "ticker": "AAPL", "signal_type": "BEARISH",
        "confidence": 0.85, "price": 185.0, "price_change_pct": -2.5,
        "description": "Test signal", "event_date": "2024-01-25",
        "car_1d": -0.01, "car_3d": -0.03, "car_5d": -0.05,
        "contradiction_id": "test_contra_001",
    }

    monkeypatch.setattr("src.pipeline.processor.EDGARClient", lambda: mock_edgar)
    monkeypatch.setattr("src.pipeline.processor.FMPClient", lambda: mock_fmp)
    monkeypatch.setattr("src.backtest.signals.SignalGenerator", lambda: mock_sig)

    # Patch model imports inside processor
    monkeypatch.setattr("src.models.nli.NLIModel", mock_nli_cls)
    monkeypatch.setattr("src.models.finbert.FinBERTSentiment", mock_finbert_cls)
    monkeypatch.setattr("src.models.groq_summarizer.GroqSummarizer", mock_groq_cls)

    return {
        "edgar": mock_edgar, "fmp": mock_fmp,
        "nli": mock_nli_inst, "finbert": mock_finbert_inst,
    }


class TestPipeline:
    def test_process_returns_summary(self, mock_externals, use_temp_db):
        from src.pipeline.processor import CompanyProcessor
        proc = CompanyProcessor()
        result = proc.process("AAPL")
        assert result["ticker"] == "AAPL"
        assert result["documents"] >= 1

    def test_process_populates_database(self, mock_externals, use_temp_db):
        from src.pipeline.processor import CompanyProcessor
        from src.database import get_db
        proc = CompanyProcessor()
        proc.process("AAPL")

        with get_db() as conn:
            companies = conn.execute("SELECT * FROM companies").fetchall()
            docs = conn.execute("SELECT * FROM documents").fetchall()
            assert len(companies) >= 1
            assert len(docs) >= 1

    def test_process_with_callback(self, mock_externals, use_temp_db):
        from src.pipeline.processor import CompanyProcessor
        steps_received = []

        def cb(step, msg):
            steps_received.append(step)

        proc = CompanyProcessor()
        proc.process("AAPL", progress_callback=cb)
        assert len(steps_received) == 10
        assert steps_received == list(range(1, 11))
