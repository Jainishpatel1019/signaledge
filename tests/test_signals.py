"""tests/test_signals.py — Signal classification tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.signals import SignalGenerator


class TestSignalClassification:
    def test_bearish_signal(self):
        s = SignalGenerator.classify_signal(contradiction_score=0.85, sentiment_shift=-0.4)
        assert s == "BEARISH"

    def test_bullish_signal(self):
        s = SignalGenerator.classify_signal(contradiction_score=0.80, sentiment_shift=0.3)
        assert s == "BULLISH"

    def test_watch_signal_high_score(self):
        s = SignalGenerator.classify_signal(contradiction_score=0.75, sentiment_shift=0.0)
        assert s == "WATCH"

    def test_neutral_signal(self):
        s = SignalGenerator.classify_signal(contradiction_score=0.3, sentiment_shift=0.0)
        assert s == "NEUTRAL"


class TestGenerateSignal:
    @patch.object(SignalGenerator, "compute_car", return_value={
        "car_1d": -0.02, "car_3d": -0.05, "car_5d": -0.08,
        "price": 150.0, "price_change_pct": -3.2,
    })
    def test_generate_signal_returns_dict(self, mock_car):
        gen = SignalGenerator()
        sig = gen.generate_signal("AAPL", 0.85, -0.4, "2024-01-25")
        assert sig["signal_type"] == "BEARISH"
        assert sig["ticker"] == "AAPL"
        assert "id" in sig
        assert "car_5d" in sig
