"""tests/test_nli.py — NLI model tests (mocked)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VALID_LABELS = {"contradiction", "entailment", "neutral"}
_LABEL_LIST = ["contradiction", "entailment", "neutral"]


@pytest.fixture(autouse=True)
def mock_nli_backend(monkeypatch):
    import src.models.nli as nli_mod
    nli_mod._NLIBackend._instance = None

    def _mock_load(self):
        self.tokenizer = MagicMock()

        def _enc(a, b, **kw):
            bs = len(a) if isinstance(a, list) else 1
            return {"input_ids": torch.zeros(bs, 8, dtype=torch.long),
                    "attention_mask": torch.ones(bs, 8, dtype=torch.long)}

        self.tokenizer.side_effect = _enc
        self.tokenizer.__call__ = _enc

        self.model = MagicMock()
        self.model.eval.return_value = self.model
        self.model.config.id2label = {0: "contradiction", 1: "entailment", 2: "neutral"}

        def _fwd(**kw):
            bs = kw["input_ids"].shape[0]
            logits = torch.zeros(bs, 3)
            logits[:, 0] = 10.0
            out = MagicMock()
            out.logits = logits
            return out

        self.model.side_effect = _fwd
        self.model.__call__ = _fwd
        self.labels = _LABEL_LIST

    monkeypatch.setattr("src.models.nli._NLIBackend._load", _mock_load)
    yield
    nli_mod._NLIBackend._instance = None


class TestNLIModel:
    def test_predict_returns_dict_with_keys(self):
        from src.models.nli import NLIModel
        r = NLIModel().predict("A", "B")
        assert "label" in r and "score" in r

    def test_score_between_0_and_1(self):
        from src.models.nli import NLIModel
        r = NLIModel().predict("A", "B")
        assert 0.0 <= r["score"] <= 1.0

    def test_label_is_valid(self):
        from src.models.nli import NLIModel
        r = NLIModel().predict("A", "B")
        assert r["label"] in VALID_LABELS

    def test_batch_returns_list(self):
        from src.models.nli import NLIModel
        results = NLIModel().predict_batch([("A", "B"), ("C", "D")])
        assert len(results) == 2
