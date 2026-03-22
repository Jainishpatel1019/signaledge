"""tests/test_chunker.py — SlidingWindowChunker tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.chunker import SlidingWindowChunker


@pytest.fixture
def chunker():
    return SlidingWindowChunker(chunk_size=20, chunk_overlap=5)


class TestChunker:
    def test_empty_text_returns_empty(self, chunker):
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_short_text_returns_single_chunk(self, chunker):
        chunks = chunker.chunk("Hello world.")
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert "Hello world" in chunks[0]["text"]

    def test_chunks_have_required_keys(self, chunker):
        text = " ".join(["word"] * 100)
        chunks = chunker.chunk(text)
        for c in chunks:
            assert "text" in c
            assert "chunk_index" in c

    def test_chunk_indices_are_sequential(self, chunker):
        text = " ".join(["token"] * 100)
        chunks = chunker.chunk(text)
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap_creates_more_chunks(self):
        no_overlap = SlidingWindowChunker(chunk_size=20, chunk_overlap=0)
        with_overlap = SlidingWindowChunker(chunk_size=20, chunk_overlap=10)
        text = " ".join(["data"] * 100)
        assert len(with_overlap.chunk(text)) >= len(no_overlap.chunk(text))
