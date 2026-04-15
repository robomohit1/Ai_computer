import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from pathlib import Path

import pytest


@pytest.fixture
def workspace(tmp_path):
    w = tmp_path / "workspace"
    w.mkdir()
    return w


@pytest.fixture(autouse=True)
def mock_keys(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", "testtoken")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-raw-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-raw-anthropic")
