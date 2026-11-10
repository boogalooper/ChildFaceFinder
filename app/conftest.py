"""Load the real engine while replacing only unavailable RAW/HEIF codecs."""
import importlib
import sys
import types
import pytest


@pytest.fixture
def engine_module(monkeypatch):
    heif = types.ModuleType('pillow_heif')
    heif.register_heif_opener = lambda: None
    monkeypatch.setitem(sys.modules, 'pillow_heif', heif)
    monkeypatch.setitem(sys.modules, 'rawpy', types.ModuleType('rawpy'))
    monkeypatch.delitem(sys.modules, 'image_loader', raising=False)
    monkeypatch.delitem(sys.modules, 'engine', raising=False)
    module = importlib.import_module('engine')
    yield module
