"""pytest plugin: `world` fixture (fresh mock World per test)."""

import pytest

from trajtest.world import World


@pytest.fixture
def world():
    """A fresh mock world for each test."""
    return World()
