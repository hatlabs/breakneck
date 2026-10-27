import pytest
from loguru import logger

from breakneck import commit


@pytest.fixture(autouse=True)
def _no_refill_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """KiCad fakes answer at once, so waiting a real poll interval only slows tests."""
    monkeypatch.setattr(commit, "REFILL_POLL_S", 0.0)


@pytest.fixture
def log():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{message}")
    yield lines
    logger.remove(sink)
