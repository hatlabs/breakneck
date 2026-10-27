import pytest

from breakneck import commit


@pytest.fixture(autouse=True)
def _no_refill_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """KiCad fakes answer at once, so waiting a real poll interval only slows tests."""
    monkeypatch.setattr(commit, "REFILL_POLL_S", 0.0)
