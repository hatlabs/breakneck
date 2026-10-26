import argparse

import pytest

from breakneck.cli import parse_via_size
from breakneck.snapshot import ViaSize


def test_via_size_parses_diameter_and_drill():
    assert parse_via_size("0.5/0.3") == ViaSize(0.5, 0.3)


@pytest.mark.parametrize("text", ["0.5", "0.6/0.3/0.1", "a/b", "0.3/0.5", "0.5/0"])
def test_bad_via_size_is_rejected(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_via_size(text)
