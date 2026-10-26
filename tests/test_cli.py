import argparse
import sys

import pytest

from breakneck import cli, placement
from breakneck.cli import parse_via_size
from breakneck.snapshot import ViaSize


def test_via_size_parses_diameter_and_drill():
    assert parse_via_size("0.5/0.3") == ViaSize(0.5, 0.3)


@pytest.mark.parametrize("text", ["0.5", "0.6/0.3/0.1", "a/b", "0.3/0.5", "0.5/0"])
def test_bad_via_size_is_rejected(text):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_via_size(text)


@pytest.mark.parametrize(
    "argv",
    [
        ["returnvia"],
        ["padvia", "--nets", "GND"],
        ["pourstitch", "--nets", "GND"],
    ],
)
def test_placement_commands_take_allow_splits(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["breakneck", *argv])
    assert not cli.parse_args().allow_splits
    monkeypatch.setattr(sys, "argv", ["breakneck", *argv, "--allow-splits"])
    assert cli.parse_args().allow_splits


def test_refused_split_exits_with_an_error(monkeypatch):
    def refuse(planner, options, settings):
        raise placement.SplitError("split")

    monkeypatch.setattr(cli, "run_placement", refuse)
    args = argparse.Namespace(
        nets=["GND"], via=None, dry_run=False, kicad_cli=None, allow_splits=False
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.execute_padvia(args)

    assert exit_info.value.code == 1
