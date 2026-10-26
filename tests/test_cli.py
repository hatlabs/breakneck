import argparse
import sys

import pytest
from loguru import logger

import breakneck
from breakneck import cli, placement, pourstitch_command
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


def parsed(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["breakneck", *argv])
    return cli.parse_args()


@pytest.mark.parametrize(
    "argv, execute",
    [
        (["returnvia"], cli.execute_returnvia),
        (["padvia", "--nets", "GND"], cli.execute_padvia),
    ],
)
@pytest.mark.parametrize("allow", [False, True])
def test_allow_splits_reaches_the_run_settings(monkeypatch, argv, execute, allow):
    seen: list[placement.RunSettings] = []
    monkeypatch.setattr(
        cli, "run_placement", lambda planner, options, settings: seen.append(settings)
    )
    flag = ["--allow-splits"] if allow else []

    execute(parsed(monkeypatch, [*argv, *flag]))

    assert [s.allow_splits for s in seen] == [allow]


class Captured(Exception):
    pass


@pytest.mark.parametrize("allow", [False, True])
def test_pourstitch_allow_splits_reaches_the_run_settings(monkeypatch, allow):
    seen: list[placement.RunSettings] = []

    def capture(planner, options, settings, kicad):
        seen.append(settings)
        raise Captured

    monkeypatch.setattr(pourstitch_command, "run_placement", capture)
    monkeypatch.setattr(
        pourstitch_command.kipy,
        "KiCad",
        lambda: type("K", (), {"get_board": lambda self: None})(),
    )
    flag = ["--allow-splits"] if allow else []

    with pytest.raises(Captured):
        cli.execute_pourstitch(
            parsed(monkeypatch, ["pourstitch", "--nets", "GND", *flag])
        )

    assert [s.allow_splits for s in seen] == [allow]


@pytest.mark.parametrize(
    "error", [placement.SplitError("split"), placement.BoardChangedError("changed")]
)
def test_pourstitch_placement_failure_exits_cleanly(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(pourstitch_command, "execute", fail)

    with pytest.raises(SystemExit) as exit_info:
        cli.execute_pourstitch(parsed(monkeypatch, ["pourstitch", "--nets", "GND"]))

    assert exit_info.value.code == 1
    assert exit_info.value.__cause__ is None
    assert exit_info.value.__suppress_context__


@pytest.mark.parametrize(
    "argv, module",
    [
        (["returnvia"], "breakneck.returnpath"),
        (["pourstitch", "--nets", "GND"], "breakneck.pourstitch"),
    ],
)
def test_command_without_the_place_extra_exits_with_the_install_hint(
    monkeypatch, argv, module
):
    # A None entry makes the import raise ModuleNotFoundError; the package
    # attribute would otherwise satisfy `from breakneck import ...`.
    monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.delattr(breakneck, module.rpartition(".")[2], raising=False)
    monkeypatch.setattr(cli, "run_placement", lambda *a: pytest.fail("ran"))
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), format="{message}")
    args = parsed(monkeypatch, argv)

    try:
        with pytest.raises(SystemExit) as exit_info:
            getattr(cli, f"execute_{argv[0]}")(args)
    finally:
        logger.remove(sink)

    assert exit_info.value.code == 1
    assert any("uv tool install 'breakneck[place]'" in line for line in lines)
