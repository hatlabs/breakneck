import argparse
import sys

import pytest
from loguru import logger

import breakneck
from breakneck import cli, overlay, placement, pourstitch_command, returnvia_command
from breakneck.cli import parse_via_size
from breakneck.plan import (
    POURSTITCH_GROUND_DISTANCE,
    POURSTITCH_POWER_DISTANCE,
    StitchClass,
)
from breakneck.returnpath import NoBoardOutlineError
from breakneck.returnvia import NoGroundNetsError
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


@pytest.mark.parametrize("allow", [False, True])
def test_allow_splits_reaches_the_run_settings(monkeypatch, allow):
    seen: list[placement.RunSettings] = []
    monkeypatch.setattr(
        cli, "run_placement", lambda planner, options, settings: seen.append(settings)
    )
    flag = ["--allow-splits"] if allow else []

    cli.execute_padvia(parsed(monkeypatch, ["padvia", "--nets", "GND", *flag]))

    assert [s.allow_splits for s in seen] == [allow]


class Captured(Exception):
    pass


@pytest.mark.parametrize(
    "argv, module",
    [
        (["pourstitch", "--nets", "GND"], pourstitch_command),
        (["returnvia"], returnvia_command),
    ],
)
@pytest.mark.parametrize("allow", [False, True])
def test_heatmap_command_allow_splits_reaches_the_run_settings(
    monkeypatch, argv, module, allow
):
    seen: list[placement.RunSettings] = []

    def capture(planner, options, settings, kicad):
        seen.append(settings)
        raise Captured

    monkeypatch.setattr(module, "run_placement", capture)
    monkeypatch.setattr(
        module.kipy,
        "KiCad",
        lambda: type("K", (), {"get_board": lambda self: None})(),
    )
    monkeypatch.setattr(returnvia_command, "check_outline", lambda board: None)
    flag = ["--allow-splits"] if allow else []

    with pytest.raises(Captured):
        getattr(cli, f"execute_{argv[0]}")(parsed(monkeypatch, [*argv, *flag]))

    assert [s.allow_splits for s in seen] == [allow]


def test_returnvia_takes_the_heatmap_options(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        returnvia_command,
        "execute",
        lambda options, outputs, **kw: seen.append((options, outputs, kw)),
    )
    argv = [
        "returnvia",
        "--check",
        "--render",
        str(tmp_path / "heat.png"),
        "--overlay",
        "--overlay-layers",
        "User.5",
        "User.6",
        "User.7",
        "User.8",
        "--distance",
        "2",
        "--full-map",
    ]

    cli.execute_returnvia(parsed(monkeypatch, argv))

    ((options, outputs, kw),) = seen
    assert options.distance == 2
    assert outputs == overlay.Outputs(
        render=tmp_path / "heat.png",
        overlay=True,
        overlay_layers=("User.5", "User.6", "User.7", "User.8"),
    )
    assert kw["check"]
    assert kw["full_map"]


@pytest.mark.parametrize("argv", [["returnvia"], ["pourstitch", "--nets", "GND"]])
def test_overlay_takes_no_directory(monkeypatch, argv):
    assert parsed(monkeypatch, [*argv, "--overlay"]).overlay is True
    with pytest.raises(SystemExit):
        parsed(monkeypatch, [*argv, "--overlay", "out"])


@pytest.mark.parametrize(
    "argv, dry_run, kicad_cli",
    [
        (["returnvia"], False, None),
        (
            ["--dry-run", "returnvia", "--kicad-cli", "/opt/kicad-cli"],
            True,
            "/opt/kicad-cli",
        ),
    ],
)
def test_returnvia_passes_dry_run_and_kicad_cli_through(
    monkeypatch, argv, dry_run, kicad_cli
):
    seen = []
    monkeypatch.setattr(
        returnvia_command, "execute", lambda options, outputs, **kw: seen.append(kw)
    )

    cli.execute_returnvia(parsed(monkeypatch, argv))

    (kw,) = seen
    assert kw["dry_run"] is dry_run
    assert kw["kicad_cli"] == kicad_cli


@pytest.mark.parametrize(
    "error",
    [
        NoBoardOutlineError("no outline"),
        NoGroundNetsError("no ground"),
        overlay.OverlayLayersError("layers"),
        placement.SplitError("split"),
    ],
)
def test_returnvia_failure_exits_cleanly(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(returnvia_command, "execute", fail)

    with pytest.raises(SystemExit) as exit_info:
        cli.execute_returnvia(parsed(monkeypatch, ["returnvia", "--check"]))

    assert exit_info.value.code == 1


def pourstitch_options(monkeypatch, argv):
    seen = []
    monkeypatch.setattr(
        pourstitch_command, "execute", lambda options, *a, **kw: seen.append(options)
    )
    cli.execute_pourstitch(parsed(monkeypatch, argv))
    (options,) = seen
    return options


def test_pourstitch_without_nets_leaves_net_selection_unset(monkeypatch):
    assert pourstitch_options(monkeypatch, ["pourstitch"]).nets is None


@pytest.mark.parametrize(
    ("argv", "stitch_class", "distance"),
    [
        ([], StitchClass.GROUND, POURSTITCH_GROUND_DISTANCE),
        (["--power"], StitchClass.POWER, POURSTITCH_POWER_DISTANCE),
        (["--distance", "4"], StitchClass.GROUND, 4.0),
        (["--power", "--distance", "3"], StitchClass.POWER, 3.0),
    ],
)
def test_pourstitch_distance_defaults_to_the_class_distance(
    monkeypatch, argv, stitch_class, distance
):
    options = pourstitch_options(monkeypatch, ["pourstitch", *argv])

    assert options.stitch_class is stitch_class
    assert options.distance == distance


def test_pourstitch_takes_ground_nets(monkeypatch):
    options = pourstitch_options(
        monkeypatch, ["pourstitch", "--ground-nets", "GND", "AGND"]
    )

    assert options.ground_nets == ("GND", "AGND")
    assert pourstitch_options(monkeypatch, ["pourstitch"]).ground_nets is None


@pytest.mark.parametrize(
    "error",
    [
        placement.SplitError("split"),
        placement.BoardChangedError("changed"),
        NoGroundNetsError("no ground"),
    ],
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


@pytest.fixture
def restore_default_sink():
    yield
    logger.remove()
    logger.add(sys.stderr)


@pytest.mark.usefixtures("restore_default_sink")
def test_cli_logs_info_by_default(monkeypatch, capsys):
    monkeypatch.delenv("LOGURU_LEVEL", raising=False)

    cli.configure_logging()
    logger.debug("per-via detail")
    logger.info("summary")

    err = capsys.readouterr().err
    assert "summary" in err
    assert "per-via detail" not in err


@pytest.mark.usefixtures("restore_default_sink")
def test_loguru_level_shows_debug(monkeypatch, capsys):
    monkeypatch.setenv("LOGURU_LEVEL", "DEBUG")

    cli.configure_logging()
    logger.debug("per-via detail")

    assert "per-via detail" in capsys.readouterr().err


@pytest.mark.usefixtures("restore_default_sink")
def test_main_logs_info_by_default(monkeypatch, capsys):
    monkeypatch.delenv("LOGURU_LEVEL", raising=False)
    monkeypatch.setattr(cli, "parse_args", lambda: argparse.Namespace(command=None))
    logger.remove()
    logger.add(sys.stderr)

    cli.main()
    logger.debug("per-via detail")

    assert "per-via detail" not in capsys.readouterr().err
