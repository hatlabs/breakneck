import io
import json
import stat
import sys
from pathlib import Path

import pytest
from kipy.errors import ApiError

from breakneck import drc
from breakneck.plan import Plan, PlannedUnit, Stub, UnfixableSpot
from breakneck.snapshot import Snapshot, ViaSize

VIA = ViaSize(0.6, 0.3)

BOARD_WITH_FONTS = """(kicad_pcb
\t(version 20250000)
\t(generator "pcbnew")
\t(net 0 "")
\t(net 1 "GND")
\t(segment
\t\t(start 1 1)
\t\t(end 2 1)
\t\t(width 0.2)
\t\t(layer "F.Cu")
\t\t(net "GND")
\t\t(uuid "aaaa")
\t)
\t(embedded_fonts no)
)
"""

BOARD_WITHOUT_FONTS = """(kicad_pcb
\t(version 20250000)
\t(generator "pcbnew")
\t(net 1 "GND")
)
"""


# ---- minimal s-expression reader for checking injector output -------------


def tokenize(text):
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "()":
            yield c
            i += 1
        elif c.isspace():
            i += 1
        elif c == '"':
            j, buf = i + 1, []
            while text[j] != '"':
                if text[j] == "\\":
                    j += 1
                buf.append(text[j])
                j += 1
            yield ("str", "".join(buf))
            i = j + 1
        else:
            j = i
            while j < n and not text[j].isspace() and text[j] not in '()"':
                j += 1
            yield text[i:j]
            i = j


def parse_sexpr(text):
    stack = [[]]
    for tok in tokenize(text):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            node = stack.pop()
            stack[-1].append(node)
        else:
            stack[-1].append(tok[1] if isinstance(tok, tuple) else tok)
    assert len(stack) == 1, "unbalanced parentheses"
    (root,) = stack[0]
    return root


def children(node, name):
    return [c for c in node if isinstance(c, list) and c and c[0] == name]


def field(node, name):
    (f,) = children(node, name)
    return f[1:]


def unit(x, y, net="GND", stub=None, forced=False):
    return PlannedUnit(position=(x, y), net=net, size=VIA, stub=stub, forced=forced)


# ---- injector ---------------------------------------------------------------


@pytest.mark.parametrize("board", [BOARD_WITH_FONTS, BOARD_WITHOUT_FONTS])
def test_injector_adds_vias_and_stubs_without_free(board):
    stub = Stub(start=(10.0, 10.0), end=(11.0, 10.0), width=0.25, layer="F.Cu")
    units = [
        unit(11.0, 10.0, stub=stub),
        unit(20.123456, 30.5),
        unit(5, 5, net='Odd "net"', forced=True),
    ]

    before = parse_sexpr(board)
    after = parse_sexpr(drc.inject(board, units))

    assert after[0] == "kicad_pcb"
    other = lambda root: [c[0] for c in root[1:] if c[0] not in ("via", "segment")]
    assert other(after) == other(before)
    vias = children(after, "via")
    new_segments = len(children(after, "segment")) - len(children(before, "segment"))
    assert len(vias) == 3
    assert new_segments == 1
    for v in vias:
        assert not children(v, "free")
        assert field(v, "layers") == ["F.Cu", "B.Cu"]
        assert len(field(v, "uuid")[0]) == 36
    assert field(vias[1], "at") == ["20.123456", "30.5"]
    assert field(vias[1], "size") == ["0.6"]
    assert field(vias[1], "drill") == ["0.3"]
    assert field(vias[2], "net") == ['Odd "net"']
    seg = children(after, "segment")[-1]
    assert field(seg, "start") == ["10", "10"]
    assert field(seg, "end") == ["11", "10"]
    assert field(seg, "width") == ["0.25"]
    assert field(seg, "layer") == ["F.Cu"]
    assert field(seg, "net") == ["GND"]


def test_injector_places_items_before_embedded_fonts():
    out = drc.inject(BOARD_WITH_FONTS, [unit(1, 1)])

    assert out.index("(via") < out.index("(embedded_fonts")


def test_injector_with_no_units_returns_board_unchanged():
    assert drc.inject(BOARD_WITH_FONTS, []) == BOARD_WITH_FONTS


# ---- report parsing and diff -------------------------------------------------


def item(desc, x, y):
    return {"description": desc, "pos": {"x": x, "y": y}, "uuid": "u"}


def violation(vtype, *items, excluded=False):
    v = {
        "description": f"{vtype} issue",
        "items": list(items),
        "severity": "error",
        "type": vtype,
    }
    if excluded:
        v["excluded"] = True
    return v


def report(violations=(), unconnected=()):
    return {"violations": list(violations), "unconnected_items": list(unconnected)}


def test_new_violation_is_planned_minus_baseline_as_multiset():
    old = violation("clearance", item("Via [GND]", 1, 1), item("Track [X]", 1.2, 1))
    moved = violation("clearance", item("Via [GND]", 5, 5), item("Track [X]", 5.2, 5))
    baseline = drc.parse_report(report([old]))
    planned = drc.parse_report(report([old, moved]))

    new = drc.new_violations(baseline, planned)

    assert [v.items[0].position for v in new] == [(5, 5)]


def test_duplicate_violations_count_separately():
    v = violation("copper_sliver", item("Zone [GND]", 1, 1))
    baseline = drc.parse_report(report([v]))
    planned = drc.parse_report(report([v, v]))

    assert len(drc.new_violations(baseline, planned)) == 1


def test_key_ignores_float_noise_below_a_micron_and_item_order():
    a = violation("clearance", item("Via [GND]", 1.0000001, 1), item("Track [X]", 2, 2))
    b = violation("clearance", item("Track [X]", 2, 2), item("Via [GND]", 1.0, 1))
    baseline = drc.parse_report(report([a]))
    planned = drc.parse_report(report([b]))

    assert drc.new_violations(baseline, planned) == []


def test_excluded_violations_are_ignored_and_unconnected_items_are_parsed():
    parsed = drc.parse_report(
        report(
            [violation("clearance", item("Via [GND]", 1, 1), excluded=True)],
            [violation("unconnected_items", item("Pad 1 [GND]", 3, 3))],
        )
    )

    assert [v.type for v in parsed] == ["unconnected_items"]


# ---- attribution -------------------------------------------------------------


def test_violation_on_planned_via_is_attributed_by_position():
    units = [unit(1, 1), unit(5, 5)]
    v = drc.parse_report(report([violation("clearance", item("Via [GND]", 5.0004, 5))]))

    result = drc.attribute(v, units)

    assert set(result.dropped) == {1}
    assert result.unattributed == []


def test_track_item_on_planned_stub_attributes_to_its_unit():
    stub = Stub(start=(0, 0), end=(2, 0), width=0.3, layer="F.Cu")
    units = [unit(2, 0, stub=stub), unit(10, 10)]
    v = drc.parse_report(
        report(
            [
                violation(
                    "clearance",
                    item("Track [GND] on F.Cu", 1.0, 0.1),
                    item("Pad 1 [X]", 1, 0.4),
                )
            ]
        )
    )

    result = drc.attribute(v, units)

    assert set(result.dropped) == {0}


RENAMED = {"F.Cu": "Top", "B.Cu": "Bottom"}


@pytest.mark.parametrize("layer_name", ["Top", "F.Cu"])
def test_track_item_on_a_renamed_layer_attributes_to_the_stub(layer_name):
    # KiCad describes items with the board's own layer names.
    stub = Stub(start=(0, 0), end=(2, 0), width=0.3, layer="F.Cu")
    v = drc.parse_report(
        report([violation("clearance", item(f"Track [GND] on {layer_name}", 1, 0))])
    )

    result = drc.attribute(v, [unit(2, 0, stub=stub)], layer_names=RENAMED)

    assert set(result.dropped) == {0}


def test_track_item_on_another_renamed_layer_is_not_attributed():
    stub = Stub(start=(0, 0), end=(2, 0), width=0.3, layer="F.Cu")
    v = drc.parse_report(
        report([violation("clearance", item("Track [GND] on Bottom", 1, 0))])
    )

    result = drc.attribute(v, [unit(2, 0, stub=stub)], layer_names=RENAMED)

    assert result.dropped == {}


def test_prune_matches_stubs_by_the_snapshot_layer_names():
    stub = Stub(start=(0, 0), end=(2, 0), width=0.3, layer="F.Cu")
    snap = Snapshot(copper_layers=("F.Cu", "B.Cu"), layer_names=RENAMED)

    def planner(snapshot, options, forbidden):
        return Plan(units=() if forbidden else (unit(2, 0, stub=stub),))

    def check(plan):
        return [
            drc.Violation(
                "clearance",
                "c",
                "error",
                (drc.ViolationItem("Track [GND] on Top", (1, 0)),),
            )
        ]

    result = drc.prune(planner, snap, None, check)

    assert [d.unit.position for d in result.dropped] == [(2, 0)]
    assert result.unattributed == ()


def test_track_item_on_another_layer_is_not_attributed_to_the_stub():
    stub = Stub(start=(0, 0), end=(2, 0), width=0.3, layer="F.Cu")
    v = drc.parse_report(
        report([violation("clearance", item("Track [GND] on B.Cu", 0.0, 0.0))])
    )

    result = drc.attribute(v, [unit(2, 0, stub=stub)])

    assert result.dropped == {}


def test_track_item_off_the_stub_is_not_attributed():
    stub = Stub(start=(0, 0), end=(2, 0), width=0.3, layer="F.Cu")
    v = drc.parse_report(
        report([violation("clearance", item("Track [GND] on F.Cu", 1.0, 0.5))])
    )

    result = drc.attribute(v, [unit(2, 0, stub=stub)])

    assert result.dropped == {}
    assert len(result.unattributed) == 1


def test_starved_thermal_attributes_units_near_the_pad():
    units = [unit(10, 10), unit(11.5, 10), unit(20, 20)]
    v = drc.parse_report(
        report(
            [
                violation(
                    "starved_thermal",
                    item("Zone [GND] on F.Cu", 0, 0),
                    item("PTH pad 3 [GND] of J1", 10, 11),
                )
            ]
        )
    )

    result = drc.attribute(v, units, thermal_radius=2.5)

    assert set(result.dropped) == {0, 1}


def test_forced_via_clearance_violation_is_reported_not_dropped():
    units = [unit(1, 1, forced=True)]
    v = drc.parse_report(
        report(
            [violation("clearance", item("Via [GND]", 1, 1), item("Track [X]", 1.3, 1))]
        )
    )

    result = drc.attribute(v, units)

    assert result.dropped == {}
    assert [(f.unit, f.violation.type) for f in result.forced] == [(0, "clearance")]
    assert result.unattributed == []


@pytest.mark.parametrize(
    "vtype", ["starved_thermal", "unconnected_items", "shorting_items"]
)
def test_forced_via_non_clearance_violation_drops_it(vtype):
    units = [unit(1, 1, forced=True)]
    v = drc.parse_report(report([violation(vtype, item("Via [GND]", 1, 1))]))

    result = drc.attribute(v, units)

    assert set(result.dropped) == {0}


def test_violation_between_forced_and_regular_unit_drops_only_the_regular():
    units = [unit(1, 1, forced=True), unit(1.5, 1)]
    v = drc.parse_report(
        report(
            [
                violation(
                    "hole_to_hole", item("Via [GND]", 1, 1), item("Via [GND]", 1.5, 1)
                )
            ]
        )
    )

    result = drc.attribute(v, units)

    assert set(result.dropped) == {1}
    assert [f.unit for f in result.forced] == [0]


# ---- prune loop ----------------------------------------------------------------


class ScriptedPlanner:
    """Places a via at each candidate position that isn't forbidden."""

    def __init__(self, candidates, forced=()):
        self.candidates = candidates
        self.forced = set(forced)
        self.calls = []

    def __call__(self, snapshot, options, forbidden):
        self.calls.append(list(forbidden))
        units = tuple(
            unit(*p, forced=p in self.forced)
            for p in self.candidates
            if p not in forbidden
        )
        return Plan(units=units, unfixable=(UnfixableSpot((0, 0), "GND", "no room"),))


def check_for(bad_positions, unattributed=()):
    def check(plan):
        out = [
            drc.Violation(
                "clearance",
                "too close",
                "error",
                (drc.ViolationItem("Via [GND]", u.position),),
            )
            for u in plan.units
            if u.position in bad_positions
        ]
        out += [
            drc.Violation(
                "copper_sliver",
                "sliver",
                "warning",
                (drc.ViolationItem("Zone [GND]", p),),
            )
            for p in unattributed
        ]
        return out

    return check


SNAP = Snapshot(copper_layers=("F.Cu", "B.Cu"))


def test_prune_drops_attributed_units_and_replans_with_them_forbidden():
    planner = ScriptedPlanner([(1, 1), (2, 2), (3, 3)])

    result = drc.prune(planner, SNAP, None, check_for({(2, 2)}))

    assert planner.calls == [[], [(2, 2)]]
    assert [u.position for u in result.plan.units] == [(1, 1), (3, 3)]
    assert [(d.unit.position, d.violations[0].type) for d in result.dropped] == [
        ((2, 2), "clearance")
    ]
    assert result.rounds == 2


def test_prune_stops_after_a_round_that_drops_nothing_and_reports_unattributed():
    planner = ScriptedPlanner([(1, 1)])

    result = drc.prune(planner, SNAP, None, check_for(set(), unattributed=[(7, 7)]))

    assert len(planner.calls) == 1
    assert [v.type for v in result.unattributed] == ["copper_sliver"]
    assert result.plan.unfixable[0].reason == "no room"


def test_prune_keeps_forced_units_with_clearance_violations():
    planner = ScriptedPlanner([(1, 1)], forced=[(1, 1)])

    result = drc.prune(planner, SNAP, None, check_for({(1, 1)}))

    assert [u.position for u in result.plan.units] == [(1, 1)]
    assert [f.violation.type for f in result.forced] == ["clearance"]


def test_prune_round_limit_drops_still_attributed_units_without_replanning():
    def ignores_forbidden(snapshot, options, forbidden):
        return Plan(units=(unit(1, 1), unit(2, 2)))

    result = drc.prune(ignores_forbidden, SNAP, None, check_for({(1, 1)}), max_rounds=2)

    assert result.rounds == 2
    assert [u.position for u in result.plan.units] == [(2, 2)]
    assert result.round_limit_hit


def test_prune_skips_drc_for_an_empty_plan():
    def check(plan):
        raise AssertionError("DRC should not run")

    result = drc.prune(lambda s, o, f: Plan(), SNAP, None, check)

    assert result.plan.units == ()


# ---- kicad-cli lookup ----------------------------------------------------------


def make_executable(path: Path, body: str = "") -> Path:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class FakeKiCad:
    def __init__(self, path=None, error=None):
        self.path = path
        self.error = error

    def get_kicad_binary_path(self, name):
        if self.error:
            raise self.error
        return self.path


@pytest.fixture
def no_system_cli(monkeypatch):
    monkeypatch.setattr(drc.shutil, "which", lambda name: None)
    monkeypatch.setattr(drc, "KNOWN_KICAD_CLI_PATHS", ())


def test_explicit_kicad_cli_wins(tmp_path, no_system_cli):
    explicit = make_executable(tmp_path / "explicit")
    other = make_executable(tmp_path / "other")

    assert drc.locate_kicad_cli(str(explicit), FakeKiCad(str(other))) == explicit


def test_missing_explicit_kicad_cli_is_an_error(tmp_path, no_system_cli):
    other = make_executable(tmp_path / "other")

    with pytest.raises(drc.KiCadCliNotFound):
        drc.locate_kicad_cli(str(tmp_path / "nope"), FakeKiCad(str(other)))


def test_kicad_cli_from_running_kicad_then_path_then_known(tmp_path, monkeypatch):
    from_api = make_executable(tmp_path / "api")
    on_path = make_executable(tmp_path / "onpath")
    known = make_executable(tmp_path / "known")
    monkeypatch.setattr(drc.shutil, "which", lambda name: str(on_path))
    monkeypatch.setattr(drc, "KNOWN_KICAD_CLI_PATHS", (str(known),))

    assert drc.locate_kicad_cli(None, FakeKiCad(str(from_api))) == from_api
    assert drc.locate_kicad_cli(None, FakeKiCad(error=ApiError("old KiCad"))) == on_path
    monkeypatch.setattr(drc.shutil, "which", lambda name: None)
    assert drc.locate_kicad_cli(None, FakeKiCad(str(tmp_path / "gone"))) == known


def test_kicad_cli_not_found_anywhere(no_system_cli):
    with pytest.raises(drc.KiCadCliNotFound):
        drc.locate_kicad_cli(None, FakeKiCad(""))


# ---- DRC runner against a fake kicad-cli ----------------------------------------

FAKE_CLI = """
import json, pathlib, sys
args = sys.argv[1:]
if args == ["version"]:
    print("10.0.7")
    sys.exit(0)
log = pathlib.Path(sys.argv[0]).with_suffix(".log")
with log.open("a") as f:
    f.write(json.dumps(args) + "\\n")
pcb = pathlib.Path(args[-1])
assert pcb.with_suffix(".kicad_pro").exists(), "project file not copied"
out = pathlib.Path(args[args.index("-o") + 1])
planned = "(via" in pcb.read_text()
items = [{"description": "Via [GND]", "pos": {"x": 1.0, "y": 2.0}}] if planned else []
violations = [{"type": "clearance", "description": "c", "severity": "error", "items": items}] if planned else []
out.write_text(json.dumps({"violations": violations, "unconnected_items": []}))
"""


def test_drc_runner_diffs_planned_copy_against_baseline(tmp_path):
    cli = make_executable(tmp_path / "kicad-cli", FAKE_CLI)
    project = tmp_path / "proj"
    project.mkdir()
    (project / "b.kicad_pro").write_text("{}")
    files = drc.ProjectFiles(pro=project / "b.kicad_pro", dru=None)
    work = tmp_path / "work"
    work.mkdir()

    runner = drc.DrcRunner(cli, files, BOARD_WITH_FONTS, work)
    new = runner.new_violations(Plan(units=(unit(1, 2),)))
    runner.new_violations(Plan(units=(unit(1, 2),)))

    assert [v.items[0].position for v in new] == [(1.0, 2.0)]
    calls = [
        json.loads(line)
        for line in (tmp_path / "kicad-cli.log").read_text().splitlines()
    ]
    assert len(calls) == 3  # one baseline, then one per check
    assert all(c[:3] == ["pcb", "drc", "--refill-zones"] for c in calls)
    assert not list(project.glob("*.kicad_pcb")), (
        "nothing written to the project directory"
    )


UTF8_CLI = """
import json, pathlib, sys
args = sys.argv[1:]
pcb = pathlib.Path(args[-1])
text = pcb.read_bytes().decode("utf-8")
assert "\\u03a9" in text, "board not written as UTF-8"
out = pathlib.Path(args[args.index("-o") + 1])
items = [{"description": "Via 10\\u03a9", "pos": {"x": 1.0, "y": 2.0}}]
violation = {"type": "clearance", "description": "c", "severity": "error", "items": items}
report = {"violations": [violation] if "(via" in text else []}
out.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
"""


def test_drc_runner_uses_utf8_regardless_of_locale(tmp_path, monkeypatch):
    # Emulate a Windows locale encoding, where Ω cannot be encoded.
    monkeypatch.setattr(
        io, "text_encoding", lambda encoding, stacklevel=2: encoding or "cp1252"
    )
    cli = make_executable(tmp_path / "kicad-cli", UTF8_CLI)
    (tmp_path / "b.kicad_pro").write_text("{}")
    files = drc.ProjectFiles(pro=tmp_path / "b.kicad_pro", dru=None)
    board = BOARD_WITH_FONTS.replace('(generator "pcbnew")', '(comment "10Ω")')

    runner = drc.DrcRunner(cli, files, board, tmp_path)
    new = runner.new_violations(Plan(units=(unit(1, 2),)))

    assert [v.items[0].description for v in new] == ["Via 10Ω"]


SLOW_CLI = "import time; time.sleep(30)"


def test_drc_runner_timeout_is_a_drc_error(tmp_path, monkeypatch):
    monkeypatch.setattr(drc, "DRC_TIMEOUT_S", 0.2)
    cli = make_executable(tmp_path / "kicad-cli", SLOW_CLI)
    (tmp_path / "b.kicad_pro").write_text("{}")
    files = drc.ProjectFiles(pro=tmp_path / "b.kicad_pro", dru=None)

    runner = drc.DrcRunner(cli, files, BOARD_WITH_FONTS, tmp_path)

    with pytest.raises(drc.DrcError, match="timed out after 0.2 s"):
        runner.new_violations(Plan(units=(unit(1, 2),)))


def test_kicad_cli_version_probe_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(drc, "VERSION_TIMEOUT_S", 0.2)
    cli = make_executable(tmp_path / "kicad-cli", SLOW_CLI)

    assert drc.kicad_cli_version(cli) == "unknown (version probe timed out)"


def test_drc_runner_raises_when_kicad_cli_fails(tmp_path):
    cli = make_executable(tmp_path / "kicad-cli", "import sys; sys.exit(3)")
    (tmp_path / "b.kicad_pro").write_text("{}")
    files = drc.ProjectFiles(pro=tmp_path / "b.kicad_pro", dru=None)

    runner = drc.DrcRunner(cli, files, BOARD_WITH_FONTS, tmp_path)

    with pytest.raises(drc.DrcError):
        runner.new_violations(Plan(units=(unit(1, 2),)))


# ---- project files -------------------------------------------------------------


class FakeProjectBoard:
    def __init__(self, directory, name):
        self.name = name
        self._dir = directory

    def get_project(self):
        return type("P", (), {"path": self._dir})()


def test_project_files_found_next_to_saved_board(tmp_path):
    (tmp_path / "b.kicad_pcb").write_text("")
    (tmp_path / "b.kicad_pro").write_text("{}")
    (tmp_path / "b.kicad_dru").write_text("")

    files = drc.project_files(FakeProjectBoard(str(tmp_path), "b.kicad_pcb"))

    assert files == drc.ProjectFiles(tmp_path / "b.kicad_pro", tmp_path / "b.kicad_dru")


def test_project_files_without_rules_file(tmp_path):
    (tmp_path / "b.kicad_pcb").write_text("")
    (tmp_path / "b.kicad_pro").write_text("{}")

    files = drc.project_files(FakeProjectBoard(str(tmp_path), "b.kicad_pcb"))

    assert files.dru is None


@pytest.mark.parametrize("directory", ["", "missing"])
def test_unsaved_board_is_refused(tmp_path, directory):
    path = str(tmp_path / directory) if directory else ""

    with pytest.raises(drc.ProjectFilesError):
        drc.project_files(FakeProjectBoard(path, "b.kicad_pcb"))


def test_board_without_project_file_is_refused(tmp_path):
    (tmp_path / "b.kicad_pcb").write_text("")

    with pytest.raises(drc.ProjectFilesError):
        drc.project_files(FakeProjectBoard(str(tmp_path), "b.kicad_pcb"))
