import argparse
import os
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import kipy.board
import kipy.board_types
from kipy.board import BoardLayer as BL
from loguru import logger

import breakneck.footprint
import breakneck.overlay
from breakneck.clearvias import stitching_vias
from breakneck.commit import (
    commit_label,
    marker_shapes,
    push_tolerant,
    refill_after_commit,
)
from breakneck.overlay import DEFAULT_OVERLAY_LAYERS
from breakneck.padvia import PadviaOptions, plan_padvia
from breakneck.placement import (
    BoardChangedError,
    RunSettings,
    SplitError,
    run_placement,
)
from breakneck.plan import (
    POURSTITCH_FALLBACK_VIA,
    POURSTITCH_GROUND_DISTANCE,
    POURSTITCH_POWER_DISTANCE,
    StitchClass,
)
from breakneck.returnvia import DEFAULT_DISTANCE, NoGroundNetsError, ReturnViaOptions
from breakneck.snapshot import (
    Snapshot,
    ViaSize,
    outline_from_edge_paths,
    read_edge_paths,
)


def break_tracks(
    board: kipy.board.Board,
    footprints: Sequence[breakneck.footprint.BNFootprint],
    tracks: Sequence[kipy.board_types.Track | kipy.board_types.ArcTrack],
    neck_length: float = 10,
    dry_run: bool = False,
) -> None:
    """
    Break tracks crossing the courtyards of footprints on a board.
    """
    all_tracks = {t.id.value: t for t in tracks}
    logger.debug("Getting all courtyards")

    max_width = max(t.width for t in all_tracks.values())
    remove_dict = {}
    create_dict = {}

    neck_length_nm = int(neck_length * 1000000)  # mm to nm

    for fpc in footprints:
        logger.debug(f"Breaking tracks for {fpc.ref}")
        items_to_remove, items_to_create = fpc.break_tracks(
            list(all_tracks.values()), max_width, neck_length_nm
        )

        # remove items from all_tracks and create_list
        for item in items_to_remove:
            all_tracks.pop(item.id.value)
            create_dict.pop(item.id.value, None)

        # add new items to all_tracks and create_list
        for item in items_to_create:
            all_tracks[item.id.value] = item
            create_dict[item.id.value] = item

        # add items to remove_list
        for item in items_to_remove:
            remove_dict[item.id.value] = item

    commit = board.begin_commit()
    board.remove_items(list(remove_dict.values()))
    board.create_items(list(create_dict.values()))
    if dry_run:
        board.drop_commit(commit)
    else:
        board.push_commit(commit)


def execute_cut(args):
    kicad = kipy.KiCad()
    board = kicad.get_board()

    tracks = board.get_tracks()
    footprints = board.get_footprints()

    num_all_tracks = len(tracks)
    num_all_footprints = len(footprints)

    if args.selection:
        items = board.get_selection()

        if len(items) == 0:
            logger.error("No items selected")
            return

        # get tracks from selection
        sel_tracks = [
            item
            for item in items
            if isinstance(item, (kipy.board_types.Track, kipy.board_types.ArcTrack))
        ]

        # get footprints from selection
        sel_footprints = [
            item
            for item in items
            if isinstance(item, kipy.board_types.FootprintInstance)
        ]

        # If selection has no tracks or footprints, use all tracks and footprints

        if sel_tracks:
            tracks = sel_tracks
        if sel_footprints:
            footprints = sel_footprints

    bnfootprints = breakneck.footprint.get_bn_footprints(footprints)

    if args.sides:
        if args.sides == "front":
            tracks = [t for t in tracks if t.layer == BL.BL_F_Cu]
            bnfootprints = [
                fpc for fpc in bnfootprints if not fpc.front_courtyard.is_empty
            ]
        elif args.sides == "back":
            tracks = [t for t in tracks if t.layer == BL.BL_B_Cu]
            bnfootprints = [
                fpc for fpc in bnfootprints if not fpc.back_courtyard.is_empty
            ]

    if args.netclass:
        nets = board.get_nets(args.netclass)
        net_names = {n.name for n in nets}
        tracks = [t for t in tracks if t.net in net_names]

    num_tracks = len(tracks)
    num_footprints = len(bnfootprints)

    logger.info(
        f"Breaking tracks on {num_tracks}/{num_all_tracks} tracks "
        f"and {num_footprints}/{num_all_footprints} footprints"
    )

    if num_tracks == 0:
        logger.warning("No tracks to break")
        return

    if num_footprints == 0:
        logger.warning("No footprints to break tracks on")
        return

    break_tracks(board, bnfootprints, tracks, dry_run=args.dry_run)


# Placement failures that leave the board untouched and need no traceback.
PLACEMENT_ERRORS = (SplitError, BoardChangedError)


@contextmanager
def _exit_on(*errors: type[Exception]) -> Iterator[None]:
    """Report an expected failure as one log line and exit 1."""
    try:
        yield
    except errors as e:
        logger.error(str(e))
        raise SystemExit(1) from None


@contextmanager
def _needs_place_extra() -> Iterator[None]:
    """Report a missing dependency of the place extra with the install hint."""
    try:
        yield
    except ModuleNotFoundError as e:
        logger.error(
            f"{e}; install the place extra: uv tool install 'breakneck[place]'"
        )
        raise SystemExit(1)


def parse_via_size(text: str) -> ViaSize:
    """Parse DIAMETER/DRILL in mm, e.g. 0.6/0.3."""
    try:
        dia, drill = (float(v) for v in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected DIAMETER/DRILL, got {text!r}")
    if not 0 < drill < dia:
        raise argparse.ArgumentTypeError(f"need 0 < drill < diameter, got {text!r}")
    return ViaSize(dia, drill)


def add_pourstitch_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "pourstitch",
        help="Stitch pours so every point is within a distance of a via",
        description="Stitch the ground nets' pours, or with --power every other "
        "net's pours, so every point is within a distance of a via.",
    )
    p.add_argument(
        "--power",
        action="store_true",
        help="Stitch the nets that are not ground instead of the ground nets",
    )
    p.add_argument(
        "--nets",
        nargs="+",
        metavar="NET",
        help="Nets to stitch, within the run's class (default: every net of the "
        "class with pours on two or more layers)",
    )
    _add_ground_nets_argument(p)
    p.add_argument(
        "--distance",
        "-d",
        type=float,
        help="Maximum distance to a stitch, through the copper, in mm "
        f"(default: {POURSTITCH_GROUND_DISTANCE:g} for ground, "
        f"{POURSTITCH_POWER_DISTANCE:g} with --power)",
    )
    p.add_argument(
        "--via",
        type=parse_via_size,
        metavar="DIAMETER/DRILL",
        help="Via size in mm, overriding the net's netclass size",
    )
    p.add_argument(
        "--fallback-via",
        type=parse_via_size,
        default=POURSTITCH_FALLBACK_VIA,
        metavar="DIAMETER/DRILL",
        help="Smaller via size in mm for spots the netclass size doesn't fit",
    )
    _add_heatmap_arguments(p, "Only measure the existing stitching; place no vias")
    p.add_argument("--kicad-cli", metavar="PATH", help="kicad-cli binary used for DRC")
    _add_allow_splits_argument(p)


def _add_heatmap_arguments(parser: argparse.ArgumentParser, check_help: str) -> None:
    parser.add_argument("--check", action="store_true", help=check_help)
    parser.add_argument(
        "--render", type=Path, metavar="PATH", help="Write the heatmap panels to PATH"
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Place per-layer heatmaps on the board as reference images",
    )
    parser.add_argument(
        "--overlay-layers",
        nargs="+",
        metavar="LAYER",
        default=list(DEFAULT_OVERLAY_LAYERS),
        help="User layers for the overlays, one per copper layer in stack order",
    )


def _add_allow_splits_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-splits",
        action="store_true",
        help="Commit even when DRC finds new unconnected or isolated copper "
        "that no via could be blamed for; the spots get markers",
    )


def _add_ground_nets_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ground-nets",
        nargs="+",
        metavar="NET",
        help="Ground nets (default: nets with zones on two or more inner layers, "
        "else GND)",
    )


def execute_returnvia(args):
    with _needs_place_extra():
        from breakneck import returnpath, returnvia_command

    options = ReturnViaOptions(
        distance=args.distance,
        ground_nets=tuple(args.ground_nets) if args.ground_nets else None,
        via=args.via,
        force=args.force,
    )
    outputs = breakneck.overlay.Outputs(
        render=args.render,
        overlay=args.overlay,
        overlay_layers=tuple(args.overlay_layers),
    )
    with _exit_on(
        returnpath.NoBoardOutlineError,
        NoGroundNetsError,
        breakneck.overlay.OverlayLayersError,
        *PLACEMENT_ERRORS,
    ):
        returnvia_command.execute(
            options,
            outputs,
            dry_run=args.dry_run,
            check=args.check,
            kicad_cli=args.kicad_cli,
            allow_splits=args.allow_splits,
            full_map=args.full_map,
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Do everything except commit the changes",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    cut_parser = subparsers.add_parser("cut", help="Add neckdowns to tracks")

    cut_parser.add_argument(
        "--selection", action="store_true", help="Use selected items"
    )

    cut_parser.add_argument(
        "--sides",
        choices=["front", "back", "both"],
        default="both",
        help="Side of the board to break tracks on",
    )

    cut_parser.add_argument(
        "--length",
        "-l",
        type=float,
        default=10,
        help="Neckdown length outside of the courtyard, in mm",
    )

    cut_parser.add_argument("--netclass", type=str, help="Netclass to break tracks on")

    padvia_parser = subparsers.add_parser(
        "padvia", help="Place a via and stub for every unserved SMD pad"
    )
    padvia_parser.add_argument(
        "--nets",
        nargs="+",
        metavar="NET",
        help="Nets whose SMD pads get vias, among pads with their net's fill, "
        "wide enough for a via, within reach on another layer "
        "(default: every such net)",
    )
    padvia_parser.add_argument(
        "--via",
        type=parse_via_size,
        metavar="DIAMETER/DRILL",
        help="Via size in mm, overriding the net's netclass size",
    )
    padvia_parser.add_argument(
        "--kicad-cli", metavar="PATH", help="kicad-cli binary used for DRC"
    )
    _add_allow_splits_argument(padvia_parser)

    add_pourstitch_parser(subparsers)

    clear_parser = subparsers.add_parser(
        "clear",
        help="Remove annotations from User.Eco2 and breakneck's overlay images",
    )
    clear_parser.add_argument(
        "--overlay-layers",
        nargs="+",
        metavar="LAYER",
        default=list(DEFAULT_OVERLAY_LAYERS),
        help="User layers to remove breakneck overlays from",
    )
    clear_parser.add_argument(
        "--vias",
        action="store_true",
        help="Also remove unlocked vias touching a pour of their own net that no "
        "track or pad ties down",
    )
    clear_parser.add_argument(
        "--nets",
        nargs="+",
        metavar="NET",
        help="Nets whose vias --vias removes (default: every net with a zone)",
    )
    returnvia_parser = subparsers.add_parser(
        "returnvia", help="Place a ground return via near every signal via"
    )
    returnvia_parser.add_argument(
        "--distance",
        "-d",
        type=float,
        default=DEFAULT_DISTANCE,
        help="Maximum distance from a via to its return via, in mm",
    )
    _add_ground_nets_argument(returnvia_parser)
    returnvia_parser.add_argument(
        "--via",
        type=parse_via_size,
        metavar="DIAMETER/DRILL",
        help="Return via size in mm (default: the ground net's netclass)",
    )
    returnvia_parser.add_argument(
        "--force",
        action="store_true",
        help="Place best-effort vias where none fits with full clearance",
    )
    _add_heatmap_arguments(
        returnvia_parser, "Only measure the existing returns; place no vias"
    )
    returnvia_parser.add_argument(
        "--full-map",
        action="store_true",
        help="Map all copper, not only the copper around unserved via ends",
    )
    returnvia_parser.add_argument(
        "--kicad-cli", metavar="PATH", help="kicad-cli binary used for DRC"
    )
    _add_allow_splits_argument(returnvia_parser)

    args = parser.parse_args()
    if args.command == "clear" and args.nets and not args.vias:
        clear_parser.error("--nets requires --vias")
    return args


def execute_clear(args):
    kicad = kipy.KiCad()
    board = kicad.get_board()

    shapes = board.get_shapes()
    eco2_shapes = marker_shapes(shapes)
    outline = outline_from_edge_paths(read_edge_paths(shapes))
    images = (
        []
        if outline.is_empty
        else breakneck.overlay.overlays_on(
            board,
            args.overlay_layers,
            breakneck.overlay.board_placement(outline.bounds),
        )
    )

    vias = (
        _find_and_log_stitching_vias(board, args.nets, args.dry_run)
        if args.vias
        else []
    )

    if not eco2_shapes and not images and not vias:
        logger.info("Nothing to clear")
        return

    commit = board.begin_commit()
    board.remove_items([*eco2_shapes, *images, *vias])
    if args.dry_run:
        board.drop_commit(commit)
        return
    push_tolerant(board, commit, commit_label("clear"))
    if vias:
        # Close the fills over the holes the removed vias leave.
        refill_after_commit(kicad, board)


def _find_and_log_stitching_vias(
    board: kipy.board.Board, nets: Sequence[str] | None, dry_run: bool
) -> list[kipy.board_types.Via]:
    snapshot = Snapshot.from_board(board)
    _warn_unclearable_nets(snapshot, nets)
    found = stitching_vias(snapshot, nets)
    if found:
        counts = Counter(v.net for v in found)
        logger.info(
            f"{'Would remove' if dry_run else 'Removing'} stitching vias: "
            + ", ".join(f"{n} on {net}" for net, n in counts.items())
        )
    ids = {v.id for v in found}
    return [v for v in board.get_vias() if v.id.value in ids]


def _warn_unclearable_nets(snapshot: Snapshot, nets: Sequence[str] | None) -> None:
    zoned = {z.net for z in snapshot.copper_zones}
    selected = zoned if nets is None else set(nets)
    for net in sorted(selected - zoned):
        logger.warning(f"No copper zone on net {net}; it has no vias to clear")
    unfilled = Counter(
        z.net for z in snapshot.copper_zones if z.net in selected and not z.fills
    )
    for net, n in sorted(unfilled.items()):
        logger.warning(
            f"{n} zone{'s' if n > 1 else ''} of {net} "
            f"{'have' if n > 1 else 'has'} no fill, so vias that touch only "
            f"{'them' if n > 1 else 'it'} stay; press B in KiCad to refill zones "
            "and run again"
        )


def execute_padvia(args):
    options = PadviaOptions(
        nets=None if args.nets is None else tuple(args.nets), via=args.via
    )
    settings = RunSettings(
        "padvia",
        dry_run=args.dry_run,
        kicad_cli=args.kicad_cli,
        allow_splits=args.allow_splits,
    )
    with _exit_on(*PLACEMENT_ERRORS):
        run_placement(plan_padvia, options, settings)


def execute_pourstitch(args):
    with _needs_place_extra():
        from breakneck import pourstitch, pourstitch_command

    options = pourstitch.Options(
        nets=None if args.nets is None else tuple(args.nets),
        stitch_class=StitchClass.POWER if args.power else StitchClass.GROUND,
        ground_nets=tuple(args.ground_nets) if args.ground_nets else None,
        distance=args.distance,
        via=args.via,
        fallback_via=args.fallback_via,
    )
    outputs = breakneck.overlay.Outputs(
        render=args.render,
        overlay=args.overlay,
        overlay_layers=tuple(args.overlay_layers),
    )
    with _exit_on(
        NoGroundNetsError,
        pourstitch.OtherClassNetsError,
        breakneck.overlay.OverlayLayersError,
        *PLACEMENT_ERRORS,
    ):
        pourstitch_command.execute(
            options,
            outputs,
            dry_run=args.dry_run,
            check=args.check,
            kicad_cli=args.kicad_cli,
            allow_splits=args.allow_splits,
        )


DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_VARIABLE = "LOGURU_LEVEL"


def configure_logging() -> None:
    """Log to stderr at INFO, or at the level in LOGURU_LEVEL.

    Loguru's own default sink logs at DEBUG.
    """
    logger.remove()
    logger.add(sys.stderr, level=os.environ.get(LOG_LEVEL_VARIABLE, DEFAULT_LOG_LEVEL))


def main():
    configure_logging()
    args = parse_args()

    match args.command:
        case "cut":
            execute_cut(args)

        case "padvia":
            execute_padvia(args)
        case "pourstitch":
            execute_pourstitch(args)
        case "returnvia":
            execute_returnvia(args)

        case "clear":
            execute_clear(args)


if __name__ == "__main__":
    main()
