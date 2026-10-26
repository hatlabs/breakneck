import argparse
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import kipy.board
import kipy.board_types
from kipy.board import BoardLayer as BL
from loguru import logger

import breakneck.footprint
import breakneck.overlay
from breakneck.commit import Marker, MarkerKind, eco2_segment, marker_items
from breakneck.overlay import DEFAULT_OVERLAY_LAYERS
from breakneck.padvia import PadviaOptions, plan_padvia
from breakneck.placement import (
    BoardChangedError,
    RunSettings,
    SplitError,
    run_placement,
)
from breakneck.plan import POURSTITCH_DISTANCE, POURSTITCH_FALLBACK_VIA
from breakneck.returnvia import (
    DEFAULT_DISTANCE,
    NoGroundNetsError,
    ReturnViaOptions,
    plan_return_vias,
    return_lines,
)
from breakneck.snapshot import Snapshot, ViaSize


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


def execute_gndvia_check(args):
    """Draw a User.Eco2 line from each via lacking a return to its nearest one.

    Vias outside every ground zone, or whose ground has no return at all,
    get a marker instead.
    """
    kicad = kipy.KiCad()
    board = kicad.get_board()
    snapshot = Snapshot.from_board(board)

    with _exit_on(NoGroundNetsError):
        lines, unmarked = return_lines(snapshot, args.ground_nets, args.distance)
    logger.info(
        f"{len(lines) + len(unmarked)} vias without a return within "
        f"{args.distance} mm ({len(unmarked)} without a ground or any return)"
    )
    items = [eco2_segment(a, b) for a, b in lines] + marker_items(
        [Marker(p, MarkerKind.UNFIXABLE) for p in unmarked]
    )

    eco2_shapes = [
        shape for shape in board.get_shapes() if shape.layer == BL.BL_Eco2_User
    ]

    commit = board.begin_commit()
    board.remove_items(eco2_shapes)
    if items:
        board.create_items(items)
    if args.dry_run:
        board.drop_commit(commit)
    else:
        board.push_commit(commit)


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
    )
    p.add_argument(
        "--nets", nargs="+", metavar="NET", required=True, help="Nets to stitch"
    )
    p.add_argument(
        "--distance",
        "-d",
        type=float,
        default=POURSTITCH_DISTANCE,
        help="Maximum distance to a stitch, through the copper, in mm",
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
    p.add_argument(
        "--check",
        action="store_true",
        help="Only measure the existing stitching; place no vias",
    )
    p.add_argument(
        "--render", type=Path, metavar="PATH", help="Write the heatmap panels to PATH"
    )
    p.add_argument(
        "--overlay",
        nargs="?",
        type=Path,
        const=Path("."),
        metavar="DIR",
        help="Write per-layer heatmap PNGs to DIR (default: .) and place them "
        "on the board as reference images",
    )
    p.add_argument(
        "--overlay-layers",
        nargs="+",
        metavar="LAYER",
        default=list(DEFAULT_OVERLAY_LAYERS),
        help="User layers for the overlays, one per copper layer in stack order",
    )
    p.add_argument("--kicad-cli", metavar="PATH", help="kicad-cli binary used for DRC")
    _add_allow_splits_argument(p)


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
    options = ReturnViaOptions(
        distance=args.distance,
        ground_nets=tuple(args.ground_nets) if args.ground_nets else None,
        via=args.via,
        force=args.force,
    )
    settings = RunSettings(
        command="returnvia",
        dry_run=args.dry_run,
        kicad_cli=args.kicad_cli,
        allow_splits=args.allow_splits,
    )
    with _exit_on(NoGroundNetsError, *PLACEMENT_ERRORS):
        run_placement(plan_return_vias, options, settings)


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

    gnd_via_parser = subparsers.add_parser(
        "gndvia", help="Visualize vias without a nearby ground return via"
    )

    gnd_via_parser.add_argument(
        "--distance",
        "-d",
        type=float,
        default=DEFAULT_DISTANCE,
        help="Maximum distance to a return via, in mm",
    )
    _add_ground_nets_argument(gnd_via_parser)

    padvia_parser = subparsers.add_parser(
        "padvia", help="Place a via and stub for every unserved SMD pad of given nets"
    )
    padvia_parser.add_argument(
        "--nets",
        nargs="+",
        metavar="NET",
        required=True,
        help="Nets whose SMD pads get vias",
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
        help="Remove annotations from User.Eco2 and overlay images",
    )
    clear_parser.add_argument(
        "--overlay-layers",
        nargs="+",
        metavar="LAYER",
        default=list(DEFAULT_OVERLAY_LAYERS),
        help="User layers whose reference images are breakneck overlays",
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
    returnvia_parser.add_argument(
        "--kicad-cli", metavar="PATH", help="kicad-cli binary used for DRC"
    )
    _add_allow_splits_argument(returnvia_parser)

    return parser.parse_args()


def execute_clear(args):
    kicad = kipy.KiCad()
    board = kicad.get_board()

    shapes = board.get_shapes()
    eco2_shapes = [shape for shape in shapes if shape.layer == BL.BL_Eco2_User]
    images = breakneck.overlay.images_on(board, args.overlay_layers)

    if not eco2_shapes and not images:
        logger.info("Nothing to clear")
        return

    commit = board.begin_commit()
    board.remove_items([*eco2_shapes, *images])
    if args.dry_run:
        board.drop_commit(commit)
    else:
        board.push_commit(commit)


def execute_padvia(args):
    options = PadviaOptions(nets=tuple(args.nets), via=args.via)
    settings = RunSettings(
        "padvia",
        dry_run=args.dry_run,
        kicad_cli=args.kicad_cli,
        allow_splits=args.allow_splits,
    )
    with _exit_on(*PLACEMENT_ERRORS):
        run_placement(plan_padvia, options, settings)


def execute_pourstitch(args):
    try:
        from breakneck import pourstitch, pourstitch_command
    except ModuleNotFoundError as e:
        logger.error(
            f"{e}; install the place extra: uv tool install 'breakneck[place]'"
        )
        raise SystemExit(1)

    options = pourstitch.Options(
        nets=tuple(args.nets),
        distance=args.distance,
        via=args.via,
        fallback_via=args.fallback_via,
    )
    outputs = pourstitch_command.Outputs(
        render=args.render,
        overlay_dir=args.overlay,
        overlay_layers=tuple(args.overlay_layers),
    )
    with _exit_on(pourstitch_command.OverlayLayersError, *PLACEMENT_ERRORS):
        pourstitch_command.execute(
            options,
            outputs,
            dry_run=args.dry_run,
            check=args.check,
            kicad_cli=args.kicad_cli,
            allow_splits=args.allow_splits,
        )


def main():
    args = parse_args()

    match args.command:
        case "cut":
            execute_cut(args)

        case "gndvia":
            execute_gndvia_check(args)

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
