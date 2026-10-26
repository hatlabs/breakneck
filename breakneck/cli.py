import argparse
from collections.abc import Sequence
from pathlib import Path

import kipy.board
import kipy.board_types
import shapely.geometry
from kipy.board import BoardLayer as BL
from kipy.proto.board import board_types_pb2
from kipy.proto.common import base_types_pb2
from loguru import logger

import breakneck.conversions
import breakneck.footprint
import breakneck.overlay
from breakneck.overlay import DEFAULT_OVERLAY_LAYERS
from breakneck.placement import RunSettings, run_placement
from breakneck.returnvia import DEFAULT_DISTANCE, ReturnViaOptions, plan_return_vias
from breakneck.snapshot import ViaSize


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
            if isinstance(item, kipy.board_types.Track)
            or isinstance(item, kipy.board_types.ArcTrack)
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
        net_names = set(n.name for n in nets)
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
    max_distance = args.distance * 1000000  # mm to nm

    kicad = kipy.KiCad()
    board = kicad.get_board()

    vias = board.get_vias()

    all_pads = board.get_pads()
    gnd_tht_pads = [
        pad
        for pad in all_pads
        if pad.net.name == "GND" and pad.proto.type == kipy.board_types.PadType.PT_PTH
    ]

    # We treat GND THT pads as vias as well
    gnd_vias = [via for via in vias if via.net.name == "GND"] + gnd_tht_pads

    gnd_via_coords = [
        breakneck.conversions.as_coords2d(via.position) for via in gnd_vias
    ]
    gnd_via_points = [shapely.geometry.Point(coords) for coords in gnd_via_coords]

    gnd_via_tree = shapely.STRtree(gnd_via_points)

    non_gnd_vias = [via for via in vias if via.net.name != "GND"]
    non_gnd_via_coords = [
        breakneck.conversions.as_coords2d(via.position) for via in non_gnd_vias
    ]
    non_gnd_via_points = [
        shapely.geometry.Point(coords) for coords in non_gnd_via_coords
    ]

    nearests_idxs = [gnd_via_tree.nearest(pt) for pt in non_gnd_via_points]

    nearests = [gnd_via_points[idx] for idx in nearests_idxs]
    distances = [
        shapely.distance(nearest, non_gnd_via)
        for nearest, non_gnd_via in zip(nearests, non_gnd_via_points)
    ]

    # Get those non-gnd vias that are more than 3000000 nm away from the nearest gnd via
    lone_via_idxs = [
        i for i, distance in enumerate(distances) if distance > max_distance
    ]

    new_lines = []
    for idx in lone_via_idxs:
        # Draw a line from the lone via to the nearest gnd via on layer BL.BL_Eco2_User
        # with thickness 0.05mm = 50000 nm
        ngvc = non_gnd_via_coords[idx]
        gvc = gnd_via_coords[nearests_idxs[idx]]
        start_proto_vector = base_types_pb2.Vector2(x_nm=ngvc[0], y_nm=-ngvc[1])
        end_proto_vector = base_types_pb2.Vector2(x_nm=gvc[0], y_nm=-gvc[1])

        shape = base_types_pb2.GraphicShape(
            segment=base_types_pb2.GraphicSegmentAttributes(
                start=start_proto_vector, end=end_proto_vector
            )
        )

        board_graphic_shape = board_types_pb2.BoardGraphicShape(shape=shape)

        segment = kipy.board_types.BoardSegment(proto=board_graphic_shape)
        segment.layer = BL.BL_Eco2_User
        segment.attributes.stroke.width = 100000
        new_lines.append(segment)

    shapes = board.get_shapes()
    eco2_shapes = [shape for shape in shapes if shape.layer == BL.BL_Eco2_User]

    commit = board.begin_commit()
    board.remove_items(eco2_shapes)
    board.create_items(new_lines)
    if args.dry_run:
        board.drop_commit(commit)
    else:
        board.push_commit(commit)


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
        default=2.5,
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
        default="0.5/0.3",
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


def _add_ground_nets_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ground-nets",
        nargs="+",
        metavar="NET",
        help="Ground nets (default: nets with zones on two or more inner layers)",
    )


def execute_returnvia(args):
    options = ReturnViaOptions(
        distance=args.distance,
        ground_nets=tuple(args.ground_nets) if args.ground_nets else None,
        via=args.via,
        force=args.force,
    )
    settings = RunSettings(
        command="returnvia", dry_run=args.dry_run, kicad_cli=args.kicad_cli
    )
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

    gnd_via_parser = subparsers.add_parser("gndvia", help="Visualize GND via distances")

    gnd_via_parser.add_argument(
        "--distance",
        "-d",
        type=float,
        default=2,
        help="Maximum distance to GND via, in mm",
    )

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
    from breakneck.padvia import PadviaOptions, plan_padvia
    from breakneck.placement import RunSettings, run_placement

    options = PadviaOptions(nets=tuple(args.nets), via=args.via)
    settings = RunSettings("padvia", dry_run=args.dry_run, kicad_cli=args.kicad_cli)
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
    try:
        pourstitch_command.execute(
            options,
            outputs,
            dry_run=args.dry_run,
            check=args.check,
            kicad_cli=args.kicad_cli,
        )
    except pourstitch_command.OverlayLayersError as e:
        logger.error(str(e))
        raise SystemExit(1)


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
