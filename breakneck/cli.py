import argparse
from collections.abc import Sequence

import kipy.board
import kipy.board_types
from loguru import logger

import breakneck.footprint


def break_tracks(
    board: kipy.board.Board,
    footprints: Sequence[breakneck.footprint.BNFootprint],
    tracks: Sequence[kipy.board_types.Track | kipy.board_types.ArcTrack],
) -> None:
    """
    Break all tracks crossing the courtyards of all footprints on a board.
    """
    all_tracks = {t.id.value: t for t in tracks}
    logger.debug("Getting all courtyards")

    max_width = max(t.width for t in all_tracks.values())
    remove_dict = {}
    create_dict = {}

    for fpc in footprints:
        logger.debug(f"Breaking tracks for {fpc.ref}")
        items_to_remove, items_to_create = fpc.break_tracks(
            list(all_tracks.values()), max_width
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
    board.push_commit(commit)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--selection", action="store_true", help="Use selected items")

    parser.add_argument(
        "--sides",
        choices=["front", "back", "both"],
        default="both",
        help="Side of the board to break tracks on",
    )

    parser.add_argument("--netclass", type=str, help="Netclass to break tracks on")

    return parser.parse_args()


def main():
    args = parse_args()

    kicad = kipy.KiCad()
    board = kicad.get_board()

    tracks = board.get_tracks()
    footprints = board.get_footprints()

    num_all_tracks = len(tracks)
    num_all_footprints = len(footprints)

    if args.selection:
        items = board.get_selection()

        # get tracks from selection
        tracks = [
            item
            for item in items
            if isinstance(item, kipy.board_types.Track)
            or isinstance(item, kipy.board_types.ArcTrack)
        ]

        # get footprints from selection
        footprints = [
            item
            for item in items
            if isinstance(item, kipy.board_types.FootprintInstance)
        ]

    bnfootprints = breakneck.footprint.get_bn_footprints(footprints)

    if args.sides:
        if args.sides == "front":
            bnfootprints = [fpc for fpc in bnfootprints if fpc.front_courtyards]
        elif args.sides == "back":
            bnfootprints = [fpc for fpc in bnfootprints if fpc.back_courtyards]

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

    break_tracks(board, bnfootprints, tracks)


if __name__ == "__main__":
    main()
