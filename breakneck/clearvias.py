"""Find stitching vias that `clear --vias` removes."""

from collections.abc import Iterable

from breakneck.returnvia import filled_layers, tied_layers
from breakneck.snapshot import Snapshot, Via


def stitching_vias(snapshot: Snapshot, nets: Iterable[str] | None = None) -> list[Via]:
    """Unlocked vias in a fill of their own net that no track or pad ties down.

    `nets` limits the search to those nets.
    A track or a lock is how a hand-placed via tells itself apart.
    """
    wanted = None if nets is None else set(nets)
    return [
        v
        for v in snapshot.vias
        if (wanted is None or v.net in wanted)
        and not v.locked
        and filled_layers(snapshot, v)
        and not tied_layers(snapshot, v)
    ]
