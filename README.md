# Breakneck: Via Stitching and Manual Neckdown for KiCad Layouts

## Introduction

Breakneck works on the board open in KiCad's PCB editor, through the KiCad IPC API. It has three jobs:

- **Via placement.** `returnvia`, `padvia` and `pourstitch` place ground return vias, pad stitching vias and pour stitching vias. Every run is checked with KiCad's own DRC before anything is committed, and lands as one undoable step.
- **Via visualisation.** `gndvia` draws lines from signal vias to their nearest return via when it is too far away, for placing vias by hand.
- **Manual neckdown.** `cut` splits tracks at footprint courtyards so custom rules can neck them down.

### Return vias

A signal changing layers through a via also changes its reference plane, and its return current needs a ground via nearby to follow it. `returnvia` places a ground via within `--distance` (default 1.5 mm) of every via on a non-ground net that has none.

Boards with isolated sections are supported: the ground net is chosen per via from the ground zones on the inner layers, so a via in a GND_ISO section gets a GND_ISO return. A via outside every ground zone is reported, not served.

![Stitching Vias](https://github.com/hatlabs/breakneck/blob/main/stitching-via-screenshot.png?raw=true)

The screenshot shows `gndvia` output: yellow lines from signal vias that lack a ground via within the distance.

### Pad stitching vias

`padvia --nets GND +3.3V` gives every SMD pad of the listed nets a via and a short stub track into the net's plane or pour, unless the pad already reaches a same-net via within 1 mm through its tracks, pads or pour. The via goes on the side of the pad facing away from the footprint centre where there is room.

### Pour stitching vias

With Manhattan routing, outer-layer pours break up into narrow fingers. A regular grid of stitching vias misses many of them, and an unstitched finger connected at one end can act as an antenna.

`pourstitch --nets ... --distance D` makes every point of each listed net's pours lie within D (default 2.5 mm) of a stitching via, measuring distance through the copper rather than in a straight line. A finger whose tip is 2 mm from a via across a gap, but 8 mm away along the copper, counts as 8 mm.

It rasterises the pours at 0.1 mm, places vias on a square lattice of pitch D·√2, then fills the remaining gaps one at a time, starting with the point farthest from any via. Narrow spots get a smaller fallback via. `--render` and `--overlay` show the result as heatmaps (see below).

### Manual Neckdown

**NB:** [This `kicad-python` bug](https://gitlab.com/kicad/code/kicad-python/-/issues/18) prevents manual neckdown from being properly used at the moment.

Neckdown, or its inverse, fanout, refers to narrowing down of PCB tracks and their clearances when routing tracks to
fine-pitch components such as BGAs or QFNs. KiCad does not provide a built-in feature to automatically neckdown tracks,
and while the KiCad Custom Rules are powerful, they do not support neckdowns. It is possible to define rules for
tracks intersecting footprint courtyards, but the rule applies to the entire length of the track segment, not just the
part that intersects the courtyard.

Breakneck is a Python script that communicates with KiCad and cuts tracks at a specified distance from the intersection with a footprint courtyard.
After these track cuts, custom rules apply to the expected track segments. The layout isn't otherwise modified.
Automatic re-healing of broken tracks is prevented by nudging the track widths by one nanometer, making adjacent
tracks different widths.

## Installation

As a prerequisite, [install uv](https://docs.astral.sh/uv/getting-started/installation/).

To run breakneck without actually installing it:

```
uvx breakneck <command> [options...]
```

`returnvia`, `padvia` and `gndvia` run on the base install. `pourstitch` needs the `place` extra (scipy, scikit-image and matplotlib):

```
uvx --from 'breakneck[place]' breakneck <command> [options...]
```

To install breakneck as a tool, with the `place` extra for `pourstitch`:

```
uv tool install 'breakneck[place]'
```

Breakneck needs KiCad 10 and kicad-python 0.8 or later.

## KiCad Configuration

The new KiCad IPC API lets you connect to an existing KiCad instance using Inter-Process Communication (IPC). The API is not enabled by default. To enable it, open the KiCad preferences and navigate to "Plugins" and check the "Enable KiCad API" checkbox.

Only one KiCad instance can serve the API at a time. Close other KiCad windows if breakneck can't find the open board.

## Usage

**NOTE:** If you are running breakneck with `uvx`, prepend all breakneck commands below with `uvx`.

Run `breakneck -h`, or `breakneck <command> -h`, to see the available options.

### Placing vias

```bash
breakneck returnvia
breakneck padvia --nets GND GND_ISO +3.3V +5V_ISO
breakneck pourstitch --nets GND GND_ISO +3.3V +5V_ISO --overlay
```

Each placement run:

1. Refills the zones, so the planner sees current fills.
2. Plans the vias from a snapshot of the board.
3. Checks the plan with `kicad-cli pcb drc` on a temporary copy of the board, against a DRC run on an unmodified copy. Vias implicated in new violations are dropped, and the planner tries again without those spots, for up to five rounds.
4. Commits the remaining vias, their stubs and any markers as one undo step, then refills the zones.

Spots that need attention get markers on User.Eco2:

- an X where no via fits (`returnvia` and `padvia`);
- a + at a forced via (`returnvia --force`);
- a square at a new DRC violation that couldn't be pinned on a single via, such as a pour split into islands by several vias together.

`pourstitch` doesn't mark uncovered spots; its heatmap shows them as areas. `breakneck clear` removes the markers.

`returnvia --force` adds a second pass: every via still without a return gets one at the spot with the most clearance, never overlapping other-net copper. These vias may violate clearance. They survive the DRC check despite that, and get a marker so you can fix them by hand.

Run `pourstitch` last. Vias placed by `returnvia` or `padvia` punch holes in other nets' pours and can create new narrow spots.

Notes:

- **Via sizes** come from each net's netclass. `--via DIAMETER/DRILL` overrides them, and `pourstitch --fallback-via` (default 0.5/0.3) sets the smaller size for narrow spots.
- **Which layers a via must join** is read from the net's zones. At each spot, a via must sit inside the net's fill on every copper layer where one of the net's zone outlines covers the spot, and there must be at least two such layers. A net with zones on only one layer can't be stitched and is skipped.
- **Other nets' pours** are not obstacles. They refill around the new vias, as in any KiCad layout.
- **Project rules are read from disk.** DRC uses the saved `.kicad_pro` and `.kicad_dru`, so save rule and netclass changes before running. The board itself is read from the editor, unsaved changes included.
- **kicad-cli** is taken from `--kicad-cli`, then from the running KiCad, then from `PATH`, then from the standard install locations.
- **Runs add vias.** Every check counts the vias already on the board, so repeating a run adds nothing new. Undo is the way to remove a run's vias: KiCad 10 doesn't allow creating groups through the API, so they can't be grouped for selecting later.

### Dry runs

`breakneck --dry-run <command>` plans, runs DRC, logs what it would place and writes `--render` output, but commits no vias or markers. The zone refill before planning still runs, and `pourstitch --overlay` still places its images.

### Heatmaps

`pourstitch --render PATH` writes a figure with one panel per net and layer, coloured by distance to the nearest stitching via, measured through the copper. Points beyond `--distance` are pink. Copper no via can ever join is grey: a fill island where the net's pour is on only one layer, such as a local zone whose other side is another net's pour. Grey areas are left out of the uncovered area that `pourstitch` logs and are reported separately.

`pourstitch --overlay [DIR]` writes one transparent PNG per copper layer to DIR (default: the current directory) and places it on the board as a reference image, scaled and positioned to line up with the copper. The images go on User.1–User.4, one per copper layer in stack order, or on the layers given with `--overlay-layers`. Those layers must be enabled in Board Setup → Board Editor Layers. Toggle a layer's visibility to show or hide its heatmap, and remove them with `breakneck clear`.

`pourstitch --check` measures the existing stitching only: no planning, no DRC and no vias. Combine it with `--overlay` or `--render` to see the current coverage:

```bash
breakneck pourstitch --nets GND +3.3V --check --overlay
```

### Visualising return vias

`breakneck gndvia` draws lines on User.Eco2 from each via without a return via within `--distance` (default 1.5 mm) to the nearest return via of the ground at that location. It uses the same check as `returnvia`.

It is possible to run `breakneck gndvia` repeatedly using `watch` to provide semi-real time updates while placing the GND vias:

```bash
watch -n 1 breakneck gndvia
```

**NOTE:** It turns out, every execution will bump the undo buffer, so until breakneck is made a bit smarter, it will become very difficult to undo any actual changes.

Breakneck has basic support for filtering the affected tracks and components by layer, netclass or selection.

## Limitations

- Component classes are not supported due to API limitations.
- Grouped tracks and footprints are ignored due to API limitations. It is possible to enter a group and
  run `breakneck --selection` to process the group members.
- Placed vias can't be grouped, because KiCad 10 refuses to create groups through the API.
- The planners use fixed clearances (0.3 mm to other-net copper, plus pad clearance overrides) and don't evaluate custom design rules. The DRC check is what enforces those.
