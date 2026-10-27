# Breakneck: Via Stitching and Manual Neckdown for KiCad Layouts

## Introduction

Breakneck works on the board open in KiCad's PCB editor, through the KiCad IPC API. It has three jobs:

- **Via placement.** `returnvia`, `padvia` and `pourstitch` place ground return vias, pad stitching vias and pour stitching vias. Every run is checked with KiCad's own DRC before anything is committed, and lands as one undoable step.
- **Heatmaps.** `returnvia` and `pourstitch` show return distance and pour coverage as heatmaps, rendered to a file or placed on the board, for checking a layout and placing vias by hand.
- **Manual neckdown.** `cut` splits tracks at footprint courtyards so custom rules can neck them down.

### Return vias

A signal changing layers through a via also changes its reference plane, and its return current needs a ground via nearby to follow it. `returnvia` places a ground via within `--distance` (default 1.5 mm) of every via on a non-ground net that has none.

**Reference planes.** A via end is a copper layer the via connects on: a same-net track, pad or fill touches it there. A via that connects on fewer than two layers carries no layer change and is exempt. Each end's reference plane is the nearest other copper layer in stack order with a ground zone outline at the via, the upper one of two equally near layers. Ends on the same plane are measured once.

**Metric.** The return distance on a plane is the distance from the via's centre to the nearest return's centre through that plane's ground fill: a straight hop across the via's own antipad, then through the fill. Returns are the ground net's vias and through-hole pads with fill within reach of them. A return of GND does not serve a via whose plane is GND_ISO. A via end is served when its distance is at most `--distance` plus a 0.1 mm margin for the 0.1 mm raster. Paths on the raster run in eight directions, which overstates a straight run by up to about 8 % between them, so the effective straight-line limit depends on direction: about 1.46–1.65 mm at the default distance.

The planner still places vias by straight-line distance, while the log and the heatmaps use the distance through the plane. A planned via can therefore leave a via end unserved, for example across a slot in its plane; `returnvia --check` after the run shows those.

Boards with isolated sections are supported: the ground net is chosen per via from the ground zones on the inner layers, so a via in a GND_ISO section gets a GND_ISO return. A via end outside every ground zone is reported, not served. So is an end whose plane has no ground fill within `--distance` of the via.

The ground nets are the nets with zones on two or more inner layers. A board with no such net, such as a two-layer board or a stackup with one inner ground plane, uses the net named GND. Without a GND net, `returnvia` stops before changing anything and asks for `--ground-nets`. Vias on ground nets never need a return.

**Stackup assumption.** The plane next to a signal layer is taken to be a ground pour. In a SIG-GND-PWR-SIG stackup, B.Cu's real reference is the power plane on In2. With no ground zone on In2, B.Cu ends are measured on In1 instead. A board-wide low-priority GND zone under the power plane on In2 makes In2 the reference, but leaves B.Cu ends with no ground fill there, so every B.Cu end fails.

### Pad stitching vias

`padvia` gives every SMD pad with its own net's fill within reach on another copper layer a via and a short stub track into the net's plane or pour, unless the pad already reaches a same-net via within 1 mm through its tracks, pads or pour. The via goes on the side of the pad facing away from the footprint centre where there is room. `--nets GND +3.3V` limits it to those pads of the listed nets. Keep `--nets` when a net meant to join its plane at a single point, such as a star ground, has a zone under its pads; leave that net out of the list.

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

Breakneck's `cut` has basic support for filtering the affected tracks and components by layer, netclass or selection.

## Installation

As a prerequisite, [install uv](https://docs.astral.sh/uv/getting-started/installation/).

To run breakneck without actually installing it:

```
uvx breakneck <command> [options...]
```

`padvia`, `cut` and `clear` run on the base install. `returnvia` and `pourstitch` need the `place` extra (scipy, scikit-image and matplotlib):

```
uvx --from 'breakneck[place]' breakneck <command> [options...]
```

To install breakneck as a tool, with the `place` extra for `returnvia` and `pourstitch`:

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
breakneck padvia
breakneck pourstitch --nets GND GND_ISO +3.3V +5V_ISO --overlay
```

Each placement run:

1. Refills the zones, so the planner sees current fills.
2. Plans the vias from a snapshot of the board.
3. Checks the plan with `kicad-cli pcb drc` on a temporary copy of the board, against a DRC run on an unmodified copy. Vias implicated in new violations are dropped, and the planner tries again without those spots, for up to five rounds.
4. Commits the remaining vias, their stubs and any markers as one undo step, replacing the markers of earlier runs, then refills the zones. If the board was edited in KiCad while the run was going, nothing is committed and the run has to be repeated.

Spots that need attention get markers on User.Eco2:

- an X where no via fits (`returnvia` and `padvia`);
- a + at a forced via (`returnvia --force`);
- a square at a new DRC violation that couldn't be pinned on a single via.

Other nets' pours are not obstacles, so a single via can cut a narrow neck of another net's pour. DRC reports that as unconnected items or isolated copper and names the pour's pads or zones, not the via. breakneck counts these violations per net, in DRC of the board as it is and with the planned vias added. When a net's count goes up, the run commits nothing and exits with an error naming the net and listing the violations. An existing violation that DRC reports at different items after the refill leaves the count unchanged, so it gets a marker but doesn't stop the run. `--allow-splits` (on `returnvia`, `padvia` and `pourstitch`) commits anyway, with a square marker at each violation for fixing by hand. A dry run lists them without failing.

`pourstitch` doesn't mark uncovered spots; its heatmap shows them as areas. Each placing `returnvia`, `padvia` or `pourstitch` run replaces the previous markers, whichever command made them: it removes every shape on User.Eco2, hand-drawn ones included, in the same undo step, also when it places nothing. So after `padvia` then `pourstitch`, only `pourstitch`'s markers remain. `--check` runs leave the markers alone. A dry run removes nothing. `breakneck clear` removes the markers without a run.

`returnvia --force` adds a second pass: every via still without a return gets one at the spot with the most clearance, never overlapping other-net copper. These vias may violate clearance. They survive the DRC check despite that, and get a marker so you can fix them by hand.

Run `pourstitch` last. Vias placed by `returnvia` or `padvia` punch holes in other nets' pours and can create new narrow spots.

Notes:

- **Via sizes** come from each net's netclass. `--via DIAMETER/DRILL` overrides them, and `pourstitch --fallback-via` (default 0.5/0.3) sets the smaller size for narrow spots.
- **Which layers a via must join** is read from the net's zones. At each spot, a via must sit inside the net's fill on every copper layer where one of the net's zone outlines covers the spot, and there must be at least two such layers. A net with zones on only one layer can't be stitched and is skipped.
- **Other nets' pours** are not obstacles. They refill around the new vias, as in any KiCad layout.
- **Project rules are read from disk.** DRC uses the saved `.kicad_pro` and `.kicad_dru`, so save rule and netclass changes before running. The board itself is read from the editor, unsaved changes included.
- **kicad-cli** is taken from `--kicad-cli`, then from the running KiCad, then from `PATH`, then from the standard install locations.
- **Set the zones' "Remove islands" to Never** before running `pourstitch` or `returnvia`. KiCad then keeps unconnected islands in the fill, so breakneck can place vias in them, and a spot over the main pour on one layer with only an island on the other passes the two-layer rule instead of dropping out, and stops showing grey in the `pourstitch` heatmap. Islands kept this way appear as isolated copper in DRC until they are stitched. Stitching only lowers that count, so the split check doesn't block the run.
- **Runs add vias.** Every check counts the vias already on the board, so repeating a run adds nothing new. To remove a run's vias, undo it, or run `breakneck clear --vias` (below) to remove every stitching via not tied to a track or pad. KiCad 10 doesn't allow creating groups through the API, so a run's vias can't be grouped for selecting later.

### Clearing stitching vias

`breakneck clear --vias` removes stitching vias along with the markers and overlays, so `pourstitch` and `returnvia` can place a fresh set after layout changes. A via goes when it touches a fill of its own net on at least one layer and no track or pad of its net touches it. Vias in pads, fanout vias at the end of a track and locked vias stay. `--nets NET [NET ...]` limits the removal to those nets; by default every net with a zone is cleared. The removal is one undo step, followed by a zone refill. With `--dry-run` it logs how many vias per net would go and removes nothing.

**This removes every via touching a pour of its own net that no track or pad ties down, including vias placed by hand.** That includes fanout vias tied to a pad only through a pour, and vias that are the only link between two pours of the same net. To keep such a via, connect it with a short track or lock it. To restore a board after clearing, run `clear --vias`, then `padvia` (it re-adds a via with a stub track for each SMD pad, and the stubs keep those vias through later clears), then `returnvia`, then `pourstitch`, which restitches overlapping pours. As above, `pourstitch` goes last.

### Dry runs

`breakneck --dry-run <command>` plans, runs DRC, logs what it would place and writes `--render` output, but commits no vias or markers. The zone refill before planning still runs, and `--overlay` still places its images. The heatmaps include the planned vias: for `returnvia` they count as returns, except forced ones, so a dry run shows which via ends the run would leave unserved.

### Heatmaps

`pourstitch --render PATH` writes a figure with one panel per net and layer, coloured by distance to the nearest stitching via, measured through the copper. Points beyond `--distance` are pink. Copper that no via could ever bring within `--distance` is grey: it is farther than that, through the net's copper on its layer, from every spot where a via could join the net's fill on two or more layers. Typical grey copper is a local zone on one layer whose other side is another net's pour, or the far end of such a zone joined to the main pour by a long neck. Other-net clearances are ignored for this, so a spot blocked only by other copper shows pink, not grey. Grey areas are left out of the uncovered area that `pourstitch` logs and are reported separately.

`returnvia --render PATH` writes one panel per copper layer on which an unserved via end lies. By default each panel shows only the copper inside the rings around the unserved via ends, which picks out the problem vias. With nothing unserved, the figure is the board outline titled as such, and no overlay is placed. `--full-map` shows the copper of every layer on which a via changing layers connects: every net's tracks, pads and pours, ground pours included, since a via placed anywhere in it would need its own return, and the ground vias. Other vias show only on layers where they have a measured end. Each pixel is coloured by the distance through its reference plane to the nearest ground return. The reference plane is the nearest other copper layer in stack order with a ground zone outline at that pixel, the same rule a via end uses. A via end shows its own measured return distance over the via and over its net's copper in its antipad, the void around it in the reference plane, merged antipads included. Where antipads merge, the copper shows the worst of their distances. Another net's copper over the antipad stays purple: it has no return there. A signal through-hole pad shows the distance a via at its centre would have, over the pad and its clearance hole; pads are not reported unserved. Copper within the served threshold, `--distance` plus the 0.1 mm margin, is dark grey-blue, lightening towards the threshold. Beyond it the colour steps to yellow and runs through matplotlib's `plasma_r` to dark blue at 3 × the threshold, and stays there farther out. Purple copper has no path to a return: it lies over a slot or void in its reference plane. Copper with no ground outline on any other layer has no reference plane and is left blank, such as a two-layer board's ground side. A pink ring of radius `--distance` marks each unserved via end on its own layer. Identical layers share a panel.

`returnvia` logs the number of exempt vias, which connect on fewer than two layers, and how many vias and via-plane returns are unserved. Set `LOGURU_LEVEL=DEBUG` to list each unserved via end with its distance or the reason it can't be served.

`pourstitch --overlay` and `returnvia --overlay` place one transparent image per copper layer on the board as a reference image, scaled and positioned to line up with the copper. The images go on User.1–User.4, one per copper layer in stack order, or on the layers given with `--overlay-layers`. Those layers must be enabled in Board Setup → Board Editor Layers. Each image carries its colourbar and label in a strip below the board outline. The board stores the images itself; the PNGs are written to a temporary directory and removed once placed. When they can't be placed, as in a dry run with the layers disabled, they stay there and the log lists their paths and placements. Toggle a layer's visibility to show or hide its heatmap, and remove them with `breakneck clear`. Both `clear` and a new `--overlay` replace only breakneck's own overlays, recognised by a `breakneck` text chunk in the PNG, also after the board outline has changed. Other reference images on those layers stay. Overlays placed by earlier breakneck versions carry no tag and are recognised only at their position and scale for the current board outline; after an outline change, delete those by hand.

`pourstitch --check` and `returnvia --check` measure the existing vias only: no planning, no DRC and no vias. Combine them with `--render` or `--overlay` to see the current coverage. Every `--overlay` run commits its images and so adds an undo step. For repeated checks while placing vias by hand, use `--check --render` and keep the PNG open in an image viewer that reloads it; use `--overlay` for a one-off look on the board:

```bash
breakneck returnvia --check --render returnvia.png
breakneck pourstitch --nets GND +3.3V --check --overlay
```

## Limitations

- Component classes are not supported due to API limitations.
- Grouped tracks and footprints are ignored due to API limitations. It is possible to enter a group and
  run `breakneck --selection` to process the group members.
- Placed vias can't be grouped, because KiCad 10 refuses to create groups through the API.
- The planners use fixed clearances (0.3 mm to other-net copper; a pad's clearance override plus 0.1 mm where it has one) and don't evaluate custom design rules. The DRC check is what enforces those.
