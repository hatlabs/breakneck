# Breakneck: Via Stitching and Manual Neckdown for KiCad Layouts

## Introduction

### Via Stitching

Ground via stitching is a common technique to reduce EMI and improve signal integrity in PCB designs. It involves placing GND vias next to signal vias to provide a return signal
path between reference ground planes and to minimize the loop area. This is important
for any signal with a high edge rate; nowadays basically every digital signal in a design.

Breakneck visualizes stitching via requirements by drawing lines between signal vias and their closest GND vias if the nearest GND via is further away than a specified distance. The user can then manually place the GND vias to meet the requirements. The lines are drawn on the User.Eco2 layer which should not be used for any other purpose (any existing content on this layer will be overwritten).

The screenshot below shows a design with routed signals lacking stitching vias. The yellow lines indicate the signal vias missing a GND via within a distance of 2mm.

![Stitching Vias](https://github.com/hatlabs/breakneck/blob/main/stitching-via-screenshot.png?raw=true)

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

This actually works really fast and the only downside is that you have to type `uvx` every time. If you want to install breakneck, run:

```
uv tool install breakneck
```

## Usage

**NOTE:** If you are running breakneck with `uvx`, prepend all breakneck commands below with `uvx`.

Run `breakneck -h` to see the available options.

`breakneck gndvia` draws stitching via lines on the User.Eco2 layer. The default distance is 2mm, but it can be changed with the `--distance` option.

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
- Multiple ground layers are not supported in `gndvia`. If you have isolated board sections, you will have a lot of bogus lines to the closest regular GND via.
