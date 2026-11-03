# rxlkit

Read Trimble RoadXML (`.rxl`) corridors and turn them into LandXML TIN surfaces that pass validity checks. This is useful when software can't read an RXL, or when you need part of a corridor as its own surface, for example one wall for distance checks on drill holes.

rxlkit was written for a rock channel with vertical walls, 3 m invert steps and 90° corners. These are the shapes where naive corridor-to-TIN conversion produces overlapping or folded triangles. It works for any corridor within the [supported subset](#supported-rxl-subset).

## Install

You need Python 3.10 or later and numpy.

```
pip install .                 # from this folder; installs the `rxlkit` command
python -m rxlkit --help       # or run in place without installing (needs numpy)
```

## Commands

| Command | What it does |
|---|---|
| `rxlkit info corridor.rxl` | Shows the alignment, checking each element's integrated end against the file's end coordinate. Then lists the VPIs with grades, then every template position with each breakpoint's offset and elevation. Finally it lists template pairs less than 5 cm apart, which are modelled steps. |
| `rxlkit points corridor.rxl points.csv` | Gives the station and offset of each point in a `P,N,E[,Z]` CSV. If a Z is given, it also gives the difference from the profile. Use it to check work points or stakeout against the alignment. |
| `rxlkit channel corridor.rxl out.xml` | Builds one TIN of the whole cross-section, from the left edge to the right edge. |
| `rxlkit walls corridor.rxl out_dir --prefix NAME` | Builds the left and right walls (toe to top) as two separate TINs. |
| `rxlkit check surface.xml` | Runs the TIN validity checks (below) on every surface in a LandXML file. Exits 1 if any check fails. |
| `rxlkit verify corridor.rxl surface.xml` | Exact check that a TIN built from the corridor matches its templates. It puts points along every template segment every metre, and each point must fall on the surface exactly once. A miss on a steep segment is reported as horizontal distance, and on a flat one as elevation difference. Exits 1 if anything is off the surface, overlapping, or out by more than `--tolerance` (default 25 mm). |
| `rxlkit compare corridor.rxl other.xml` | Compares any surface, such as a designer's or a previous model, with the corridor. It reports the centreline elevation difference and, on each side, how far the surface is from the design at given heights above the centreline (`--heights`). It also shows the stretch of centreline the surface covers. LandXML faces flagged invisible are ignored, as most software does. |

Output files are LandXML 1.2 in metric units. Points are written as `N E Z`, to 4 decimals.

## How steps and vertical walls are modelled

A TIN stores one elevation per plan position, so it can't hold a truly vertical face or an overhang. In the RXL:

- **Vertical walls:** a "vertical" wall is a very steep segment, for example 1000V:1H. A 35 m wall then leans 35 mm.
- **Steps and corners:** an invert step or a square corner is a pair of templates a short distance apart. 10 mm works in Trimble Business Center, but stations under about 1 mm apart appear to be merged, which turns the step into a long taper. For a step in the invert itself, give the profile two VPIs at the same spacing.

When building surfaces, rxlkit treats any template pair less than 5 cm apart as a step:

- **`channel`:** at a step, each of the two columns also gets the other's offsets where that's safe: beyond its own edge, held at the edge elevation, or on its own invert. It never adds them on a steep segment, where 1 mm of offset is metres of elevation. The step and cross faces then sit inside the gap.

  Adjacent columns are joined along the shorter 3D diagonal, so toe stays with toe and top with top. Beyond the edge, a column gets a thin flat ledge at its top elevation.
- **`walls`:** each wall is a strip from toe to top. Where the toe steps outward at a template pair (the channel widening downstream), the strip turns a 90° corner. The cross face leans back into the rock by the wall's own horizontal run, so nothing is vertical.

Both builders keep every triangle's plan winding the same, so no triangle overlaps another in plan.

## TIN checks (`check`)

- **Winding:** every face winds the same way in plan. A face winding the other way is folded over its neighbours.
- **Area:** no face has zero plan area. The smallest is reported.
- **Duplicates:** no two points share the same XY (to 0.1 mm).
- **Edges:** no edge is shared by more than two faces.

These catch the faults that make survey software flag a surface or measure to the wrong face.

## Supported RXL subset

Anything outside this subset raises `NotImplementedError` rather than being misread.

- **Horizontal:** Straight, Arc, EntrySpiral and ExitSpiral elements (clothoid). A spiral may join two arcs.
- **Vertical:** VPIs only, with no vertical curves.
- **Templates:**
  - Segments must be `DistanceAndVerticalDistance`, and `ApplyTemplates` must be `UserDefined` or `None`, with both sides set the same way.
  - Values are interpolated linearly between template stations.
  - Superelevation, widening and rollover are not applied.
- **`walls`:** each side must have exactly an invert segment and one wall segment.

Offsets are positive to the right. Azimuths are measured clockwise from north.

## Python use

```python
from rxlkit import Corridor, channel_surface, write_surfaces, tin

c = Corridor.load("corridor.rxl")
n, e = c.alignment.point(125.0, -4.0)          # station, offset (+ right)
sta, off = c.alignment.station_offset(n, e)
z = c.profile.elevation(125.0)
s = channel_surface(c, name="CHANNEL")
print(tin.quality(s.points, s.faces))
write_surfaces("channel.xml", [s])
```

## Tests

```
python -m unittest discover -s tests
```

The tests build a made-up corridor in [tests/synthetic.py](tests/synthetic.py), so they need no project data. It has a straight, an arc and an exit spiral, an invert-only entrance, a 3 m step-and-widen pair and 1000:1 walls. Its element end coordinates are computed independently of rxlkit.

## License

MIT. See [LICENSE](LICENSE).
