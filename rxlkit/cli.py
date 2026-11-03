"""Command line: python -m rxlkit <command> ...  (see README.md)"""
import argparse
import csv
import math
import os
import sys

import numpy as np

from . import landxml, surfaces, tin
from .roadxml import Corridor


def _out(path):
    return sys.stdout if path in (None, "-") else open(path, "w", encoding="utf-8")


def cmd_info(a):
    c = Corridor.load(a.rxl)
    o = _out(a.out)
    al = c.alignment
    print(f"corridor: {c.name}", file=o)
    print(f"alignment {al.start:.3f} to {al.end:.3f}; element ends, integrated vs file:", file=o)
    for tag, s0, dn, de in al.closure():
        print(f"  {tag:<11} from {s0:>11.3f}  dN {dn:+.4f}  dE {de:+.4f}", file=o)
    if c.profile:
        print("VPIs:", file=o)
        for s, z, g in c.profile.grades():
            print(f"  {s:>12.4f}  {z:>9.4f}" + ("" if g is None else f"  grade in {100 * g:+.4f}%"), file=o)
    print("templates (offset, elevation of each breakpoint, outward):", file=o)
    for s, L, R in c.templates.positions:
        if L is None:
            print(f"  {s:>11.4f}  (no template)", file=o)
            continue
        z = c.profile.elevation(s)
        side = lambda k: "  ".join(f"{abs(off):.3f}@{pz:.3f}" for off, pz in c.side(s, k))
        print(f"  {s:>11.4f}  CL {z:8.3f} | L {side('L')} | R {side('R')}", file=o)
    if c.steps():
        print("template pairs < 5 cm apart (steps/corners): "
              + ", ".join(f"{x:.4f}/{y:.4f}" for x, y in c.steps()), file=o)


def cmd_points(a):
    """Station/offset of points from a P,N,E[,Z] CSV; with Z, also the difference from the profile."""
    c = Corridor.load(a.rxl)
    o = _out(a.out)
    rows = [r for r in csv.reader(open(a.csv, newline="", encoding="utf-8-sig")) if r and not r[0].startswith("#")]
    print(f"{'point':<10} {'station':>11} {'offset':>9} {'Z':>9} {'profile':>9} {'dZ':>8}", file=o)
    for r in rows:
        try:
            n, e = float(r[1]), float(r[2])
        except ValueError:
            continue  # header
        s, off = c.alignment.station_offset(n, e)
        line = f"{r[0]:<10} {s:>11.3f} {off:>+9.3f}"
        if len(r) > 3 and r[3].strip() and c.profile:
            z, zp = float(r[3]), c.profile.elevation(s)
            line += f" {z:>9.3f} {zp:>9.3f} {z - zp:>+8.3f}"
        print(line, file=o)


def cmd_walls(a):
    c = Corridor.load(a.rxl)
    os.makedirs(a.out_dir, exist_ok=True)
    for side, label in (("L", a.left_label), ("R", a.right_label)):
        name = f"{a.prefix} - {label}" if a.prefix else label
        s = surfaces.wall_surface(c, side, step=a.step, name=name,
                                  desc=f"{label} of {c.name}, toe to top, from RXL templates")
        path = os.path.join(a.out_dir, f"{name}.xml")
        landxml.write_surfaces(path, [s], "rxlkit", f"From {c.name}")
        _report(path, s)


def cmd_channel(a):
    c = Corridor.load(a.rxl)
    name = a.name or os.path.splitext(os.path.basename(a.out))[0]
    s = surfaces.channel_surface(c, step=a.step, name=name, desc=f"Full cross-section of {c.name}, from RXL templates")
    landxml.write_surfaces(a.out, [s], "rxlkit", f"From {c.name}")
    _report(a.out, s)


def _report(path, s):
    q = tin.quality(s.points, s.faces)
    print(f"{path}: {q['points']} points, {q['faces']} faces; " + ("TIN checks pass" if tin.quality_ok(q) else f"TIN CHECKS FAIL {q}"))


def cmd_check(a):
    bad = False
    for s in landxml.read_surfaces(a.xml):
        q = tin.quality(s.points, s.faces)
        ok = tin.quality_ok(q)
        bad |= not ok
        print(f"{s.name}: {q['points']} points, {q['faces']} faces | winding +{q['ccw']}/-{q['cw']} | "
              f"min plan area {q['min_plan_area'] * 1e4:.2f} cm2 | duplicate XY {q['duplicate_xy']} | "
              f"edges on >2 faces {q['nonmanifold_edges']} | {'OK' if ok else 'PROBLEM'}")
    return 1 if bad else 0


def _design_crossing(c, sta, side, dz):
    """Offset (positive) where the template polyline on `side` first rises dz above the CL elevation."""
    z0 = c.profile.elevation(sta)
    prev = (0.0, z0)
    for off, z in c.side(sta, side):
        if z - z0 >= dz:
            (o1, z1), (o2, z2) = prev, (abs(off), z)
            return o1 + (o2 - o1) * (z0 + dz - z1) / (z2 - z1)
        prev = (abs(off), z)
    return None


def cmd_compare(a):
    """Any TIN vs the corridor: CL elevation and the offset where each side reaches given heights above CL."""
    c = Corridor.load(a.rxl)
    s = landxml.read_surface(a.xml, a.surface)
    o = _out(a.out)
    S, offs, Z = tin.raster_station_offset(s.points, s.faces, c.alignment, ds=a.step, do=a.resolution)
    jc = int(np.argmin(np.abs(offs)))
    heights = [float(h) for h in a.heights.split(",")]
    first, last = c.extent
    print(f"surface '{s.name}': {len(s.points)} points, {len(s.faces)} visible faces, Z "
          f"{s.points[:, 2].min():.3f}..{s.points[:, 2].max():.3f}", file=o)
    covered = S[~np.isnan(Z[:, jc])]
    if len(covered):
        print(f"covers the centreline from {covered.min():.1f} to {covered.max():.1f} "
              f"(corridor {first:.2f} to {last:.2f}); raster {a.step} m x {a.resolution} m, so offsets are "
              f"+/- {a.resolution} m", file=o)
    print("offset differences are surface minus design (+ = surface wider); '--' = surface does not reach", file=o)
    print(f"{'station':>9} {'CL dZ':>7} | " + " | ".join(f"h{h:g}: L / R" for h in heights), file=o)

    def crossing(row, sgn, zlim):
        rng = range(jc, len(offs)) if sgn > 0 else range(jc, -1, -1)
        for j in rng:
            if np.isnan(row[j]):
                return None
            if row[j] > zlim:
                return abs(offs[j])
        return None

    fm = lambda x: "   -- " if x is None else f"{x:+6.2f}"
    for i, sta in enumerate(S):
        if not (first <= sta <= last) or np.isnan(Z[i, jc]) or abs(sta / a.every - round(sta / a.every)) > 1e-6:
            continue
        zc = c.profile.elevation(sta)
        cells = []
        for h in heights:
            pair = []
            for sgn, side in ((-1, "L"), (1, "R")):
                d = _design_crossing(c, sta, side, h)
                t = crossing(Z[i], sgn, zc + h)
                pair.append(fm(None if d is None or t is None else t - d))
            cells.append(" / ".join(pair))
        print(f"{sta:>9.2f} {Z[i, jc] - zc:>+7.3f} | " + " | ".join(cells), file=o)


def cmd_verify(a):
    """Exact checks of a TIN built from the corridor: points on every template segment must lie on the surface,
    once. Misses on steep segments are reported as horizontal distance, on flat ones as elevation."""
    c = Corridor.load(a.rxl)
    s = landxml.read_surface(a.xml, a.surface)
    P, F = s.points, s.faces
    first, last = c.extent
    lo, hi = a.start if a.start is not None else first, a.end if a.end is not None else last
    worst, worst_at, miss, overlap, n = 0.0, None, 0, 0, 0
    for sta in np.arange(lo, hi, a.every) + a.every * 0.37:
        if sta > hi:
            break
        z0 = c.profile.elevation(sta)
        for side in ("L", "R"):
            prev = (0.0, z0)
            for off, z in c.side(sta, side):
                for t in (0.05, 0.5, 0.95):
                    po, pz = prev[0] + (off - prev[0]) * t, prev[1] + (z - prev[1]) * t
                    zs = tin.elevations_at(P, F, *c.alignment.point(sta, po))
                    n += 1
                    if len(zs) == 0:
                        miss += 1
                        continue
                    if zs.max() - zs.min() > 0.001:
                        overlap += 1
                    dh, dv = abs(off - prev[0]), abs(z - prev[1])
                    err = abs(zs[0] - pz) * (dh / dv if dv > dh else 1.0)
                    if err > worst:
                        worst, worst_at = err, (round(float(sta), 2), side, round(float(po), 3))
                prev = (off, z)
    print(f"{n} points on template segments, {lo:.2f} to {hi:.2f} every {a.every} m: worst miss "
          f"{worst * 1000:.1f} mm at {worst_at} (station, side, offset); off surface {miss}; overlapping {overlap}")
    return 1 if miss or overlap or worst > a.tolerance else 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="rxlkit", description="Trimble RoadXML corridor tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("info", help="alignment closure, VPIs and templates of an RXL")
    q.add_argument("rxl"); q.add_argument("-o", "--out")
    q.set_defaults(fn=cmd_info)

    q = sub.add_parser("points", help="station/offset (and dZ to profile) of points in a P,N,E[,Z] CSV")
    q.add_argument("rxl"); q.add_argument("csv"); q.add_argument("-o", "--out")
    q.set_defaults(fn=cmd_points)

    q = sub.add_parser("walls", help="separate left and right wall TINs (LandXML)")
    q.add_argument("rxl"); q.add_argument("out_dir")
    q.add_argument("--prefix", default="", help="surface/file name prefix")
    q.add_argument("--left-label", default="LEFT WALL"); q.add_argument("--right-label", default="RIGHT WALL")
    q.add_argument("--step", type=float, default=0.5, help="sample spacing between templates (m)")
    q.set_defaults(fn=cmd_walls)

    q = sub.add_parser("channel", help="one TIN of the whole cross-section (LandXML)")
    q.add_argument("rxl"); q.add_argument("out"); q.add_argument("--name")
    q.add_argument("--step", type=float, default=0.5, help="sample spacing between templates (m)")
    q.set_defaults(fn=cmd_channel)

    q = sub.add_parser("check", help="TIN validity checks for every surface in a LandXML file")
    q.add_argument("xml")
    q.set_defaults(fn=cmd_check)

    q = sub.add_parser("compare", help="compare any TIN with the corridor (coverage, CL, wall offsets)")
    q.add_argument("rxl"); q.add_argument("xml"); q.add_argument("--surface", help="surface name if several")
    q.add_argument("--heights", default="0.1,1,5,10,20", help="heights above CL to compare offsets at")
    q.add_argument("--every", type=float, default=2.0); q.add_argument("--step", type=float, default=0.5)
    q.add_argument("--resolution", type=float, default=0.05, help="offset raster cell (m)")
    q.add_argument("-o", "--out")
    q.set_defaults(fn=cmd_compare)

    q = sub.add_parser("verify", help="exact check that a TIN built from the corridor matches its templates")
    q.add_argument("rxl"); q.add_argument("xml"); q.add_argument("--surface")
    q.add_argument("--every", type=float, default=1.0); q.add_argument("--start", type=float)
    q.add_argument("--end", type=float); q.add_argument("--tolerance", type=float, default=0.025)
    q.set_defaults(fn=cmd_verify)

    a = p.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
