"""Build LandXML-ready TIN surfaces from a corridor.

A 2.5D TIN cannot hold a truly vertical face, so steps and corners are modelled inside the gap between two
template stations a short distance apart (e.g. 10 mm). Both builders keep every face's plan winding the same,
so no triangle overlaps another in plan.
"""
import math

import numpy as np

from .landxml import Surface


def _side_sign(side):
    if side not in ("L", "R"):
        raise ValueError("side must be 'L' or 'R'")
    return -1 if side == "L" else 1


def _grid(tpl, step, min_gap=0.0):
    out = set(tpl)
    s = math.ceil(tpl[0] / step) * step
    while s < tpl[-1]:
        if min_gap <= 0 or min(abs(s - t) for t in tpl) > min_gap:
            out.add(round(s, 6))
        s += step
    return sorted(out)


def wall_surface(corridor, side, step=0.5, corner_gap=0.05, name="", desc=""):
    """One side's wall (toe to top) as a triangle strip.

    The side's template must have exactly two segments where it has a wall (invert, then wall); stations with
    only the invert segment are left out and must be at the ends of the run. A template pair closer than
    `corner_gap` where the toe steps outward downstream by more than 0.1 is built as a 90 degree corner: the
    cross face leans back into rock (upstream) by the wall's horizontal run, so nothing is vertical.
    """
    sgn = _side_sign(side)
    k = 1 if side == "L" else 2
    applied = corridor.templates.applied
    has_wall = [len(p[k]) == 2 for p in applied]
    if any(len(p[k]) > 2 for p in applied):
        raise NotImplementedError("wall_surface handles invert + one wall segment only")
    idx = [i for i, w in enumerate(has_wall) if w]
    if not idx:
        raise ValueError(f"no wall on side {side}")
    if idx != list(range(idx[0], idx[-1] + 1)):
        raise ValueError(f"side {side} wall is not continuous")
    tpl = [applied[i][0] for i in idx]
    stations = _grid(tpl, step)

    def vals(sta):
        (to, tz), (po, pz) = corridor.side(sta, side)
        return abs(to), abs(po) - abs(to), pz - tz, tz  # toe width, wall H, wall V, toe z

    corners = []
    for a, b in zip(tpl, tpl[1:]):
        wa, wb = vals(a)[0], vals(b)[0]
        if b - a < corner_gap and abs(wb - wa) > 0.1:
            if wb < wa:
                raise NotImplementedError(f"inward toe step at {a}/{b} (only widening downstream is handled)")
            corners.append((a, b))
    skip = {s for a, b in corners for s in (a, b)}

    pts, faces = [], []

    def add(sta, off, z):
        pts.append(corridor.xyz(sta, off, z))
        return len(pts) - 1

    prev = None  # (toe, top) of the previous column
    for sta in stations:
        corner = next(((a, b) for a, b in corners if sta == a), None)
        if sta in skip and corner is None:
            continue  # second station of a corner pair: built with the first
        if corner is None:
            w, h, v, z = vals(sta)
            col = (add(sta, sgn * w, z), add(sta, sgn * (w + h), z + v))
            if prev:
                faces += [(prev[0], col[0], prev[1]), (prev[1], col[0], col[1])]
            prev = col
            continue
        sa, sb = corner
        wa, ha, va, za = vals(sa)
        wb, hb, vb, zb = vals(sb)
        lean = max(ha, hb)
        p1 = add(sa, sgn * wa, za)                      # upstream wall toe end
        q1 = add(sb - lean, sgn * (wa + ha), za + va)   # top corner: upstream wall / cross face
        faces += [(prev[0], p1, prev[1]), (prev[1], p1, q1)]
        p2 = add(sb, sgn * wa, zb)                      # cross face bottom, inner end
        p3 = add(sb, sgn * wb, zb)                      # cross face bottom, outer end = downstream toe
        q2 = add(sb - lean, sgn * (wb + hb), zb + vb)   # top corner: cross face / downstream wall
        faces += [(p1, p2, q1), (p2, p3, q2), (p2, q2, q1)]
        prev = (p3, q2)
    return Surface(name, np.array(pts), np.array(faces, int), desc)


def _at_offset(col, o):
    """Elevation on a column [(offset, z)] at offset o; beyond the column's edge, the edge elevation."""
    if o <= col[0][0]:
        return col[0][1]
    if o >= col[-1][0]:
        return col[-1][1]
    for (oa, za), (ob, zb) in zip(col, col[1:]):
        if oa <= o <= ob:
            return za if ob == oa else za + (zb - za) * (o - oa) / (ob - oa)


def _extend(col, others, invert_z):
    """Add others' offsets to col where safe: beyond col's edges (at edge elevation) or on its invert (points at
    invert_z). Never on a steep segment, where a mm of offset is metres of elevation; skip any within 1 cm of
    col's own points."""
    flat = [o for o, z in col if abs(z - invert_z) < 1e-6]
    lo, hi = (min(flat), max(flat)) if flat else (0.0, 0.0)
    add = []
    for o in others:
        if min(abs(o - p) for p, _ in col) <= 0.01:
            continue
        if o < col[0][0] or o > col[-1][0] or lo < o < hi:
            add.append((o, _at_offset(col, o)))
    return sorted(col + add)


def channel_surface(corridor, step=0.5, step_gap=0.05, name="", desc=""):
    """The whole corridor cross-section (left edge to right edge) as one TIN.

    Each station is a column of template points, monotone in offset. Adjacent columns are zipped by taking the
    shorter 3D diagonal, which keeps toe with toe and top with top. At a template pair closer than `step_gap`
    (a step or corner) each column also takes the other's offsets where safe, so the near-vertical step and
    cross faces lie inside the gap. Continuous templates < 5 mm apart are merged.
    """
    tpl = sorted(p[0] for p in corridor.templates.applied)
    stations = _grid(tpl, step, min_gap=step_gap)
    close = [(a, b) for a, b in zip(stations, stations[1:]) if b - a < 0.005]
    for a, b in close:
        ca, cb = corridor.column(a), corridor.column(b)
        if len(ca) != len(cb) or any(abs(p[0] - q[0]) > 0.01 or abs(p[1] - q[1]) > 0.01 for p, q in zip(ca, cb)):
            raise ValueError(f"templates at {a} and {b} differ but are < 5 mm apart")
    stations = [s for s in stations if s not in {a for a, _ in close}]

    cols = {s: corridor.column(s) for s in stations}
    for a, b in [(a, b) for a, b in zip(stations, stations[1:]) if b - a < step_gap]:
        ca, cb = cols[a], cols[b]
        cols[a] = _extend(ca, [o for o, _ in cb], corridor.profile.elevation(a))
        cols[b] = _extend(cb, [o for o, _ in ca], corridor.profile.elevation(b))

    pts, ids = [], {}
    for s in stations:
        ids[s] = []
        for o, z in cols[s]:
            pts.append(corridor.xyz(s, o, z))
            ids[s].append(len(pts) - 1)
    faces = []

    def tri(a, b, c):  # counter-clockwise in plan (east = x, north = y)
        (na, ea, _), (nb, eb, _), (nc, ec, _) = pts[a], pts[b], pts[c]
        faces.append((a, b, c) if (eb - ea) * (nc - na) - (nb - na) * (ec - ea) > 0 else (a, c, b))

    d3 = lambda p, q: math.dist(pts[p], pts[q])
    for sa, sb in zip(stations, stations[1:]):
        ia, ib = ids[sa], ids[sb]
        i = j = 0
        while i < len(ia) - 1 or j < len(ib) - 1:
            if j == len(ib) - 1 or (i < len(ia) - 1 and d3(ia[i + 1], ib[j]) <= d3(ia[i], ib[j + 1])):
                tri(ia[i], ib[j], ia[i + 1]); i += 1
            else:
                tri(ia[i], ib[j], ib[j + 1]); j += 1
    return Surface(name, np.array(pts), np.array(faces, int), desc)
