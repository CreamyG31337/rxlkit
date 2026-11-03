"""TIN checks and queries. Points are (north, east, z); faces are 0-based index triples."""
from collections import Counter

import numpy as np


def plan_areas(P, F):
    """Signed plan area of each face; + = counter-clockwise with east as x and north as y."""
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    return 0.5 * ((b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]) - (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]))


def quality(P, F, xy_decimals=4):
    """Checks a 2.5D TIN must pass: one plan winding (no folded faces), no zero-area faces, no two points at
    the same XY, and no edge shared by more than two faces."""
    area = plan_areas(P, F)
    edges = Counter()
    for t in F:
        for i, j in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edges[(min(i, j), max(i, j))] += 1
    xy = {(round(n, xy_decimals), round(e, xy_decimals)) for n, e in P[:, :2]}
    return dict(points=len(P), faces=len(F), ccw=int((area > 0).sum()), cw=int((area < 0).sum()),
                min_plan_area=float(np.abs(area).min()) if len(F) else 0.0,
                duplicate_xy=len(P) - len(xy), nonmanifold_edges=sum(1 for v in edges.values() if v > 2))


def quality_ok(q, min_area=1e-6):
    return (q["ccw"] == 0 or q["cw"] == 0) and q["min_plan_area"] > min_area and q["duplicate_xy"] == 0 \
        and q["nonmanifold_edges"] == 0


def elevations_at(P, F, north, east):
    """Elevations of every face containing (north, east) in plan; more than one distinct value means overlap."""
    A, B, C = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    d = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    with np.errstate(divide="ignore", invalid="ignore"):
        u = ((north - A[:, 0]) * (C[:, 1] - A[:, 1]) - (east - A[:, 1]) * (C[:, 0] - A[:, 0])) / d
        v = ((B[:, 0] - A[:, 0]) * (east - A[:, 1]) - (B[:, 1] - A[:, 1]) * (north - A[:, 0])) / d
    m = (d != 0) & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
    return A[m, 2] + u[m] * (B[m, 2] - A[m, 2]) + v[m] * (C[m, 2] - A[m, 2])


def _closest_on_triangle(p, a, b, c):
    # Ericson, Real-Time Collision Detection, 5.1.5
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0 and d2 <= 0:
        return a
    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + ab * (d1 / (d1 - d3))
    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + ac * (d2 / (d2 - d6))
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)))
    den = 1 / (va + vb + vc)
    return a + ab * (vb * den) + ac * (vc * den)


def distance(P, F, point, candidates=200):
    """3D distance from point (north, east, z) to the nearest face. Faces are pre-filtered by plan distance of
    their centroids, so pass more candidates for long, thin faces far from the point."""
    p = np.asarray(point, float)
    cent = P[F].mean(axis=1)
    idx = np.argsort(((cent[:, :2] - p[:2]) ** 2).sum(1))[:candidates]
    return min(float(np.linalg.norm(p - _closest_on_triangle(p, *P[F[i]]))) for i in idx)


def raster_station_offset(P, F, alignment, ds=0.5, do=0.05, off_range=(-60.0, 60.0)):
    """Rasterise a TIN in station/offset space; where faces overlap in plan the highest wins.
    Returns (stations, offsets, Z) with Z[i, j] NaN where there is no surface. Features narrower than `do` in
    plan (e.g. near-vertical walls) can fall between cells; use elevations_at for exact queries."""
    S, O = alignment.station_offset(P[:, 0], P[:, 1])
    s_lo = np.floor(S.min())
    ns = int((S.max() - s_lo) / ds) + 2
    no = int((off_range[1] - off_range[0]) / do) + 1
    Z = np.full((ns, no), np.nan)
    for a, b, c in F:
        xs, ys, zs = S[[a, b, c]], O[[a, b, c]], P[[a, b, c], 2]
        i0, i1 = int(np.ceil((xs.min() - s_lo) / ds)), int(np.floor((xs.max() - s_lo) / ds))
        j0, j1 = int(np.ceil((ys.min() - off_range[0]) / do)), int(np.floor((ys.max() - off_range[0]) / do))
        i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1, ns - 1), min(j1, no - 1)
        if i1 < i0 or j1 < j0:
            continue
        d = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(d) < 1e-12:
            continue
        gi, gj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        px, py = s_lo + gi * ds, off_range[0] + gj * do
        u = ((px - xs[0]) * (ys[2] - ys[0]) - (py - ys[0]) * (xs[2] - xs[0])) / d
        v = ((xs[1] - xs[0]) * (py - ys[0]) - (ys[1] - ys[0]) * (px - xs[0])) / d
        m = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9)
        z = zs[0] + u * (zs[1] - zs[0]) + v * (zs[2] - zs[0])
        sub = Z[i0:i1 + 1, j0:j1 + 1]
        sub[m] = np.fmax(sub[m], z[m])
    return s_lo + np.arange(ns) * ds, off_range[0] + np.arange(no) * do, Z
