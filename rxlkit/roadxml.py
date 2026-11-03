"""Read Trimble RoadXML (.rxl) corridors: horizontal alignment, VPI profile and templates.

Supported subset (anything else raises NotImplementedError rather than being silently misread):
- Horizontal: Straight, Arc, EntrySpiral, ExitSpiral (clothoid; a spiral may join two arcs).
- Vertical: VPIs without vertical curves (grade breaks only).
- Templates: DistanceAndVerticalDistance segments; ApplyTemplates of Type UserDefined or None.
  Superelevation, widening and rollover are not applied.

Conventions: stations and offsets in file units; offset + = right of the alignment; azimuth in radians,
clockwise from north; points are (north, east).
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

H_ELEMENTS = ("Straight", "Arc", "EntrySpiral", "ExitSpiral")


def _f(el, tag):
    t = el.findtext(tag)
    return None if t is None or not t.strip() else float(t)


class Alignment:
    """Horizontal alignment, integrated from each element's start coordinate into a dense table."""

    def __init__(self, ha, step=0.01):
        self.elements = []
        for el in ha:
            if el.tag in ("StartStation", "EndStation", "StartPoint"):
                continue
            if el.tag not in H_ELEMENTS:
                raise NotImplementedError(f"horizontal element {el.tag}")
            L = _f(el, "Length")
            sign = 1.0 if (el.findtext("Direction") or "Right") == "Right" else -1.0
            if el.tag == "Straight":
                k0 = k1 = 0.0
            elif el.tag == "Arc":
                k0 = k1 = sign / _f(el, "Radius")
            else:  # clothoid: curvature linear in length; a missing radius is infinite
                r0, r1 = _f(el, "StartRadius"), _f(el, "EndRadius")
                k0 = sign / r0 if r0 else 0.0
                k1 = sign / r1 if r1 else 0.0
            sc, ec = el.find("StartCoordinate"), el.find("EndCoordinate")
            self.elements.append(dict(
                tag=el.tag, s0=_f(el, "StartStation"), L=L, k0=k0, k1=k1,
                n0=_f(sc, "North"), e0=_f(sc, "East"), az0=math.radians(_f(el, "StartAzimuth")),
                end=None if ec is None else (_f(ec, "North"), _f(ec, "East"))))
        if not self.elements:
            raise ValueError("no horizontal elements")

        rows = []
        for el in self.elements:
            rows += self._integrate(el, step)
        self.table = np.array(rows)  # sta, n, e, az, curvature
        self._coarse = self.table[::100]

    @staticmethod
    def _integrate(el, step):
        s0, L, k0, k1 = el["s0"], el["L"], el["k0"], el["k1"]
        k = lambda s: k0 + (k1 - k0) * s / L
        n, e, az = el["n0"], el["e0"], el["az0"]
        rows, s = [], 0.0
        while True:
            rows.append((s0 + s, n, e, az, k(s)))
            if s >= L - 1e-9:
                return rows
            h = min(step, L - s)
            ka, kb = k(s), k(s + h)
            azm = az + ka * h / 2 + (kb - ka) * h / 6  # heading at mid-chord for linearly varying curvature
            n += h * math.cos(azm)
            e += h * math.sin(azm)
            az += (ka + kb) / 2 * h
            s += h

    @property
    def start(self):
        return self.elements[0]["s0"]

    @property
    def end(self):
        return self.elements[-1]["s0"] + self.elements[-1]["L"]

    def closure(self):
        """[(tag, start station, dN, dE)]: integrated element end vs the file's EndCoordinate."""
        out = []
        for el in self.elements:
            if el["end"] is None:
                continue
            n, e = self._integrate(el, 0.005)[-1][1:3]
            out.append((el["tag"], el["s0"], n - el["end"][0], e - el["end"][1]))
        return out

    def element_at(self, sta):
        for el in self.elements:
            if el["s0"] - 1e-9 <= sta <= el["s0"] + el["L"] + 1e-9:
                return el
        raise ValueError(f"station {sta} outside alignment {self.start}..{self.end}")

    def at(self, sta):
        """(north, east, azimuth) on the alignment at sta (scalar or array)."""
        sta = np.asarray(sta, dtype=float)
        if np.any(sta < self.start - 1e-6) or np.any(sta > self.end + 1e-6):
            raise ValueError(f"station outside alignment {self.start}..{self.end}")
        T = self.table
        i = np.clip(np.searchsorted(T[:, 0], sta, side="right") - 1, 0, len(T) - 1)
        s, n, e, az, k = T[i].T
        d = sta - s  # < one table step: advance along a circular arc of the row's curvature
        out = n + d * np.cos(az + k * d / 2), e + d * np.sin(az + k * d / 2), az + k * d
        return tuple(float(x) for x in out) if out[0].ndim == 0 else out

    def point(self, sta, off):
        """(north, east) at station and offset (+ right)."""
        n, e, az = self.at(sta)
        return n - off * np.sin(az), e + off * np.cos(az)

    def station_offset(self, north, east):
        """Station and offset (+ right) of points (scalars or arrays) by projection onto the alignment.
        Nearest-point projection; ambiguous inside a curve's centre at offsets near the radius."""
        N, E = np.atleast_1d(np.asarray(north, float)), np.atleast_1d(np.asarray(east, float))
        C = self._coarse
        sta = np.empty(len(N))
        for a in range(0, len(N), 256):
            dn = N[a:a + 256, None] - C[None, :, 1]
            de = E[a:a + 256, None] - C[None, :, 2]
            sta[a:a + 256] = C[np.argmin(dn * dn + de * de, axis=1), 0]
        for _ in range(6):
            n, e, az = self.at(np.clip(sta, self.start, self.end))
            sta = np.clip(sta + (N - n) * np.cos(az) + (E - e) * np.sin(az), self.start, self.end)
        n, e, az = self.at(sta)
        off = -(N - n) * np.sin(az) + (E - e) * np.cos(az)
        if np.ndim(north) == 0:
            return float(sta[0]), float(off[0])
        return sta, off


class Profile:
    """Vertical alignment from VPIs (no vertical curves)."""

    ALLOWED = {"KeyinMethod", "IntersectionPoint", "GradeIn", "GradeOut"}

    def __init__(self, va):
        self.vpis = []
        for v in va.findall("VerticalPoint"):
            extra = {c.tag for c in v} - self.ALLOWED
            if extra:
                raise NotImplementedError(f"vertical point with {sorted(extra)} (vertical curves not supported)")
            self.vpis.append((_f(v, "IntersectionPoint/Station"), _f(v, "IntersectionPoint/Elevation")))
        if any(b[0] <= a[0] for a, b in zip(self.vpis, self.vpis[1:])):
            raise ValueError("VPI stations must increase")
        self._s = np.array([s for s, _ in self.vpis])
        self._z = np.array([z for _, z in self.vpis])

    def elevation(self, sta):
        s = np.asarray(sta, float)
        if np.any(s < self._s[0] - 1e-6) or np.any(s > self._s[-1] + 1e-6):
            raise ValueError(f"station outside profile {self._s[0]}..{self._s[-1]}")
        z = np.interp(s, self._s, self._z)
        return float(z) if z.ndim == 0 else z

    def grades(self):
        """[(station, elevation, grade in from previous VPI as a fraction or None)]."""
        out = []
        for i, (s, z) in enumerate(self.vpis):
            g = None if i == 0 else (z - self.vpis[i - 1][1]) / (s - self.vpis[i - 1][0])
            out.append((s, z, g))
        return out


class Templates:
    """Template records and their positioning. A side's segments are [(code, horizontal, vertical)], outward."""

    def __init__(self, root):
        self.records = {}
        for t in root.findall("TemplateRecord"):
            segs = []
            for d in t:
                if d.tag == "Name":
                    continue
                if d.tag != "DistanceAndVerticalDistance":
                    raise NotImplementedError(f"template segment {d.tag} in {t.findtext('Name')}")
                segs.append((d.findtext("Code") or "", _f(d, "HorizontalDistance"), _f(d, "VerticalDistance")))
            self.records[t.findtext("Name")] = segs
        self.positions = []  # (station, left segs or None, right segs or None)
        tp = root.find("TemplatePositioning")
        for a in ([] if tp is None else tp.findall("ApplyTemplates")):
            sides = []
            for tag in ("LeftSide", "RightSide"):
                s = a.find(tag)
                typ = s.findtext("Type")
                if typ == "None":
                    sides.append(None)
                elif typ == "UserDefined":
                    sides.append(self.records[s.findtext("Name")])
                else:
                    raise NotImplementedError(f"ApplyTemplates type {typ}")
            if (sides[0] is None) != (sides[1] is None):
                raise NotImplementedError("template on one side only")
            self.positions.append((_f(a, "Station"), sides[0], sides[1]))
        if any(b[0] <= a[0] for a, b in zip(self.positions, self.positions[1:])):
            raise ValueError("template stations must increase")

    @property
    def applied(self):
        """Positions where templates are applied (both sides)."""
        return [p for p in self.positions if p[1] is not None]

    def at(self, sta):
        """(left, right) segment lists at sta, interpolated linearly between the bracketing template stations.
        Between a None position and a template there is no corridor (ValueError)."""
        P = self.positions
        for (s0, l0, r0), (s1, l1, r1) in zip(P, P[1:]):
            if s0 <= sta <= s1:
                if sta == s0 and l0 is not None:
                    return l0, r0
                if sta == s1 and l1 is not None:
                    return l1, r1
                if l0 is None or l1 is None:
                    break
                if len(l0) != len(l1) or len(r0) != len(r1):
                    raise ValueError(f"segment count changes between templates at {s0} and {s1}")
                t = (sta - s0) / (s1 - s0)
                mix = lambda A, B: [(a[0], a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t) for a, b in zip(A, B)]
                return mix(l0, l1), mix(r0, r1)
        raise ValueError(f"no corridor at station {sta}")


class Corridor:
    """An RXL corridor: alignment + profile + templates."""

    def __init__(self, root):
        self.root = root
        self.name = root.findtext("Name")
        self.alignment = Alignment(root.find("HorizontalAlignment"))
        va = root.find("VerticalAlignment")
        self.profile = None if va is None else Profile(va)
        self.templates = Templates(root)

    @classmethod
    def load(cls, path):
        return cls(ET.parse(path).getroot())

    def side(self, sta, side):
        """[(offset, z)] outward from the alignment on side 'L' or 'R' (offset signed, + right), excluding CL."""
        z = self.profile.elevation(sta)
        L, R = self.templates.at(sta)
        sgn, segs = (-1, L) if side == "L" else (1, R)
        o, pts = 0.0, []
        for _, h, v in segs:
            o += h
            z += v
            pts.append((sgn * o, z))
        return pts

    def column(self, sta):
        """[(offset, z)] across the corridor, left to right."""
        return self.side(sta, "L")[::-1] + self.side(sta, "R")

    def xyz(self, sta, off, z):
        n, e = self.alignment.point(sta, off)
        return float(n), float(e), float(z)

    @property
    def extent(self):
        """(first, last) station with templates applied."""
        a = self.templates.applied
        return a[0][0], a[-1][0]

    def steps(self, max_gap=0.05):
        """Pairs of consecutive applied template stations closer than max_gap (modelled steps/corners)."""
        tpl = sorted(p[0] for p in self.templates.applied)
        return [(a, b) for a, b in zip(tpl, tpl[1:]) if b - a < max_gap]
