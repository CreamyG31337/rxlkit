"""Read and write LandXML 1.2 TIN surfaces, alignments, and plan features.

Points are (north, east, z), matching LandXML's "N E Z" order in <P>, <Start>, <End>.
Faces are 0-based index triples in memory and 1-based ids in the file.
"""
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from xml.sax.saxutils import quoteattr

import numpy as np

NS = {"l": "http://www.landxml.org/schema/LandXML-1.2"}


@dataclass
class Surface:
    name: str
    points: np.ndarray  # (n, 3) north, east, z
    faces: np.ndarray  # (m, 3) 0-based indices
    desc: str = ""
    color: str = ""  # "R,G,B" - TBC's display colour (Feature trimbleCADProperties/color); "" = none


@dataclass
class PlanFeature:
    """3D (or 2D) linework feature in LandXML <PlanFeatures> (points are north, east, z)."""
    name: str
    points: np.ndarray  # (n, 3) north, east, z (z may be NaN for 2D)
    layer: str = ""
    color: str = ""  # "R,G,B"
    closed: bool = False
    segments: list | None = None  # optional analytical line/curve segments for <CoordGeom>


def _color(s):
    for p in s.iterfind(".//l:Feature[@code='trimbleCADProperties']/l:Property[@label='color']", NS):
        return p.get("value") or ""
    return ""


def read_surfaces(path, visible_only=True):
    """All TIN surfaces in a LandXML file. Faces flagged i="1" (invisible) are dropped unless visible_only=False."""
    root = ET.parse(path).getroot()
    out = []
    for s in root.iterfind(".//l:Surface", NS):
        pid, pts = {}, []
        for p in s.iterfind(".//l:Pnts/l:P", NS):
            pid[p.get("id")] = len(pts)
            pts.append(tuple(map(float, p.text.split()[:3])))
        faces = [[pid[x] for x in f.text.split()[:3]] for f in s.iterfind(".//l:Faces/l:F", NS)
                 if not (visible_only and f.get("i") == "1")]
        out.append(Surface(s.get("name") or "", np.array(pts, float).reshape(-1, 3),
                           np.array(faces, int).reshape(-1, 3), s.get("desc") or "", _color(s)))
    return out


def read_surface(path, name=None, visible_only=True):
    """One surface: the named one, or the only one in the file."""
    surfs = read_surfaces(path, visible_only)
    if name is not None:
        surfs = [s for s in surfs if s.name == name]
    if len(surfs) != 1:
        raise ValueError(f"{path}: expected one surface{'' if name is None else ' named ' + name}, "
                         f"found {len(surfs)}")
    return surfs[0]


def write_surfaces(path, surfaces, application="rxlkit", app_desc=""):
    """Write surfaces to a LandXML 1.2 file (metric units, CRLF line ends)."""
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2" '
           f'date="{date.today().isoformat()}" time="00:00:00">',
           '  <Units><Metric areaUnit="squareMeter" linearUnit="meter" volumeUnit="cubicMeter" '
           'temperatureUnit="celsius" pressureUnit="milliBars" angularUnit="decimal degrees" '
           'directionUnit="decimal degrees"/></Units>',
           f'  <Application name={quoteattr(application)} desc={quoteattr(app_desc)}/>',
           '  <Surfaces>']
    for s in surfaces:
        out += [f'    <Surface name={quoteattr(s.name)} desc={quoteattr(s.desc)}>',
                '      <Definition surfType="TIN">',
                '        <Pnts>']
        out += [f'          <P id="{i}">{n:.4f} {e:.4f} {z:.4f}</P>' for i, (n, e, z) in enumerate(s.points, 1)]
        out += ['        </Pnts>', '        <Faces>']
        out += [f'          <F>{a + 1} {b + 1} {c + 1}</F>' for a, b, c in s.faces]
        out += ['        </Faces>']
        if s.color:
            out += ['        <Feature code="trimbleCADProperties">',
                    f'          <Property label="color" value={quoteattr(s.color)} />',
                    '        </Feature>']
        out += ['      </Definition>', '    </Surface>']
    out += ['  </Surfaces>', '</LandXML>']
    with open(path, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write("\n".join(out) + "\n")


def _local_tag(el):
    return el.tag.rsplit("}", 1)[-1]


def _child_point(el, tag_name):
    for c in el:
        if _local_tag(c) == tag_name and c.text and c.text.strip():
            vals = [float(x) for x in c.text.strip().split()]
            if len(vals) >= 3:
                return (vals[0], vals[1], vals[2])
            if len(vals) == 2:
                return (vals[0], vals[1], float("nan"))
    return None


class LandXMLAlignment:
    """Horizontal alignment parsed from a LandXML <Alignment><CoordGeom> element.

    Provides the same core query interface as ``rxlkit.roadxml.Alignment``:
    ``at(sta)``, ``point(sta, off)``, ``station_offset(north, east)``, plus ``sample(step)``.
    Conventions: offset + = right of alignment; azimuth in radians clockwise from north.
    """

    def __init__(self, name, elements, step=0.05):
        if not elements:
            raise ValueError(f"alignment {name!r}: no horizontal elements")
        self.name = name
        self.elements = elements
        rows = []
        for el in elements:
            seg_rows = self._integrate(el, step)
            if rows:
                seg_rows = seg_rows[1:]
            rows.extend(seg_rows)
        self.table = np.array(rows, dtype=float)  # sta, n, e, az, curvature
        self._coarse = self.table[::max(1, min(100, len(self.table) // 20 or 1))]

    @staticmethod
    def _integrate(el, step):
        s0, L, k0, k1 = el["s0"], el["L"], el["k0"], el["k1"]
        n, e, az = el["n0"], el["e0"], el["az0"]
        if L <= 0:
            return [(s0, n, e, az, k0)]
        nstep = max(2, int(math.ceil(L / step)))
        h = L / nstep
        rows = [(s0, n, e, az, k0)]
        for i in range(nstep):
            sa = i * h
            sb = (i + 1) * h
            ka = k0 + (k1 - k0) * sa / L
            kb = k0 + (k1 - k0) * sb / L
            azm = az + ka * h / 2.0 + (kb - ka) * h / 6.0
            n += h * math.cos(azm)
            e += h * math.sin(azm)
            az += 0.5 * (ka + kb) * h
            rows.append((s0 + sb, n, e, az, kb))
        if el.get("end") is not None and el["tag"] == "Spiral":
            # Absorb tiny numeric clothoid closure drift linearly across the spiral
            dn = el["end"][0] - rows[-1][1]
            de = el["end"][1] - rows[-1][2]
            if math.hypot(dn, de) < 0.1:
                rows = [(s, rn + ((s - s0) / L) * dn, re + ((s - s0) / L) * de, raz, rk)
                        for s, rn, re, raz, rk in rows]
        return rows

    @property
    def start(self):
        return self.elements[0]["s0"]

    @property
    def end(self):
        return self.elements[-1]["s0"] + self.elements[-1]["L"]

    def at(self, sta):
        """(north, east, azimuth) on the alignment at sta (scalar or array)."""
        sta_arr = np.asarray(sta, dtype=float)
        if np.any(sta_arr < self.start - 1e-6) or np.any(sta_arr > self.end + 1e-6):
            raise ValueError(f"station outside alignment {self.start}..{self.end}")
        T = self.table
        i = np.clip(np.searchsorted(T[:, 0], sta_arr, side="right") - 1, 0, len(T) - 1)
        s, n, e, az, k = T[i].T
        d = sta_arr - s
        out = (n + d * np.cos(az + k * d / 2.0),
               e + d * np.sin(az + k * d / 2.0),
               az + k * d)
        return tuple(float(x) for x in out) if out[0].ndim == 0 else out

    def point(self, sta, off=0.0):
        """(north, east) at station and offset (+ right)."""
        n, e, az = self.at(sta)
        return n - off * np.sin(az), e + off * np.cos(az)

    def station_offset(self, north, east):
        """Station and offset (+ right) of points (scalars or arrays) projected onto the alignment."""
        N = np.atleast_1d(np.asarray(north, float))
        E = np.atleast_1d(np.asarray(east, float))
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

    def sample(self, step=1.0):
        """Sample the alignment at <= ``step`` m spacing; returns ``(stations, norths, easts)``."""
        length = self.end - self.start
        npts = max(2, int(math.ceil(length / step)) + 1)
        stas = np.linspace(self.start, self.end, npts)
        n, e, _ = self.at(stas)
        return stas, n, e


def _parse_radius(val):
    if val is None or str(val).strip().upper() in ("", "INF", "INFINITE"):
        return 0.0
    r = float(val)
    return 0.0 if r == 0 else 1.0 / r


def read_alignments(path, step=0.05):
    """Read all LandXML ``<Alignment>`` elements in ``path`` as ``LandXMLAlignment`` objects.

    Supports ``<Line>``, ``<Curve>`` (``rot="cw"|"ccw"``), and ``<Spiral>`` (clothoid).
    """
    root = ET.parse(path).getroot()
    out = []
    for aln in root.iter():
        if _local_tag(aln) != "Alignment":
            continue
        cg = next((c for c in aln if _local_tag(c) == "CoordGeom"), None)
        if cg is None:
            continue
        name = aln.attrib.get("name") or ""
        sta = float(aln.attrib.get("staStart", 0.0))
        elements = []
        for seg in cg:
            tag = _local_tag(seg)
            p0 = _child_point(seg, "Start")
            p1 = _child_point(seg, "End")
            if p0 is None or p1 is None:
                continue
            if "staStart" in seg.attrib:
                sta = float(seg.attrib["staStart"])
            if tag == "Line":
                dn, de = p1[0] - p0[0], p1[1] - p0[1]
                length = float(seg.attrib.get("length", math.hypot(dn, de)))
                if length <= 1e-12:
                    continue
                az0 = math.atan2(de, dn)
                elements.append(dict(tag="Line", s0=sta, L=length, k0=0.0, k1=0.0,
                                     n0=p0[0], e0=p0[1], az0=az0, end=(p1[0], p1[1])))
                sta += length
            elif tag == "Curve":
                pc = _child_point(seg, "Center")
                rot = (seg.attrib.get("rot") or "cw").lower()
                if pc is None:
                    raise ValueError(f"Curve in alignment {name!r} missing <Center>")
                r = float(seg.attrib.get("radius", math.hypot(p0[0] - pc[0], p0[1] - pc[1])))
                sign = 1.0 if rot == "cw" else -1.0  # + curvature turns right (clockwise azimuth)
                # Radius vector from center to start: (n0-nc, e0-ec)
                rn0, re0 = p0[0] - pc[0], p0[1] - pc[1]
                # Tangent azimuth clockwise from north:
                # For CW (turning right), center is to the right of heading -> heading = radial rotated 90 deg CCW:
                # radial = (rn, re); turning right means heading_N = re, heading_E = -rn? Wait:
                # If center is East (re < 0, i.e. p0 is West of center: rn=0, re=-R), heading North (dN=1, dE=0):
                # az=0 -> (cos az, sin az) = (1, 0) = (-re/R, rn/R) -> atan2(rn, -re) for CW!
                # Check: p0 North of center (rn=+R, re=0), CW moves East (az=pi/2): atan2(+R, 0) = +pi/2. Exact!
                # For CCW (center to left): p0 South of center (rn=-R, re=0), moves East (az=pi/2): atan2(-(-R), 0)=+pi/2.
                az0 = math.atan2(rn0, -re0) if rot == "cw" else math.atan2(-rn0, re0)
                if "length" in seg.attrib:
                    length = float(seg.attrib["length"])
                else:
                    a0 = math.atan2(rn0, re0)
                    a1 = math.atan2(p1[0] - pc[0], p1[1] - pc[1])
                    da = (a0 - a1) if rot == "cw" else (a1 - a0)
                    while da <= 0:
                        da += 2.0 * math.pi
                    length = r * da
                elements.append(dict(tag="Curve", s0=sta, L=length, k0=sign / r, k1=sign / r,
                                     n0=p0[0], e0=p0[1], az0=az0, end=(p1[0], p1[1])))
                sta += length
            elif tag == "Spiral":
                length = float(seg.attrib.get("length", 0.0))
                if length <= 1e-12:
                    continue
                rot = (seg.attrib.get("rot") or "cw").lower()
                sign = 1.0 if rot == "cw" else -1.0
                k0 = sign * _parse_radius(seg.attrib.get("radiusStart"))
                k1 = sign * _parse_radius(seg.attrib.get("radiusEnd"))
                pi = _child_point(seg, "PI")
                if pi is not None and math.hypot(pi[0] - p0[0], pi[1] - p0[1]) > 1e-6:
                    az0 = math.atan2(pi[1] - p0[1], pi[0] - p0[0])
                elif elements:
                    # Continuous tangent from previous element end
                    prev = elements[-1]
                    az0 = prev["az0"] + 0.5 * (prev["k0"] + prev["k1"]) * prev["L"]
                else:
                    az0 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
                elements.append(dict(tag="Spiral", s0=sta, L=length, k0=k0, k1=k1,
                                     n0=p0[0], e0=p0[1], az0=az0, end=(p1[0], p1[1])))
                sta += length
        if elements:
            out.append(LandXMLAlignment(name, elements, step=step))
    return out


def read_alignment(path, name=None, step=0.05):
    """Read one LandXML ``<Alignment>``: the named one, or the only one in the file."""
    alns = read_alignments(path, step=step)
    if name is not None:
        alns = [a for a in alns if a.name == name]
    if len(alns) != 1:
        raise ValueError(f"{path}: expected one alignment{'' if name is None else ' named ' + name}, "
                         f"found {len(alns)}")
    return alns[0]


def _fmt_pt(pt):
    if len(pt) >= 3 and math.isfinite(pt[2]):
        return f"{pt[0]:.5f} {pt[1]:.5f} {pt[2]:.5f}"
    return f"{pt[0]:.5f} {pt[1]:.5f}"


def write_plan_features(path, features, application="rxlkit", app_desc="", project_name=""):
    """Write ``PlanFeature`` linework to a TBC-compatible LandXML 1.2 ``<PlanFeatures>`` file."""
    layers = {}
    for f in features:
        if f.layer and f.layer not in layers:
            layers[f.layer] = f.color or "0,255,0"

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2" '
        f'date="{date.today().isoformat()}" time="00:00:00">',
    ]
    if project_name or layers:
        out.append(f'  <Project name={quoteattr(project_name or "PlanFeatures")} desc="">')
        if layers:
            out.append('    <Feature code="trimbleLayers">')
            for lname, lcol in layers.items():
                out += [
                    '      <Feature code="trimbleLayer">',
                    f'        <Property label="name" value={quoteattr(lname)} />',
                    f'        <Property label="color" value={quoteattr(lcol)} />',
                    '        <Property label="lineStyleName" value="Solid" />',
                    '        <Property label="lineWeight" value="0" />',
                    '      </Feature>',
                ]
            out.append('    </Feature>')
        out.append('  </Project>')
    out += [
        '  <Units><Metric areaUnit="squareMeter" linearUnit="meter" volumeUnit="cubicMeter" '
        'temperatureUnit="celsius" pressureUnit="milliBars" angularUnit="radians" '
        'directionUnit="radians"/></Units>',
        f'  <Application name={quoteattr(application)} desc={quoteattr(app_desc)}/>',
        '  <PlanFeatures>',
    ]
    for f in features:
        out += [f'    <PlanFeature name={quoteattr(f.name)}>', '      <CoordGeom>']
        sta = 0.0
        if f.segments:
            for seg in f.segments:
                kind = seg["kind"].lower()
                p0, p1 = seg["start"], seg["end"]
                if kind == "line":
                    L = float(seg.get("length", math.hypot(p1[0] - p0[0], p1[1] - p0[1])))
                    out += [
                        f'        <Line length="{L:.5f}" staStart="{sta:.5f}">',
                        f'          <Start>{_fmt_pt(p0)}</Start>',
                        f'          <End>{_fmt_pt(p1)}</End>',
                        '        </Line>',
                    ]
                    sta += L
                elif kind == "curve":
                    rot = seg.get("rot", "cw").lower()
                    pc = seg["center"]
                    r = float(seg.get("radius", math.hypot(p0[0] - pc[0], p0[1] - pc[1])))
                    rn0, re0 = p0[0] - pc[0], p0[1] - pc[1]
                    rn1, re1 = p1[0] - pc[0], p1[1] - pc[1]
                    a0 = math.atan2(rn0, re0)
                    a1 = math.atan2(rn1, re1)
                    da = (a0 - a1) if rot == "cw" else (a1 - a0)
                    while da <= 0:
                        da += 2.0 * math.pi
                    L = float(seg.get("length", r * da))
                    az0 = math.atan2(rn0, -re0) if rot == "cw" else math.atan2(-rn0, re0)
                    az1 = math.atan2(rn1, -re1) if rot == "cw" else math.atan2(-rn1, re1)
                    out += [
                        f'        <Curve rot="{rot}" radius="{r:.5f}" length="{L:.5f}" '
                        f'dirStart="{az0:.11f}" dirEnd="{az1:.11f}" staStart="{sta:.5f}">',
                        f'          <Start>{_fmt_pt(p0)}</Start>',
                        f'          <Center>{_fmt_pt(pc)}</Center>',
                        f'          <End>{_fmt_pt(p1)}</End>',
                    ]
                    if seg.get("pi") is not None:
                        out.append(f'          <PI>{_fmt_pt(seg["pi"])}</PI>')
                    out.append('        </Curve>')
                    sta += L
                else:
                    raise ValueError(f"unsupported PlanFeature segment kind: {kind!r}")
        else:
            pts = np.asarray(f.points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] not in (2, 3) or len(pts) < 2:
                raise ValueError(f"PlanFeature {f.name!r}: expected (n>=2, 2|3) points")
            seq = list(pts)
            if f.closed and math.hypot(seq[0][0] - seq[-1][0], seq[0][1] - seq[-1][1]) > 1e-9:
                seq.append(seq[0])
            for p0, p1 in zip(seq[:-1], seq[1:]):
                L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
                out += [
                    f'        <Line length="{L:.5f}" staStart="{sta:.5f}">',
                    f'          <Start>{_fmt_pt(p0)}</Start>',
                    f'          <End>{_fmt_pt(p1)}</End>',
                    '        </Line>',
                ]
                sta += L
        out.append('      </CoordGeom>')
        if f.layer or f.color:
            out.append('      <Feature code="trimbleCADProperties">')
            if f.layer:
                out.append(f'        <Property label="layer" value={quoteattr(f.layer)} />')
            if f.color:
                out.append(f'        <Property label="color" value={quoteattr(f.color)} />')
            out.append('      </Feature>')
        out.append('    </PlanFeature>')
    out += ['  </PlanFeatures>', '</LandXML>']
    with open(path, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write("\n".join(out) + "\n")


def read_plan_features(path, arc_step=0.25):
    """Read LandXML ``<PlanFeature>`` elements into ``list[PlanFeature]``."""
    root = ET.parse(path).getroot()
    out = []
    for pf in root.iter():
        if _local_tag(pf) != "PlanFeature":
            continue
        name = pf.attrib.get("name") or ""
        layer, color = "", ""
        for feat in pf:
            if _local_tag(feat) == "Feature" and feat.attrib.get("code") == "trimbleCADProperties":
                for prop in feat:
                    if _local_tag(prop) == "Property":
                        lbl = prop.attrib.get("label")
                        if lbl == "layer":
                            layer = prop.attrib.get("value") or ""
                        elif lbl == "color":
                            color = prop.attrib.get("value") or ""
        cg = next((c for c in pf if _local_tag(c) == "CoordGeom"), None)
        if cg is None:
            continue
        pts = []
        segs = []
        for seg in cg:
            tag = _local_tag(seg)
            p0 = _child_point(seg, "Start")
            p1 = _child_point(seg, "End")
            if p0 is None or p1 is None:
                continue
            if tag == "Line":
                segs.append({"kind": "line", "start": p0, "end": p1})
                if not pts or math.hypot(pts[-1][0] - p0[0], pts[-1][1] - p0[1]) > 1e-6:
                    pts.append(p0)
                pts.append(p1)
            elif tag == "Curve":
                pc = _child_point(seg, "Center")
                rot = (seg.attrib.get("rot") or "cw").lower()
                pi = _child_point(seg, "PI")
                if pc is None:
                    continue
                r = float(seg.attrib.get("radius", math.hypot(p0[0] - pc[0], p0[1] - pc[1])))
                segs.append({"kind": "curve", "rot": rot, "radius": r, "start": p0,
                             "center": pc, "end": p1, "pi": pi})
                a0 = math.atan2(p0[0] - pc[0], p0[1] - pc[1])
                a1 = math.atan2(p1[0] - pc[0], p1[1] - pc[1])
                da = (a0 - a1) if rot == "cw" else (a1 - a0)
                while da <= 0:
                    da += 2.0 * math.pi
                arc_len = r * da
                nstep = max(2, int(math.ceil(arc_len / arc_step)))
                for k in range(nstep + 1):
                    t = k / nstep
                    ang = a0 - t * da if rot == "cw" else a0 + t * da
                    n = pc[0] + r * math.sin(ang)
                    e = pc[1] + r * math.cos(ang)
                    z = p0[2] + t * (p1[2] - p0[2]) if (math.isfinite(p0[2]) and math.isfinite(p1[2])) else float("nan")
                    if k == 0 and pts and math.hypot(pts[-1][0] - n, pts[-1][1] - e) <= 1e-6:
                        continue
                    pts.append((n, e, z))
        if pts:
            arr = np.array(pts, dtype=float)
            closed = len(arr) >= 3 and math.hypot(arr[0, 0] - arr[-1, 0], arr[0, 1] - arr[-1, 1]) < 1e-6
            out.append(PlanFeature(name=name, points=arr, layer=layer, color=color,
                                   closed=closed, segments=segs))
    return out
