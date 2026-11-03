"""Read and write LandXML 1.2 TIN surfaces.

Points are (north, east, z), matching LandXML's "N E Z" order in <P>. Faces are 0-based index triples in
memory and 1-based ids in the file.
"""
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
                           np.array(faces, int).reshape(-1, 3), s.get("desc") or ""))
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
        out += ['        </Faces>', '      </Definition>', '    </Surface>']
    out += ['  </Surfaces>', '</LandXML>']
    with open(path, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write("\n".join(out) + "\n")
