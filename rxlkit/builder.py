"""Build Trimble RoadXML (.rxl) corridor files from PIs, PVIs, and templates."""
import math
from xml.sax.saxutils import escape


def _pi_elements(pis, start_station=0.0):
    """Convert [(north, east, radius_or_None), ...] into Straight and Arc element dicts."""
    if len(pis) < 2:
        raise ValueError("horizontal alignment requires at least 2 PIs")
    pts = [(float(p["north"]), float(p["east"]), p.get("radius")) for p in pis]
    legs = []
    for (n0, e0, _), (n1, e1, _) in zip(pts, pts[1:]):
        dn, de = n1 - n0, e1 - e0
        d = math.hypot(dn, de)
        if d <= 1e-9:
            raise ValueError("consecutive PIs cannot be coincident")
        az = math.atan2(de, dn) % (2.0 * math.pi)
        legs.append((d, az))

    curves = [None] * len(pts)
    for i in range(1, len(pts) - 1):
        r = pts[i][2]
        if r is None or r <= 0:
            continue
        _, az_in = legs[i - 1]
        _, az_out = legs[i]
        delta = (az_out - az_in + math.pi) % (2.0 * math.pi) - math.pi
        theta = abs(delta)
        if theta < 1e-9:
            continue
        t_len = r * math.tan(theta / 2.0)
        arc_len = r * theta
        ni, ei, _ = pts[i]
        pc = (ni - t_len * math.cos(az_in), ei - t_len * math.sin(az_in))
        pt = (ni + t_len * math.cos(az_out), ei + t_len * math.sin(az_out))
        direction = "Right" if delta > 0 else "Left"
        curves[i] = dict(r=float(r), t=t_len, length=arc_len, pc=pc, pt=pt, az_in=az_in, direction=direction)

    elements = []
    sta = float(start_station)
    for i, (d, az) in enumerate(legs):
        c_prev = curves[i]
        c_next = curves[i + 1]
        t_prev = c_prev["t"] if c_prev else 0.0
        t_next = c_next["t"] if c_next else 0.0
        straight_len = d - t_prev - t_next
        if straight_len < -1e-6:
            raise ValueError(f"curve tangents overlap on leg {i + 1} (need {t_prev + t_next:.3f} m, leg is {d:.3f} m)")
        p_start = c_prev["pt"] if c_prev else (pts[i][0], pts[i][1])
        p_end = c_next["pc"] if c_next else (pts[i + 1][0], pts[i + 1][1])
        if straight_len > 1e-6:
            elements.append(dict(
                tag="Straight", start_station=sta, start_az=math.degrees(az),
                start=p_start, end=p_end, length=straight_len,
            ))
            sta += straight_len
        if c_next:
            elements.append(dict(
                tag="Arc", start_station=sta, start_az=math.degrees(c_next["az_in"]),
                start=c_next["pc"], end=c_next["pt"], length=c_next["length"],
                direction=c_next["direction"], radius=c_next["r"],
            ))
            sta += c_next["length"]
    return elements


def build_rxl(spec):
    """Build a Trimble RoadXML 6.4 string from a corridor spec dict."""
    name = spec.get("name", "CORRIDOR")
    horiz = spec["horizontal"]
    start_sta = float(horiz.get("start_station", 0.0))
    elements = _pi_elements(horiz["pis"], start_station=start_sta)

    out = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<TrimbleRoad version="6.4">',
        f'  <Name>{escape(name)}</Name>',
        '  <HorizontalAlignment spiralType="Clothoid">',
        f'    <StartStation>{start_sta:.4f}</StartStation>',
    ]
    for el in elements:
        n0, e0 = el["start"]
        n1, e1 = el["end"]
        out += [
            f'    <{el["tag"]}>',
            f'      <StartStation>{el["start_station"]:.6f}</StartStation>',
            f'      <StartAzimuth>{el["start_az"]:.10f}</StartAzimuth>',
            f'      <StartCoordinate><North>{n0:.6f}</North><East>{e0:.6f}</East></StartCoordinate>',
            f'      <Length>{el["length"]:.6f}</Length>',
        ]
        if el["tag"] == "Arc":
            out.append(f'      <Direction>{el["direction"]}</Direction>')
            out.append(f'      <Radius>{el["radius"]:.6f}</Radius>')
        out += [
            f'      <EndCoordinate><North>{n1:.6f}</North><East>{e1:.6f}</East></EndCoordinate>',
            f'    </{el["tag"]}>',
        ]
    out.append('  </HorizontalAlignment>')

    vpis = spec.get("vertical", {}).get("vpis", [])
    if vpis:
        out.append('  <VerticalAlignment>')
        for v in vpis:
            lvc = float(v.get("length", 0.0) or 0.0)
            out += [
                '    <VerticalPoint>',
                '      <KeyinMethod>VPI</KeyinMethod>',
                f'      <IntersectionPoint><Station>{float(v["station"]):.6f}</Station>'
                f'<Elevation>{float(v["elevation"]):.6f}</Elevation></IntersectionPoint>',
            ]
            if lvc > 0:
                out.append(f'      <SymmetricParabola><Length>{lvc:.6f}</Length></SymmetricParabola>')
            out.append('    </VerticalPoint>')
        out.append('  </VerticalAlignment>')

    tpl = spec.get("templates", {})
    records = tpl.get("records", {})
    for tname, segs in records.items():
        out += ['  <TemplateRecord>', f'    <Name>{escape(tname)}</Name>']
        for seg in segs:
            code = escape(seg.get("code", ""))
            h = float(seg["h"])
            v = float(seg["v"])
            out.append(
                f'    <DistanceAndVerticalDistance><Code>{code}</Code>'
                f'<HorizontalDistance>{h:.6f}</HorizontalDistance>'
                f'<VerticalDistance>{v:.6f}</VerticalDistance></DistanceAndVerticalDistance>'
            )
        out.append('  </TemplateRecord>')

    positions = tpl.get("positions", [])
    if positions:
        out.append('  <TemplatePositioning>')
        for p in positions:
            def _side_xml(val):
                if val is None:
                    return '<Type>None</Type>'
                return f'<Type>UserDefined</Type><Name>{escape(val)}</Name>'
            out.append(
                f'    <ApplyTemplates><Station>{float(p["station"]):.6f}</Station>'
                f'<LeftSide>{_side_xml(p.get("left"))}</LeftSide>'
                f'<RightSide>{_side_xml(p.get("right"))}</RightSide></ApplyTemplates>'
            )
        out.append('  </TemplatePositioning>')

    out.append('</TrimbleRoad>')
    return '\n'.join(out) + '\n'


def write_rxl(path, spec):
    """Write a Trimble RoadXML (.rxl) file from a corridor spec dict."""
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(build_rxl(spec))
