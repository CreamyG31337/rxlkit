"""A small made-up corridor for tests: straight, arc, exit spiral; an invert-only entrance, walls with a 10 mm
step-down-and-widen pair, and a 1000:1 'vertical' wall. Element end coordinates are computed independently of
rxlkit (closed form, or Simpson's rule for the spiral)."""
import math

N0, E0, AZ0 = 5000000.0, 300000.0, 45.0
STRAIGHT, R, SPIRAL = 50.0, 100.0, 30.0
TOP = 135.0
# (station, half width, has wall); profile steps from 100 to 97 at 40
TEMPLATES = [(0.0, 10.0, False), (9.99, 10.0, False), (10.0, 5.0, True), (39.99, 5.0, True), (40.0, 6.0, True),
             (100.0, 6.0, True)]
VPIS = [(0.0, 100.0), (39.99, 100.0), (40.0, 97.0), (110.0, 96.3)]


def _spiral_end(n, e, az, k0, L, steps=4000):
    """Exit spiral (curvature k0 -> 0) end point by Simpson's rule on the heading."""
    head = lambda s: az + k0 * (s - s * s / (2 * L))
    h = L / steps
    sc = sum((1 if i in (0, steps) else 4 if i % 2 else 2) * math.cos(head(i * h)) for i in range(steps + 1))
    ss = sum((1 if i in (0, steps) else 4 if i % 2 else 2) * math.sin(head(i * h)) for i in range(steps + 1))
    return n + sc * h / 3, e + ss * h / 3, head(L)


def geometry():
    az = math.radians(AZ0)
    n1, e1 = N0 + STRAIGHT * math.cos(az), E0 + STRAIGHT * math.sin(az)
    L_arc = 30.0
    d = L_arc / R  # right turn
    az2 = az + d
    # chord of an arc: length 2R sin(d/2) along the mean heading
    c = 2 * R * math.sin(d / 2)
    n2, e2 = n1 + c * math.cos(az + d / 2), e1 + c * math.sin(az + d / 2)
    n3, e3, az3 = _spiral_end(n2, e2, az2, 1 / R, SPIRAL)
    return [("Straight", 0.0, STRAIGHT, (N0, E0, az), (n1, e1, az), None),
            ("Arc", STRAIGHT, L_arc, (n1, e1, az), (n2, e2, az2), R),
            ("ExitSpiral", STRAIGHT + L_arc, SPIRAL, (n2, e2, az2), (n3, e3, az3), R)]


def rxl_text(vertical_curve=False):
    out = ['<?xml version="1.0" encoding="utf-8"?>', '<TrimbleRoad version="6.4">', '  <Name>TEST CHANNEL</Name>',
           '  <HorizontalAlignment spiralType="Clothoid">', '    <StartStation>0</StartStation>']
    for tag, s0, L, (na, ea, aa), (nb, eb, ab), r in geometry():
        out += [f'    <{tag}>', f'      <StartStation>{s0}</StartStation>',
                f'      <StartAzimuth>{math.degrees(aa):.10f}</StartAzimuth>',
                f'      <StartCoordinate><North>{na:.8f}</North><East>{ea:.8f}</East></StartCoordinate>',
                f'      <Length>{L}</Length>']
        if tag != "Straight":
            out.append('      <Direction>Right</Direction>')
        if tag == "Arc":
            out.append(f'      <Radius>{r}</Radius>')
        if tag == "ExitSpiral":
            out += [f'      <StartRadius>{r}</StartRadius>', '      <EndRadius />']
        out += [f'      <EndCoordinate><North>{nb:.8f}</North><East>{eb:.8f}</East></EndCoordinate>',
                f'    </{tag}>']
    out += ['  </HorizontalAlignment>', '  <VerticalAlignment>']
    for s, z in VPIS:
        out += ['    <VerticalPoint>', '      <KeyinMethod>VPI</KeyinMethod>',
                f'      <IntersectionPoint><Station>{s}</Station><Elevation>{z}</Elevation></IntersectionPoint>']
        if vertical_curve:
            out.append('      <SymmetricParabola><Length>10</Length></SymmetricParabola>')
        out.append('    </VerticalPoint>')
    out.append('  </VerticalAlignment>')

    def z_at(s):
        for (s0, z0), (s1, z1) in zip(VPIS, VPIS[1:]):
            if s0 <= s <= s1:
                return z0 + (z1 - z0) * (s - s0) / (s1 - s0)

    pos = []
    for s, w, wall in TEMPLATES:
        name = f"T {s}"
        seg = lambda code, h, v: (f'    <DistanceAndVerticalDistance><Code>{code}</Code>'
                                  f'<HorizontalDistance>{h}</HorizontalDistance>'
                                  f'<VerticalDistance>{v}</VerticalDistance></DistanceAndVerticalDistance>')
        body = [seg("INVERT", w, 0.0)]
        if wall:
            v = TOP - z_at(s)
            body.append(seg("WALL", v * 0.001, v))
        out += ['  <TemplateRecord>', f'    <Name>{name}</Name>'] + body + ['  </TemplateRecord>']
        pos.append((s, name))
    out.append('  <TemplatePositioning>')
    for s, name in pos:
        side = f'<Type>UserDefined</Type><Name>{name}</Name>'
        out.append(f'    <ApplyTemplates><Station>{s}</Station><LeftSide>{side}</LeftSide>'
                   f'<RightSide>{side}</RightSide></ApplyTemplates>')
    out += ['  </TemplatePositioning>', '</TrimbleRoad>']
    return "\n".join(out) + "\n"
