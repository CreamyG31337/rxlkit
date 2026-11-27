"""Synthetic tests for LandXML <Alignment> reading and <PlanFeature> read/write (N 1000 / E 2000)."""
import math

import numpy as np
import pytest

from rxlkit.landxml import (
    PlanFeature,
    read_alignment,
    read_alignments,
    read_plan_features,
    write_plan_features,
)


def _write_sample_alignment(path):
    # Synthetic alignment starting at (N 1000, E 2000), staStart=100.0:
    # 1. Line North 50 m -> (1050, 2000)
    # 2. Curve CW (right) 90 deg, R=20 m, center=(1050, 2020) -> (1070, 2020), L=10*pi
    # 3. Line East 40 m -> (1070, 2060)
    # 4. Curve CCW (left) 90 deg, R=10 m, center=(1080, 2060) -> (1080, 2070), L=5*pi
    l_c1 = 20.0 * math.pi / 2.0
    l_c2 = 10.0 * math.pi / 2.0
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
  <Alignments>
    <Alignment name="CL-TEST" staStart="100.0">
      <CoordGeom>
        <Line length="50.0">
          <Start>1000.0 2000.0</Start>
          <End>1050.0 2000.0</End>
        </Line>
        <Curve rot="cw" radius="20.0" length="{l_c1:.8f}">
          <Start>1050.0 2000.0</Start>
          <Center>1050.0 2020.0</Center>
          <End>1070.0 2020.0</End>
        </Curve>
        <Line length="40.0">
          <Start>1070.0 2020.0</Start>
          <End>1070.0 2060.0</End>
        </Line>
        <Curve rot="ccw" radius="10.0" length="{l_c2:.8f}">
          <Start>1070.0 2060.0</Start>
          <Center>1080.0 2060.0</Center>
          <End>1080.0 2070.0</End>
        </Curve>
      </CoordGeom>
    </Alignment>
  </Alignments>
</LandXML>
"""
    path.write_text(xml, encoding="utf-8")
    return 100.0 + 50.0 + l_c1 + 40.0 + l_c2


def test_read_alignment_lines_and_curves(tmp_path):
    xml = tmp_path / "aln.xml"
    expected_end = _write_sample_alignment(xml)

    aln = read_alignment(xml, "CL-TEST")
    assert aln.name == "CL-TEST"
    assert aln.start == pytest.approx(100.0)
    assert aln.end == pytest.approx(expected_end, abs=1e-5)

    # Start and end coordinates
    n0, e0, az0 = aln.at(100.0)
    assert (n0, e0) == pytest.approx((1000.0, 2000.0), abs=1e-4)
    assert az0 == pytest.approx(0.0, abs=1e-4)

    n_end, e_end, az_end = aln.at(aln.end)
    assert (n_end, e_end) == pytest.approx((1080.0, 2070.0), abs=1e-3)
    assert az_end == pytest.approx(0.0, abs=1e-3)

    # Midpoint of CW curve (45 deg): N = 1050 + 20*sin(45), E = 2020 - 20*cos(45)
    s_mid = 150.0 + 20.0 * math.pi / 4.0
    nm, em, azm = aln.at(s_mid)
    assert nm == pytest.approx(1050.0 + 20.0 * math.sin(math.pi / 4.0), abs=1e-3)
    assert em == pytest.approx(2020.0 - 20.0 * math.cos(math.pi / 4.0), abs=1e-3)
    assert azm == pytest.approx(math.pi / 4.0, abs=1e-3)

    # Right offset (+5 m) on North-heading tangent -> East +5 m
    np_, ep_ = aln.point(125.0, 5.0)
    assert (np_, ep_) == pytest.approx((1025.0, 2005.0), abs=1e-4)

    # Inverse station_offset round-trip (scalar and array)
    s_rt, o_rt = aln.station_offset(1025.0, 2005.0)
    assert s_rt == pytest.approx(125.0, abs=1e-3)
    assert o_rt == pytest.approx(5.0, abs=1e-3)

    # Offset on East-heading tangent (right = South)
    ss, oo = aln.station_offset([1067.0, 1072.0], [2040.0, 2040.0])
    s_east = 150.0 + 20.0 * math.pi / 2.0 + 20.0
    assert ss == pytest.approx([s_east, s_east], abs=1e-2)
    assert oo == pytest.approx([3.0, -2.0], abs=1e-2)

    stas, ns, es = aln.sample(step=5.0)
    assert len(stas) == len(ns) == len(es)
    assert stas[0] == pytest.approx(100.0)
    assert stas[-1] == pytest.approx(aln.end)


def test_read_alignment_with_spiral(tmp_path):
    # Synthetic clothoid: straight North 20 m -> CW spiral INF to R=50 m, L=20 m
    L = 20.0
    R = 50.0
    # Exact Fresnel-like integration for endpoint
    k1 = 1.0 / R
    nstep = 400
    h = L / nstep
    n, e, az = 1020.0, 2000.0, 0.0
    for i in range(nstep):
        sa, sb = i * h, (i + 1) * h
        ka, kb = k1 * sa / L, k1 * sb / L
        azm = az + ka * h / 2.0 + (kb - ka) * h / 6.0
        n += h * math.cos(azm)
        e += h * math.sin(azm)
        az += 0.5 * (ka + kb) * h
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
  <Alignments>
    <Alignment name="SPIRAL-TEST" staStart="0.0">
      <CoordGeom>
        <Line length="20.0">
          <Start>1000.0 2000.0</Start>
          <End>1020.0 2000.0</End>
        </Line>
        <Spiral rot="cw" radiusStart="INF" radiusEnd="{R:.3f}" length="{L:.3f}">
          <Start>1020.0 2000.0</Start>
          <PI>1033.3333 2000.0</PI>
          <End>{n:.6f} {e:.6f}</End>
        </Spiral>
      </CoordGeom>
    </Alignment>
  </Alignments>
</LandXML>
"""
    p = tmp_path / "spiral.xml"
    p.write_text(xml, encoding="utf-8")
    alns = read_alignments(p)
    assert len(alns) == 1
    aln = alns[0]
    ne, ee, aze = aln.at(40.0)
    assert (ne, ee) == pytest.approx((n, e), abs=1e-4)
    assert aze == pytest.approx(0.5 * k1 * L, abs=1e-3)


def test_plan_features_round_trip(tmp_path):
    xml = tmp_path / "pipes.xml"
    f1 = PlanFeature(
        name="PF1",
        points=np.array([[1000.0, 2000.0, 100.0], [1005.0, 2000.0, 99.95], [1005.0, 2008.0, 99.87]]),
        layer="DUCT-CL",
        color="0,255,0",
    )
    f2 = PlanFeature(
        name="PF2-ARC",
        points=np.zeros((0, 3)),
        layer="DUCT-CL",
        color="255,128,0",
        segments=[
            {"kind": "line", "start": (1000.0, 2002.0, 100.0), "end": (1004.0, 2002.0, 99.98)},
            {"kind": "curve", "rot": "cw", "radius": 2.0,
             "start": (1004.0, 2002.0, 99.98), "center": (1004.0, 2004.0),
             "end": (1006.0, 2004.0, 99.95), "pi": (1006.0, 2002.0)},
        ],
    )
    f3 = PlanFeature(
        name="BOX",
        points=np.array([[1000.0, 2000.0, 100.0], [1000.0, 2001.0, 100.0], [1001.0, 2001.0, 100.0]]),
        layer="BOXES",
        closed=True,
    )
    write_plan_features(xml, [f1, f2, f3], project_name="SYNTH-TEST")
    loaded = read_plan_features(xml, arc_step=0.2)
    assert [f.name for f in loaded] == ["PF1", "PF2-ARC", "BOX"]
    assert loaded[0].layer == "DUCT-CL"
    assert loaded[0].color == "0,255,0"
    assert loaded[0].points == pytest.approx(f1.points, abs=1e-4)
    assert loaded[1].points[0] == pytest.approx([1000.0, 2002.0, 100.0], abs=1e-4)
    assert loaded[1].points[-1] == pytest.approx([1006.0, 2004.0, 99.95], abs=1e-4)
    assert len(loaded[1].points) > 10  # arc was sampled smoothly
    assert loaded[2].closed is True


def test_plan_feature_curve_dir_is_ccw_from_east(tmp_path):
    # LandXML directions are math angles (CCW from +E), like Civil 3D/TBC Line@dir, not north azimuths.
    # CW quarter circle: start heading North (dir = 90 deg), end heading East (dir = 0).
    import xml.etree.ElementTree as ET
    xml = tmp_path / "arc.xml"
    seg = {"kind": "curve", "rot": "cw", "start": (1000.0, 2000.0, 0.0), "center": (1000.0, 2010.0),
           "end": (1010.0, 2010.0, 0.0)}
    write_plan_features(xml, [PlanFeature("ARC", np.zeros((0, 3)), segments=[seg])])
    c = next(e for e in ET.parse(xml).getroot().iter() if e.tag.endswith("Curve"))
    assert float(c.get("dirStart")) == pytest.approx(math.pi / 2.0, abs=1e-9)
    assert float(c.get("dirEnd")) == pytest.approx(0.0, abs=1e-9)
    assert float(c.get("length")) == pytest.approx(5.0 * math.pi, abs=1e-4)


def test_spiral_without_pi_uses_dir_start_in_degrees(tmp_path):
    # First element is a spiral with no PI: start tangent must come from dirStart (degrees, CCW from E).
    L, R = 20.0, 50.0
    k1, nstep = 1.0 / R, 400
    h = L / nstep
    n, e, az = 1000.0, 2000.0, math.pi / 4.0  # heading NE
    for i in range(nstep):
        ka, kb = k1 * i * h / L, k1 * (i + 1) * h / L
        azm = az + ka * h / 2.0 + (kb - ka) * h / 6.0
        n += h * math.cos(azm)
        e += h * math.sin(azm)
        az += 0.5 * (ka + kb) * h
    p = tmp_path / "spi.xml"
    p.write_text(f"""<?xml version="1.0"?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
  <Units><Metric linearUnit="meter" angularUnit="decimal degrees" directionUnit="decimal degrees"/></Units>
  <Alignments><Alignment name="S" staStart="0"><CoordGeom>
    <Spiral rot="cw" radiusStart="INF" radiusEnd="{R}" length="{L}" dirStart="45.0">
      <Start>1000 2000</Start><End>{n:.6f} {e:.6f}</End>
    </Spiral>
  </CoordGeom></Alignment></Alignments>
</LandXML>
""", encoding="utf-8")
    aln = read_alignment(p)
    assert aln.at(0.0)[2] == pytest.approx(math.pi / 4.0, abs=1e-9)
    assert aln.at(10.0)[:2] == pytest.approx(_mid_clothoid(10.0, L, R), abs=2e-3)


def _mid_clothoid(s, L, R, n=1000.0, e=2000.0, az=math.pi / 4.0, steps=2000):
    h = s / steps
    for i in range(steps):
        sm = (i + 0.5) * h
        a = az + sm * sm / (2.0 * R * L)
        n += h * math.cos(a)
        e += h * math.sin(a)
    return n, e


def test_unsupported_alignment_element_warns(tmp_path):
    p = tmp_path / "irr.xml"
    p.write_text("""<?xml version="1.0"?>
<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" version="1.2">
  <Alignments><Alignment name="A" staStart="0"><CoordGeom>
    <Line><Start>1000 2000</Start><End>1010 2000</End></Line>
    <IrregularLine><Start>1010 2000</Start><End>1020 2000</End><PntList2D>1010 2000 1020 2000</PntList2D></IrregularLine>
  </CoordGeom></Alignment></Alignments>
</LandXML>
""", encoding="utf-8")
    with pytest.warns(UserWarning, match="IrregularLine"):
        aln = read_alignment(p)
    assert aln.end == pytest.approx(10.0)
