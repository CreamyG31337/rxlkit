"""python -m unittest discover -s tests   (from the rxl_tools folder)"""
import math
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rxlkit import Corridor, channel_surface, read_surface, tin, wall_surface, write_surfaces  # noqa: E402
from rxlkit.cli import main  # noqa: E402

import synthetic  # noqa: E402


def corridor(**kw):
    return Corridor(ET.fromstring(synthetic.rxl_text(**kw)))


class AlignmentTest(unittest.TestCase):
    def setUp(self):
        self.al = corridor().alignment

    def test_element_ends_close(self):
        for tag, _, dn, de in self.al.closure():
            self.assertLess(math.hypot(dn, de), 1e-4, tag)

    def test_station_offset_round_trip(self):
        rng = np.random.default_rng(1)
        s = rng.uniform(0, 110, 200)
        o = rng.uniform(-15, 15, 200)
        n, e = self.al.point(s, o)
        s2, o2 = self.al.station_offset(n, e)
        self.assertLess(np.abs(s2 - s).max(), 1e-5)
        self.assertLess(np.abs(o2 - o).max(), 1e-5)

    def test_offset_is_right_positive(self):
        # heading 45 deg at the start: right of the line is south-east
        n, e = self.al.point(1.0, 2.0)
        n0, e0, _ = self.al.at(1.0)
        self.assertLess(n - n0, 0)
        self.assertGreater(e - e0, 0)

    def test_heading_continuous_between_table_rows(self):
        _, _, a = self.al.at(np.array([60.004, 60.005, 60.006]))
        self.assertAlmostEqual(a[1] - a[0], a[2] - a[1], places=9)
        self.assertGreater(a[1] - a[0], 0)  # right-hand arc: azimuth increasing

    def test_outside_raises(self):
        with self.assertRaises(ValueError):
            self.al.at(-1.0)


class ProfileTemplateTest(unittest.TestCase):
    def setUp(self):
        self.c = corridor()

    def test_profile_step(self):
        p = self.c.profile
        self.assertEqual(p.elevation(39.99), 100.0)
        self.assertEqual(p.elevation(40.0), 97.0)
        self.assertAlmostEqual(p.elevation(75.0), 97 - 0.7 / 2, places=9)

    def test_vertical_curves_rejected(self):
        with self.assertRaises(NotImplementedError):
            corridor(vertical_curve=True)

    def test_template_interpolation(self):
        L, R = self.c.templates.at(70.0)
        self.assertAlmostEqual(R[0][1], 6.0)
        self.assertEqual(len(R), 2)
        side = self.c.side(70.0, "L")
        self.assertAlmostEqual(side[0][0], -6.0)
        self.assertAlmostEqual(side[1][1], 135.0, places=6)  # wall top at TOP

    def test_steps(self):
        self.assertEqual(self.c.steps(), [(9.99, 10.0), (39.99, 40.0)])


class SurfaceTest(unittest.TestCase):
    def setUp(self):
        self.c = corridor()

    def assertGoodTin(self, s):
        q = tin.quality(s.points, s.faces)
        self.assertTrue(tin.quality_ok(q), q)

    def test_channel_surface(self):
        s = channel_surface(self.c)
        self.assertGoodTin(s)
        al, P, F = self.c.alignment, s.points, s.faces
        cases = [(5.0, 0.0, 100.0), (5.0, 9.5, 100.0),      # invert-only entrance
                 (25.0, 0.0, 100.0), (25.0, -4.9, 100.0),   # channel
                 (39.995, 0.0, 98.5),                       # inside the step
                 (60.0, 5.9, 97 - 0.2)]                     # after the step, near the toe
        for sta, off, z in cases:
            zs = tin.elevations_at(P, F, *al.point(sta, off))
            self.assertEqual(len(np.unique(np.round(zs, 6))), 1, (sta, off, zs))
            self.assertAlmostEqual(zs[0], z, delta=0.01, msg=(sta, off))
        # wall: half way up the 1000:1 wall at station 25, offset toe + half the run
        zs = tin.elevations_at(P, F, *al.point(25.0, 5.0 + 0.0175))
        self.assertAlmostEqual(zs[0], 117.5, delta=0.5)

    def test_wall_surfaces_and_corner(self):
        for side, sgn in (("L", -1), ("R", 1)):
            s = wall_surface(self.c, side)
            self.assertGoodTin(s)
            # just downstream of the 40.0 corner, between the upstream wall line (5.0) and the box wall (6.0):
            # the nearest face is the cross face, 0.15 m upstream
            n, e = self.c.alignment.point(40.15, sgn * 5.5)
            self.assertAlmostEqual(tin.distance(s.points, s.faces, (n, e, 100.0)), 0.15, delta=0.04)
            # well upstream: horizontal distance to the 5.0 wall
            n, e = self.c.alignment.point(25.0, sgn * 4.0)
            self.assertAlmostEqual(tin.distance(s.points, s.faces, (n, e, 110.0)), 1.0, delta=0.01)

    def test_landxml_round_trip_and_cli(self):
        s = channel_surface(self.c, name="T & <channel>")
        with tempfile.TemporaryDirectory() as d:
            rxl = os.path.join(d, "t.rxl")
            with open(rxl, "w", encoding="utf-8") as fh:
                fh.write(synthetic.rxl_text())
            xml = os.path.join(d, "t.xml")
            write_surfaces(xml, [s])
            r = read_surface(xml)
            self.assertEqual(r.name, "T & <channel>")
            self.assertLess(np.abs(r.points - s.points).max(), 5e-5)
            self.assertTrue(np.array_equal(r.faces, s.faces))
            self.assertEqual(main(["check", xml]), 0)
            self.assertEqual(main(["verify", rxl, xml]), 0)
            self.assertEqual(main(["walls", rxl, d, "--prefix", "T"]), 0)
            self.assertTrue(os.path.exists(os.path.join(d, "T - LEFT WALL.xml")))


if __name__ == "__main__":
    unittest.main()
