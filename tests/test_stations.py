"""Station text formatting and parsing."""
import math

import pytest

from rxlkit.stations import excel_sta_format, fmt_sta, parse_sta


@pytest.mark.parametrize("s, dec, unit, txt", [
    (1005.814, 2, 1000, "1+005.81"),
    (-5.81, 2, 1000, "-0+005.81"),
    (-33.97, 2, 1000, "-0+033.97"),
    (0.0, 2, 1000, "0+000.00"),
    (999.996, 2, 1000, "1+000.00"),     # rounds before splitting: never 0+1000.00
    (-0.001, 2, 1000, "0+000.00"),      # no "-0+000.00"
    (50.0, 0, 1000, "0+050"),
    (191.04, 1, 1000, "0+191.0"),
    (12345.6785, 3, 1000, "12+345.679"),
    (0.125, 2, 1000, "0+000.13"),       # half up, not banker's rounding
    (90125.0, 0, 100, "901+25"),
    (11160.0, 2, 100, "111+60.00"),
    (-12.5, 1, 100, "-0+12.5"),
])
def test_fmt_sta(s, dec, unit, txt):
    assert fmt_sta(s, dec, unit=unit) == txt


def test_fmt_sta_blank_for_missing():
    assert fmt_sta(None) == ""
    assert fmt_sta(float("nan")) == ""


@pytest.mark.parametrize("txt, unit, v", [
    ("1+005.81", 1000, 1005.81),
    ("-0+005.81", 1000, -5.81),
    (" 0 + 050 ", 1000, 50.0),
    ("0+50", 1000, 50.0),
    ("901+25", 100, 90125.0),
    ("125.4", 1000, 125.4),
    ("1,005.81", 1000, 1005.81),
    (42, 1000, 42.0),
])
def test_parse_sta(txt, unit, v):
    assert parse_sta(txt, unit=unit) == pytest.approx(v)


def test_parse_rejects_bad_text():
    with pytest.raises(ValueError):
        parse_sta("STA A")
    with pytest.raises(ValueError):
        parse_sta("0+1005")


def test_round_trip():
    for s in (-1234.56, -0.5, 0.0, 7.25, 999.99, 1000.0, 25432.1):
        assert parse_sta(fmt_sta(s)) == pytest.approx(s)
        assert math.isclose(parse_sta(fmt_sta(s * 10, 2, unit=100), unit=100), s * 10)


def test_excel_format():
    assert excel_sta_format() == '0"+"000.00;-0"+"000.00'
    assert excel_sta_format(0) == '0"+"000;-0"+"000'
    assert excel_sta_format(2, unit=100) == '0"+"00.00;-0"+"00.00'
