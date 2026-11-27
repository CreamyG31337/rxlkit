"""Station text: format and parse chainage like TBC / Civil 3D ("1+005.81", "-0+033.97", "901+25").

``unit`` is the station block: 1000 for metric km stationing (``0+000.000``), 100 for 100 m / 100 ft
highway stationing (``901+25.00``). Values are rounded *before* splitting, so 999.996 m formats as
``1+000.00``, never ``0+1000.00``, and tiny negatives that round to zero lose their sign.
"""
import math
import re


def fmt_sta(s, dec=2, unit=1000):
    """Station text: ``fmt_sta(1005.814) -> '1+005.81'``, ``fmt_sta(-5.81) -> '-0+005.81'``,
    ``fmt_sta(90125, 0, unit=100) -> '901+25'``. NaN/None -> ''."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = float(s)
    q = int(math.floor(abs(s) * 10 ** dec + 0.5 + 1e-9))  # half-up, in units of the last shown decimal
    blocks, rem = divmod(q, unit * 10 ** dec)
    digits = len(str(unit - 1))
    sign = "-" if s < 0 and q else ""
    if dec <= 0:
        return f"{sign}{blocks}+{rem:0{digits}d}"
    whole, frac = divmod(rem, 10 ** dec)
    return f"{sign}{blocks}+{whole:0{digits}d}.{frac:0{dec}d}"


_STA = re.compile(r"^\s*([+-]?)\s*(\d+)\s*\+\s*(\d+(?:\.\d*)?)\s*$")


def parse_sta(text, unit=1000):
    """Station text or number -> metres (float). Accepts ``'1+005.81'``, ``'-0+033.97'``, ``'0+50'``,
    ``'901+25'`` (with ``unit=100``), ``'125.4'`` and plain numbers. Raises ValueError otherwise.

    The part after ``+`` is taken as metres within the block, so ``'0+50'`` is 50 m, not 500 m.
    """
    if isinstance(text, (int, float)):
        return float(text)
    t = str(text).strip().replace(",", "")
    m = _STA.match(t)
    if m:
        sign, blocks, rem = m.groups()
        v = int(blocks) * unit + float(rem)
        if float(rem) >= unit:
            raise ValueError(f"station {text!r}: {rem} is not less than the {unit} m block")
        return -v if sign == "-" else v
    try:
        return float(t)
    except ValueError:
        raise ValueError(f"not a station: {text!r}") from None


def excel_sta_format(dec=2, unit=1000):
    """Excel/LibreOffice number format that *shows* a numeric station as station text:
    ``excel_sta_format() -> '0"+"000.00;-0"+"000.00'`` (1005.81 shows ``1+005.81``, -5.81 ``-0+005.81``).

    The cell stays a number, so it sorts, filters and works in formulas, and the minus sign is never lost
    (a CSV opened in Excel turns "-0+005.81" into text or drops it).
    """
    digits = "0" * len(str(unit - 1))
    pos = f'0"+"{digits}' + (f".{'0' * dec}" if dec > 0 else "")
    return f"{pos};-{pos}"
