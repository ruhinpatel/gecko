"""Tests for the fixed daltonproject.dalton.OutputParser.polarizabilities method.

Background
----------
BTS-75 (DaltonProject Integration) — Step 2: replacing Gecko's legacy Dalton
output parser with daltonproject.dalton.OutputParser.

A bug was found in OutputParser.polarizabilities that caused an IndexError on
multi-frequency alpha calculations for non-linear molecules (H2O, C6H6, etc.).

Root cause
----------
The original code counted '@ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES' markers
to advance its frequency index.  For non-linear molecules Dalton emits one RSPLR
block per dipole operator (XDIPLEN, YDIPLEN, ZDIPLEN), each repeating the full
set of N-1 frequency markers.  With N=5 frequencies across 3 blocks that is
3*(5-1)=12 markers, but polarizability_mat has shape (5,3,3) — valid indices 0-4.
count reached 5 triggering: IndexError: index 5 is out of bounds for axis 0 with size 5.

Fix
---
Skip all intermediate RSPLR blocks and parse only the consolidated final-results
section (lines starting with '@ -<<'), which Dalton emits exactly once per output
regardless of molecular symmetry.  See output_parser.py lines 128-157.

Guards added (Option B fork)
-----------------------------
1. Missing '@ -<<' section: raises Exception with message including
   'Could not find consolidated polarizability results'.
2. Missing 'Total CPU  time used in RESPONSE' terminator: raises Exception
   with message including 'appears truncated'.
Both previously silently returned all-zero tensors; now fail loudly.

Test results (2026-04-29)
-------------------------
All tests in this file: PASS
Existing gecko alpha/raman contract tests: PASS (4/4, unaffected — those tests use
Gecko's own parser in plugins/dalton/parse.py, not daltonproject's OutputParser).
"""
from __future__ import annotations

import os
import textwrap
import tempfile

import numpy as np
import pytest

from daltonproject.dalton.output_parser import OutputParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_out(directory: str, stem: str, content: str) -> str:
    """Write a minimal synthetic Dalton .out file, return the stem path."""
    path = os.path.join(directory, stem + ".out")
    with open(path, "w") as f:
        f.write(textwrap.dedent(content))
    return os.path.join(directory, stem)


# ---------------------------------------------------------------------------
# Fixtures: minimal synthetic Dalton linear-response output files
#
# Each fixture contains only the sections that OutputParser.polarizabilities
# actually reads.  Real output files have many thousands of additional lines.
# ---------------------------------------------------------------------------

# 3 frequencies, non-linear molecule (was broken — the bug case)
# Tensor values chosen to be easily verified:
#   omega=0:     xx=9.0, yy=8.0, zz=7.0
#   omega=0.020: xx=9.1, yy=8.1, zz=7.1
#   omega=0.040: xx=9.2, yy=8.2, zz=7.2
# Off-diagonal elements all zero.
_NONLINEAR_3FREQ = """\
     RSPLR -- operator label : XDIPLEN
     RSPLR -- frequencies    :  0.000000  0.020000  0.040000

     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.020000
     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.040000

     RSPLR -- operator label : YDIPLEN
     RSPLR -- frequencies    :  0.000000  0.020000  0.040000

     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.020000
     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.040000

     RSPLR -- operator label : ZDIPLEN
     RSPLR -- frequencies    :  0.000000  0.020000  0.040000

     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.020000
     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.040000

 @ -<< XDIPLEN  ; XDIPLEN  >> =   9.0000000000D+00
 @ -<< XDIPLEN  ; YDIPLEN  >> =   0.0000000000D+00
 @ -<< XDIPLEN  ; ZDIPLEN  >> =   0.0000000000D+00
 @ -<< YDIPLEN  ; XDIPLEN  >> =   0.0000000000D+00
 @ -<< YDIPLEN  ; YDIPLEN  >> =   8.0000000000D+00
 @ -<< YDIPLEN  ; ZDIPLEN  >> =   0.0000000000D+00
 @ -<< ZDIPLEN  ; XDIPLEN  >> =   0.0000000000D+00
 @ -<< ZDIPLEN  ; YDIPLEN  >> =   0.0000000000D+00
 @ -<< ZDIPLEN  ; ZDIPLEN  >> =   7.0000000000D+00
 @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.020000
 @ -<< XDIPLEN  ; XDIPLEN  >> =   9.1000000000D+00
 @ -<< XDIPLEN  ; YDIPLEN  >> =   0.0000000000D+00
 @ -<< XDIPLEN  ; ZDIPLEN  >> =   0.0000000000D+00
 @ -<< YDIPLEN  ; XDIPLEN  >> =   0.0000000000D+00
 @ -<< YDIPLEN  ; YDIPLEN  >> =   8.1000000000D+00
 @ -<< YDIPLEN  ; ZDIPLEN  >> =   0.0000000000D+00
 @ -<< ZDIPLEN  ; XDIPLEN  >> =   0.0000000000D+00
 @ -<< ZDIPLEN  ; YDIPLEN  >> =   0.0000000000D+00
 @ -<< ZDIPLEN  ; ZDIPLEN  >> =   7.1000000000D+00
 @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.040000
 @ -<< XDIPLEN  ; XDIPLEN  >> =   9.2000000000D+00
 @ -<< XDIPLEN  ; YDIPLEN  >> =   0.0000000000D+00
 @ -<< XDIPLEN  ; ZDIPLEN  >> =   0.0000000000D+00
 @ -<< YDIPLEN  ; XDIPLEN  >> =   0.0000000000D+00
 @ -<< YDIPLEN  ; YDIPLEN  >> =   8.2000000000D+00
 @ -<< YDIPLEN  ; ZDIPLEN  >> =   0.0000000000D+00
 @ -<< ZDIPLEN  ; XDIPLEN  >> =   0.0000000000D+00
 @ -<< ZDIPLEN  ; YDIPLEN  >> =   0.0000000000D+00
 @ -<< ZDIPLEN  ; ZDIPLEN  >> =   7.2000000000D+00
  Total CPU  time used in RESPONSE:   1.23 seconds
"""

# 1 frequency, static only — regression: was working before, must still work
# Tensor: xx=5.0, yy=6.0, zz=7.0
_STATIC_ONLY = """\
     RSPLR -- operator label : XDIPLEN
     RSPLR -- frequencies    :  0.000000

 @ -<< XDIPLEN  ; XDIPLEN  >> =   5.0000000000D+00
 @ -<< XDIPLEN  ; YDIPLEN  >> =   0.0000000000D+00
 @ -<< XDIPLEN  ; ZDIPLEN  >> =   0.0000000000D+00
 @ -<< YDIPLEN  ; XDIPLEN  >> =   0.0000000000D+00
 @ -<< YDIPLEN  ; YDIPLEN  >> =   6.0000000000D+00
 @ -<< YDIPLEN  ; ZDIPLEN  >> =   0.0000000000D+00
 @ -<< ZDIPLEN  ; XDIPLEN  >> =   0.0000000000D+00
 @ -<< ZDIPLEN  ; YDIPLEN  >> =   0.0000000000D+00
 @ -<< ZDIPLEN  ; ZDIPLEN  >> =   7.0000000000D+00
  Total CPU  time used in RESPONSE:   0.50 seconds
"""

# 5 frequencies, non-linear — edge: larger count to confirm no off-by-one
_NONLINEAR_5FREQ = """\
     RSPLR -- operator label : XDIPLEN
     RSPLR -- frequencies    :  0.000000  0.015000  0.030000  0.045000  0.060000

     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.015000
     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.030000
     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.045000
     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.060000

     RSPLR -- operator label : YDIPLEN
     RSPLR -- frequencies    :  0.000000  0.015000  0.030000  0.045000  0.060000

     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.015000
     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.030000
     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.045000
     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.060000

     RSPLR -- operator label : ZDIPLEN
     RSPLR -- frequencies    :  0.000000  0.015000  0.030000  0.045000  0.060000

     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.015000
     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.030000
     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.045000
     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.060000

 @ -<< XDIPLEN  ; XDIPLEN  >> =   1.0000000000D+01
 @ -<< YDIPLEN  ; YDIPLEN  >> =   1.0000000000D+01
 @ -<< ZDIPLEN  ; ZDIPLEN  >> =   1.0000000000D+01
 @ -<< XDIPLEN  ; YDIPLEN  >> =   0.0D+00
 @ -<< XDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; YDIPLEN  >> =   0.0D+00
 @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.015000
 @ -<< XDIPLEN  ; XDIPLEN  >> =   1.0100000000D+01
 @ -<< YDIPLEN  ; YDIPLEN  >> =   1.0100000000D+01
 @ -<< ZDIPLEN  ; ZDIPLEN  >> =   1.0100000000D+01
 @ -<< XDIPLEN  ; YDIPLEN  >> =   0.0D+00
 @ -<< XDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; YDIPLEN  >> =   0.0D+00
 @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.030000
 @ -<< XDIPLEN  ; XDIPLEN  >> =   1.0200000000D+01
 @ -<< YDIPLEN  ; YDIPLEN  >> =   1.0200000000D+01
 @ -<< ZDIPLEN  ; ZDIPLEN  >> =   1.0200000000D+01
 @ -<< XDIPLEN  ; YDIPLEN  >> =   0.0D+00
 @ -<< XDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; YDIPLEN  >> =   0.0D+00
 @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.045000
 @ -<< XDIPLEN  ; XDIPLEN  >> =   1.0300000000D+01
 @ -<< YDIPLEN  ; YDIPLEN  >> =   1.0300000000D+01
 @ -<< ZDIPLEN  ; ZDIPLEN  >> =   1.0300000000D+01
 @ -<< XDIPLEN  ; YDIPLEN  >> =   0.0D+00
 @ -<< XDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; YDIPLEN  >> =   0.0D+00
 @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.060000
 @ -<< XDIPLEN  ; XDIPLEN  >> =   1.0400000000D+01
 @ -<< YDIPLEN  ; YDIPLEN  >> =   1.0400000000D+01
 @ -<< ZDIPLEN  ; ZDIPLEN  >> =   1.0400000000D+01
 @ -<< XDIPLEN  ; YDIPLEN  >> =   0.0D+00
 @ -<< XDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; YDIPLEN  >> =   0.0D+00
  Total CPU  time used in RESPONSE:   5.00 seconds
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def tmpdir(tmp_path):
    return str(tmp_path)


def test_nonlinear_3freq_no_indexerror(tmpdir):
    """3-frequency non-linear molecule: was IndexError before fix.

    3 RSPLR blocks × 2 non-zero freq markers = 6 markers total.
    Old code: count reached 3, then 4 → IndexError on polarizability_mat[5] (size 5
    for 5 frequencies). Now fixed to parse consolidated section only.
    """
    stem = _write_out(tmpdir, "nonlinear_3freq", _NONLINEAR_3FREQ)
    pol = OutputParser(stem).polarizabilities

    assert pol.values.shape == (3, 3, 3)
    assert list(pol.frequencies) == pytest.approx([0.0, 0.02, 0.04])

    # omega = 0
    assert pol.values[0, 0, 0] == pytest.approx(9.0)
    assert pol.values[0, 1, 1] == pytest.approx(8.0)
    assert pol.values[0, 2, 2] == pytest.approx(7.0)
    assert pol.values[0, 0, 1] == pytest.approx(0.0)

    # omega = 0.020
    assert pol.values[1, 0, 0] == pytest.approx(9.1)
    assert pol.values[1, 1, 1] == pytest.approx(8.1)
    assert pol.values[1, 2, 2] == pytest.approx(7.1)

    # omega = 0.040
    assert pol.values[2, 0, 0] == pytest.approx(9.2)
    assert pol.values[2, 1, 1] == pytest.approx(8.2)
    assert pol.values[2, 2, 2] == pytest.approx(7.2)


def test_static_only_regression(tmpdir):
    """Static-only (omega=0): was working before fix, must still work."""
    stem = _write_out(tmpdir, "static_only", _STATIC_ONLY)
    pol = OutputParser(stem).polarizabilities

    assert pol.values.shape == (1, 3, 3)
    assert pol.frequencies[0] == pytest.approx(0.0)
    assert pol.values[0, 0, 0] == pytest.approx(5.0)
    assert pol.values[0, 1, 1] == pytest.approx(6.0)
    assert pol.values[0, 2, 2] == pytest.approx(7.0)


def test_nonlinear_5freq_no_indexerror(tmpdir):
    """5-frequency non-linear molecule: the exact scenario from the bug report.

    3 RSPLR blocks × 4 non-zero freq markers = 12 markers total.
    Old code: count reached 5 → IndexError on polarizability_mat[5] (size 5).
    """
    stem = _write_out(tmpdir, "nonlinear_5freq", _NONLINEAR_5FREQ)
    pol = OutputParser(stem).polarizabilities

    assert pol.values.shape == (5, 3, 3)
    assert list(pol.frequencies) == pytest.approx([0.0, 0.015, 0.030, 0.045, 0.060])

    expected_xx = [10.0, 10.1, 10.2, 10.3, 10.4]
    for i, xx in enumerate(expected_xx):
        assert pol.values[i, 0, 0] == pytest.approx(xx), f"freq index {i} xx mismatch"
        assert pol.values[i, 1, 1] == pytest.approx(xx), f"freq index {i} yy mismatch"
        assert pol.values[i, 2, 2] == pytest.approx(xx), f"freq index {i} zz mismatch"


def test_fortran_double_notation_parsed(tmpdir):
    """Fortran D-notation (e.g. 9.9278D+00) is correctly parsed as float."""
    stem = _write_out(tmpdir, "static_only", _STATIC_ONLY)
    pol = OutputParser(stem).polarizabilities
    # Values were written as xD+00 notation; assert they're real floats not NaN
    assert np.all(np.isfinite(pol.values))


def test_off_diagonal_elements_set(tmpdir):
    """Off-diagonal tensor components are populated from the consolidated section."""
    content = """\
     RSPLR -- operator label : XDIPLEN
     RSPLR -- frequencies    :  0.000000

 @ -<< XDIPLEN  ; XDIPLEN  >> =   5.0000000000D+00
 @ -<< XDIPLEN  ; YDIPLEN  >> =   1.0000000000D+00
 @ -<< XDIPLEN  ; ZDIPLEN  >> =   2.0000000000D+00
 @ -<< YDIPLEN  ; XDIPLEN  >> =   1.0000000000D+00
 @ -<< YDIPLEN  ; YDIPLEN  >> =   6.0000000000D+00
 @ -<< YDIPLEN  ; ZDIPLEN  >> =   3.0000000000D+00
 @ -<< ZDIPLEN  ; XDIPLEN  >> =   2.0000000000D+00
 @ -<< ZDIPLEN  ; YDIPLEN  >> =   3.0000000000D+00
 @ -<< ZDIPLEN  ; ZDIPLEN  >> =   7.0000000000D+00
  Total CPU  time used in RESPONSE:   0.50 seconds
"""
    stem = _write_out(tmpdir, "offdiag", content)
    pol = OutputParser(stem).polarizabilities

    t = pol.values[0]
    assert t[0, 0] == pytest.approx(5.0)
    assert t[1, 1] == pytest.approx(6.0)
    assert t[2, 2] == pytest.approx(7.0)
    assert t[0, 1] == pytest.approx(1.0)
    assert t[0, 2] == pytest.approx(2.0)
    assert t[1, 2] == pytest.approx(3.0)
    # Tensor is symmetric
    assert t[1, 0] == pytest.approx(t[0, 1])
    assert t[2, 0] == pytest.approx(t[0, 2])
    assert t[2, 1] == pytest.approx(t[1, 2])


# ---------------------------------------------------------------------------
# Blindspot documentation tests
# These tests characterise current silent-failure behaviour.
# They are NOT expected to raise exceptions — they document that the parser
# silently returns zeros rather than erroring when the output is malformed.
# Proper guards should be added in the upstream daltonproject PR.
# ---------------------------------------------------------------------------

def test_blindspot_missing_consolidated_section_raises(tmpdir):
    """Missing '@ -<<' section now raises a descriptive exception (guard added)."""
    content = """\
     RSPLR -- operator label : XDIPLEN
     RSPLR -- frequencies    :  0.000000  0.020000

     @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.020000

  Total CPU  time used in RESPONSE:   0.50 seconds
"""
    stem = _write_out(tmpdir, "missing_consolidated", content)
    with pytest.raises(Exception, match='Could not find consolidated polarizability results'):
        OutputParser(stem).polarizabilities


def test_blindspot_missing_terminator_raises(tmpdir):
    """Missing 'Total CPU  time used in RESPONSE' now raises a descriptive exception (guard added)."""
    content = """\
     RSPLR -- operator label : XDIPLEN
     RSPLR -- frequencies    :  0.000000  0.020000

 @ -<< XDIPLEN  ; XDIPLEN  >> =   9.0000000000D+00
 @ -<< YDIPLEN  ; YDIPLEN  >> =   8.0000000000D+00
 @ -<< ZDIPLEN  ; ZDIPLEN  >> =   7.0000000000D+00
 @ -<< XDIPLEN  ; YDIPLEN  >> =   0.0D+00
 @ -<< XDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; YDIPLEN  >> =   0.0D+00
 @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.020000
 @ -<< XDIPLEN  ; XDIPLEN  >> =   9.1000000000D+00
 @ -<< YDIPLEN  ; YDIPLEN  >> =   8.1000000000D+00
 @ -<< ZDIPLEN  ; ZDIPLEN  >> =   7.1000000000D+00
 @ -<< XDIPLEN  ; YDIPLEN  >> =   0.0D+00
 @ -<< XDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; ZDIPLEN  >> =   0.0D+00
 @ -<< YDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; XDIPLEN  >> =   0.0D+00
 @ -<< ZDIPLEN  ; YDIPLEN  >> =   0.0D+00
"""
    stem = _write_out(tmpdir, "missing_terminator", content)
    with pytest.raises(Exception, match='appears truncated'):
        OutputParser(stem).polarizabilities
