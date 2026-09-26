"""Offline threshold calibration (thin wrapper; see face_liveness.calibration).

    python tools/calibrate.py --data ./calib --model-dir ./models --out report.json

Equivalent to the installed ``face-liveness-calibrate`` entry point. The report is
evidence for a human release decision; it never changes service configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from face_liveness.calibration import main

if __name__ == "__main__":
    sys.exit(main())
