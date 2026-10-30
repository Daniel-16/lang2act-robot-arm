#!/usr/bin/env python
"""Phase 1 gate: TCP reaches 20 random reachable points within 5 mm.

    python scripts/02_ik_test.py
    python scripts/02_ik_test.py --points 50 --viewer
"""

import sys

from lang2act.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["ik-test", *sys.argv[1:]]))
