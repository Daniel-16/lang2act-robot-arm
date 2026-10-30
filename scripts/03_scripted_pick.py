#!/usr/bin/env python
"""Phase 1/2 gate: place every cube in the bowl with a hard-coded skill list.

    python scripts/03_scripted_pick.py --seed 2
    python scripts/03_scripted_pick.py --seed 2 --viewer
    python scripts/03_scripted_pick.py --seed 2 --video results/pick.mp4
"""

import sys

from lang2act.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["pick", *sys.argv[1:]]))
