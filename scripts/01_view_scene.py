#!/usr/bin/env python
"""Phase 1: build the scene from a seed and look at it.

    python scripts/01_view_scene.py --seed 3 --viewer
    python scripts/01_view_scene.py --seed 3 --video results/scene.mp4
"""

import sys

from lang2act.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["view", *sys.argv[1:]]))
