#!/usr/bin/env python
"""Phase 3: one natural-language command, planned and executed.

    python scripts/run_command.py "put the red cube in the bowl" --viewer
    python scripts/run_command.py "put all the cubes in the bowl" --llm anthropic
"""

import sys

from lang2act.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
