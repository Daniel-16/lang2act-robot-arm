"""Lang2Act: natural-language tabletop manipulation on a simulated Franka Panda.

Three layers, strictly separated:

* :mod:`lang2act.planner`  -- the LLM decides *what* (a JSON plan over 5 skills).
* :mod:`lang2act.skills`   -- deterministic code decides *how* (IK + motion).
* :mod:`lang2act.executor` -- the simulator decides *whether it worked*.
"""

__version__ = "0.1.0"
