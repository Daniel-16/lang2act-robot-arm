# results/

Everything in this directory is generated. `python -m lang2act.eval` writes one
timestamped run folder per evaluation:

```
results/<run-id>/
  episodes/<command-id>_seed<N>.json   # full episode log (command, scene, plan, steps, outcome)
  summary.json                         # machine-readable aggregate
  summary.md                           # the results table that goes in the README
```

The folder is gitignored apart from this file, so a fresh clone starts empty and
`python -m lang2act.eval` regenerates the report.
