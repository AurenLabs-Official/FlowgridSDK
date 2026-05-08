# Model Comparison Reporting

Phase C requires comparable artifacts for runtime and ML decisions. Use the same JSON shapes produced by CLI runs.

## Train reports

JSON from `flowgrid-llm train --run-report-out ...` includes fields such as `before_ce`, `after_ce` (schema may evolve).

Compare two train runs:

```bash
python tools/compare_run_reports.py --kind train --a target/mlops/run_a.json --b target/mlops/run_b.json
```

## Eval reports

Eval run reports wrap metrics under `report` (e.g. `ppl`).

```bash
python tools/compare_run_reports.py --kind eval --a target/mlops/eval_a.json --b target/mlops/eval_b.json
```

## Release gate bundle

Combine with [validate_release_gates.py](../tools/validate_release_gates.py) for cut decisions.

## Practices

- Store artifacts under `target/mlops/` or your artifact store; filenames should encode model id + git SHA.
- Attach comparison output to PRs when changing training defaults or runtime knobs.
