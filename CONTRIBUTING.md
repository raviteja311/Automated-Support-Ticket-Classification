# Contributing

## Workflow

1. Branch from `main`: `feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>`.
2. Make the change, then run the same checks CI runs (below).
3. Open a pull request into `main`. CI must be green before merge.
4. Squash-merge, so `main` keeps one commit per change.

`main` is always deployable. Merging to `main` triggers an automatic deploy, so
do not merge anything you would not want live.

## Before you push

```powershell
ruff format .
ruff check . --fix
pytest
```

CI runs `ruff format --check .`, `ruff check .` and `pytest` on Python 3.12,
then builds the Docker image and smoke-tests it. Green locally is the only
reliable predictor of green in CI.

## Commit messages

Conventional Commits: `type: summary`.

| Type | When |
|---|---|
| `feat` | a new capability |
| `fix` | a bug fix |
| `chore` | maintenance, config, tooling |
| `test` | adding or changing tests |
| `docs` | documentation only |
| `ci` | CI/CD pipeline changes |
| `build` | build system or dependencies |
| `refactor` | code change, same behaviour |

Explain *why* in the body when the reason is not obvious from the diff.

## Changing the model or data

Code and artifacts are versioned separately. Git holds the code and DVC
pointers; the bytes live in the DVC remote.

```powershell
dvc repro          # rerun only the stages whose inputs changed
dvc push           # upload the new artifacts
git add dvc.lock metrics/metrics.json
```

Commit `dvc.lock` and `metrics/metrics.json` together with the code change, so
the pull request shows the metric movement alongside its cause.

Never `git add` a data file or `models/*.joblib`. DVC owns those, and Git and
DVC cannot both own the same file.

## Adding a dependency

Pin it exactly in `requirements.txt`, and add it to `requirements-serve.txt`
only if serving genuinely needs it at runtime. Dev-only tools belong in
`requirements-dev.txt`.

If a transitive dependency ever breaks a pinned tool, pin the transitive one too
and leave a comment explaining why. There are existing examples in
`requirements.txt`.

## Configuration

Every tunable value lives in `params.yaml` and is read through the typed loader
in `config.py`. Do not hard-code paths or hyperparameters in modules. DVC watches
the declared params and reruns the affected stages when they change.
