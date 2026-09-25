# Automated Support Ticket Classification

[![CI](https://github.com/raviteja311/Automated-Support-Ticket-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/raviteja311/Automated-Support-Ticket-Classification/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end MLOps service that classifies customer support tickets into
**billing**, **technical**, **account**, **shipping** and **general**, returning a
label, a confidence score, and the full probability distribution.

The machine learning is deliberately simple, a linear model that trains in
seconds, so the engineering takes centre stage: reproducible pipelines, versioned
data, experiment tracking, tests, containers, CI and monitoring.

> **Runs locally.** This is a portfolio project, not a hosted service. Everything
> below runs on your machine in a couple of commands. A `render.yaml` blueprint is
> included so it *can* be deployed, but no public instance exists and none is
> needed to see it work.

## Architecture

```
generate -> preprocess -> train (MLflow) -> evaluate      [DVC pipeline]
                                    |
                              model.joblib
                                    |
                     FastAPI -> Docker -> GitHub Actions -> Render
                                    |
                              Prometheus (/metrics)
```

DVC versions the data and model and makes the pipeline reproducible. MLflow
records each run's params and metrics. The trained sklearn `Pipeline` ships as a
single artifact, baked into the image at build time, so the container needs no
external storage.

## Tech stack

Python 3.12, scikit-learn, pandas, pydantic, DVC, MLflow, FastAPI, uvicorn,
Docker, GitHub Actions, pytest, ruff, Prometheus, Render.

## Current model

| Metric | Value |
|---|---|
| accuracy | 0.9300 |
| macro F1 | 0.9297 |
| classes | 5 |

Per-class scores live in [metrics/metrics.json](metrics/metrics.json), which is
Git-tracked so metric changes appear in pull request diffs.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .

dvc repro                       # generate -> preprocess -> train -> evaluate
uvicorn automated_support_ticket_classification.api.app:app --reload
```

Open http://127.0.0.1:8000/docs

## Call the API

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" -Body '{"text":"my invoice is wrong"}'
```

```
label       confidence  all_scores
-----       ----------  ----------
billing     0.938       @{billing=0.938; technical=0.018; ...}
```

Endpoints: `/` service info, `/ui` test page, `/health` liveness,
`/predict` classification, `/docs` interactive docs, `/metrics` Prometheus.

### Try it in a browser

Open **http://127.0.0.1:8000/ui** for a small test page: click an example
ticket for each category, or paste your own, and see the predicted label with
the full probability distribution as bars. No build step, no dependencies,
the page is served straight from the app.

One ticket per class, against the locally running service:

| Ticket | Predicted | Confidence |
|---|---|---|
| I need a refund for the duplicate payment of $49 | billing | 0.938 |
| The app crashes every time I open the reports page | technical | 0.902 |
| I cannot reset my password, the email never arrives | account | 0.839 |
| My order #10231 has not arrived after two weeks | shipping | 0.893 |
| What are your customer support working hours | general | 0.866 |

## Reproduce the pipeline

```powershell
dvc repro          # only stages whose inputs changed are rerun
dvc metrics show   # accuracy, macro F1 and per-class scores
dvc push           # send data and model to the configured remote
```

`dvc.yaml` invokes `python`, so run it from the activated environment or the
stages fail with `ModuleNotFoundError`.

## Tests

```powershell
pytest
```

13 tests covering data generation, preprocessing, the model pipeline, and the
API. `tests/conftest.py` builds a model on demand when none exists, so the suite
is self-sufficient on a clean CI runner.

## Docker

```powershell
docker build -t automated-support-ticket-classification .
docker run -p 8000:8000 automated-support-ticket-classification

# or the full stack, API plus Prometheus:
docker compose up --build
```

Prometheus UI at http://127.0.0.1:9090. Counters start at zero, so send a few
requests first.

The runtime image installs `requirements-serve.txt`, which excludes MLflow and
DVC. Neither is used at serving time, and omitting them keeps the image lean.

Docker is optional. `uvicorn` in the Quickstart is enough to run and exercise the
whole service; the image exists to prove the build is portable and to let CI
verify it on a clean machine.

## Deployment (optional, not used)

`render.yaml` is a working blueprint for deploying the container to Render, and
`.github/workflows/smoke.yml` is a post-deploy health check that reads a
`SERVICE_URL` secret. Neither is active: this project runs locally by design, so
there is no public instance and the smoke workflow is manual-only.

If you do deploy, uncomment the schedule in `smoke.yml` and set the secret.

## Project layout

```
src/automated_support_ticket_classification/
  config.py        typed loader for params.yaml
  logger.py        shared logging utility
  data/            generate.py, preprocess.py
  models/          train.py, evaluate.py
  api/             schemas.py, app.py
tests/             unit and API tests plus the model fixture
params.yaml        every hyperparameter and path
dvc.yaml/.lock     pipeline definition and reproducibility hashes
Dockerfile         multi-stage build, model baked in
docker-compose.yml api plus Prometheus
render.yaml        deploy blueprint
```

## Experiments

[docs/experiments.md](docs/experiments.md) records each experiment, the decision
taken, and the reasoning. It includes a data defect found by reading per-class
metrics, where a missing class and a contaminated one were costing real accuracy.

## Future improvements

Swap in a transformer for messier real-world text, replace the synthetic
generator with a real corpus, add an MLflow model registry with staged
promotion, add drift and data-quality monitoring with Evidently and a Grafana
dashboard, move serving to Kubernetes, and provision infrastructure with
Terraform.

## License

MIT. See [LICENSE](LICENSE).
