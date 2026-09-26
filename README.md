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
                     FastAPI -> Docker -> GitHub Actions (CI)
                                    |
                              Prometheus (/metrics)
```

DVC versions the data and model and makes the pipeline reproducible. MLflow
records each run's params and metrics. The trained sklearn `Pipeline` ships as a
single artifact, baked into the image at build time, so the container needs no
external storage.

## Tech stack

Python 3.12, scikit-learn, pandas, pydantic, DVC, MLflow, FastAPI, uvicorn,
Docker, docker-compose, GitHub Actions, pytest, ruff, Prometheus, Evidently.
A Render blueprint (`render.yaml`) is included but not deployed.

## Current model

| Metric | Value |
|---|---|
| accuracy | 0.9282 |
| macro F1 | 0.9171 |
| classes | 5 |
| corpus | banking77, 13,083 real support messages |

Per-class scores live in [metrics/metrics.json](metrics/metrics.json), which is
Git-tracked so metric changes appear in pull request diffs.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pip install -e .

dvc repro          # generate -> preprocess -> train -> evaluate
.\run.ps1          # starts the API and opens the test page
```

`run.ps1` finds the virtual environment whether or not it is activated, refuses
to start if the port is busy, waits until the server actually answers, then
opens http://127.0.0.1:8000/ui in your browser. Press Ctrl+C to stop it.

```powershell
.\run.ps1 -Port 8010      # serve somewhere else
.\run.ps1 -NoBrowser      # start without opening a browser
```

Double-clicking `run.cmd` does the same thing, for when you would rather not
open a terminal first.

To start it by hand instead:

```powershell
uvicorn automated_support_ticket_classification.api.app:app
```

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
| Why was I charged a fee on a cash withdrawal? | billing | 0.999 |
| I think my transfer was declined, but why? | technical | 0.987 |
| What is the need to verify my identity? | account | 0.998 |
| Can I track when my card will be delivered? | shipping | 0.946 |
| Are both Visa and Mastercard accepted? | general | 0.975 |

The model is trained on banking support text, so it expects that domain.
Retail-style tickets about parcels or app crashes are outside its training
distribution and will be routed on surface vocabulary rather than meaning.

## Beyond the model

- **Model registry with a promotion gate** ([`models/registry.py`](src/automated_support_ticket_classification/models/registry.py), [`models/promote.py`](src/automated_support_ticket_classification/models/promote.py)): a new model only takes the MLflow `production` alias if it beats the current one on macro F1, trained on the same corpus.
- **Drift monitoring** ([`monitoring/drift.py`](src/automated_support_ticket_classification/monitoring/drift.py)): Evidently checks label and text drift, with results in [metrics/drift.json](metrics/drift.json). It stays quiet on train vs test and fires on real vs synthetic data.
- **API plus Prometheus in one command**: `docker compose up` runs the service and a Prometheus instance scraping `/metrics`.
- **Tests and CI**: `pytest` runs 30 tests across data, model, API, registry and drift. CI runs ruff, the tests, a Docker build and a container smoke test.

See [docs/experiments.md](docs/experiments.md) for the experiment log and [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## License

MIT, see [LICENSE](LICENSE).

## Author

**Jetti Raviteja** · [Portfolio](https://portfolio-website-drab-six-15.vercel.app) · [GitHub](https://github.com/raviteja311) · [LinkedIn](https://www.linkedin.com/in/jettiraviteja/)
