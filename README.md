# Automated Support Ticket Classification

[![CI](https://github.com/raviteja311/Automated-Support-Ticket-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/raviteja311/Automated-Support-Ticket-Classification/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end MLOps service that classifies customer support tickets into
**billing**, **technical**, **account**, **card_delivery** and **general**, returning a
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
generate -> preprocess -> train (MLflow) -> evaluate -> baselines   [DVC pipeline]
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

### Baselines

The `baselines` stage scores a majority-class predictor and two alternative
classifiers on the same 2,617-row test split and the same TF-IDF settings
(5,000 features, unigrams + bigrams). Written to
[metrics/baselines.json](metrics/baselines.json), Git-tracked like the metrics.

| Model | accuracy | macro F1 |
|---|---|---|
| majority class (DummyClassifier) | 0.3779 | 0.1097 |
| TF-IDF + ComplementNB | 0.9117 | 0.8968 |
| **TF-IDF + LogisticRegression (production)** | **0.9282** | **0.9171** |
| TF-IDF + LinearSVC | 0.9404 | 0.9333 |

The majority-class floor shows how little of the headline number is class
balance: always answering `billing` gets 37.8% accuracy and 0.11 macro F1.

LinearSVC beats production by 1.62pp macro F1, past the stage's 1pp "clear win"
margin, so `baselines.json` reports `production_beaten: true`. Production is
deliberately unchanged: LinearSVC has no `predict_proba`, and the API returns a
confidence and a full probability distribution. Adopting it means wrapping it
in `CalibratedClassifierCV` and re-measuring, which is a separate change. See
E6 in [docs/experiments.md](docs/experiments.md).

### Latency

Measured with `python scripts/benchmark_latency.py -n 500` (500 timed calls
after 20 warm-up calls, texts drawn from the test split):

| Measurement | p50 | p95 | p99 |
|---|---|---|---|
| `POST /predict`, in-process TestClient | 10.34 ms | 13.89 ms | 17.33 ms |
| `POST /predict` over HTTP, local uvicorn on loopback | 12.19 ms | 16.80 ms | 21.26 ms |
| `predict_proba` on one message, model only | 0.36 ms | 0.51 ms | 0.57 ms |
| `POST /predict` over the internet, deployed | not yet measured | | |

The HTTP row comes from `python scripts/benchmark_latency.py --url
http://127.0.0.1:8000 -n 500` against a local `uvicorn` server: real sockets
and uvicorn, but no network distance. Run the same command with the deployed
URL to fill in the last row. The script reports the first request separately
as `first_request_ms`, because a free-tier instance that has been asleep takes
30 to 60 seconds to answer it, and that cold start would otherwise distort p95.

Single run on a Windows 11 laptop (AMD64 CPU, family 25 model 80, Python 3.12.10), one request at a
time. The in-process figure excludes the network and uvicorn but includes validation,
JSON encoding and the prediction log write. Treat both as an order of
magnitude, not a service-level objective: they will differ on other machines
and under concurrent load. Almost all of the API time is framework and I/O,
not the model.

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

For an exact rebuild of the environment, every transitive dependency is pinned
in [requirements.lock.txt](requirements.lock.txt) (captured with `pip freeze` on
Windows, Python 3.12.10):

```powershell
pip install -r requirements.lock.txt
pip install -e . --no-deps
```

`requirements*.txt` pin direct dependencies only and stay the source of truth;
regenerate the lock file after changing them.

### Network access

The default corpus is banking77, so the first `dvc repro` and **every Docker
build** download it (about 1 MB from GitHub). The download retries with
exponential backoff (4 attempts) and never caches a partial file, but it does
need network. The test suite does not: CI's `pytest` job never runs the
pipeline, because `tests/conftest.py` builds a small model on demand. To build
fully offline, set `data.source: synthetic` in `params.yaml`, which uses the
seeded template generator instead, at the cost of training on templates.

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
label       predicted_label  confidence  all_scores
-----       ---------------  ----------  ----------
billing     billing          0.938       @{billing=0.938; technical=0.018; ...}
```

### Human review for low-confidence predictions

A real triage system routes the messages it is sure about and hands the rest
to a person. When the top probability is below `serve.review_threshold` in
`params.yaml`, `/predict` answers `label: "needs_review"`, and keeps the
model's best guess in `predicted_label` and the full `all_scores`, so the
reviewer starts from the suggestion. The default of `0.0` disables the route.

The evaluate stage writes the trade-off to `reports/coverage_accuracy.csv` and
`.png`: for each threshold, the share of messages auto-routed and the accuracy
on them. The Prometheus counter `ticket_routed_total{queue=...}` counts
answers by routed queue, so the review rate can be graphed and alerted on. A
rising review rate means the model is less sure of its traffic, which is a
drift signal of its own.

Endpoints: `/` service info, `/ui` test page, `/health` liveness,
`/predict` classification, `/docs` interactive docs, `/metrics` Prometheus.

`/predict` accepts 1 to 5,000 characters; longer input is rejected with 422.
If the model artifact is missing (not yet trained or pulled), it answers 503
with a message saying so, rather than a 500, and recovers without a restart once
the file appears.

### Prediction log and privacy

Every prediction is appended to `data/predictions.jsonl`, which the drift
monitor reads. Before anything is written, personal identifiers are replaced
with placeholders:

| Identifier | Example | Becomes |
|---|---|---|
| Email | `ravi.kumar@gmail.com` | `[email]` |
| UPI ID | `ravi@okaxis`, `9876543210@ybl` | `[upi]` |
| PAN | `ABCDE1234F` (any case) | `[pan]` |
| IFSC | `SBIN0001234` (any case) | `[ifsc]` |
| 8+ digit runs: card, account, Aadhaar, phone | `1234 5678 9012`, `+91 98765 43210` | `[number]` |
| Card fragments | `card ending 4321`, `xxxx4321` | `card ending [number]` |

Short numbers such as amounts and order IDs are kept, because they help routing
and identify nobody. `python scripts/pii_recall.py` scores the redaction on 56
hand-written cases in `tests/fixtures/pii_cases.csv` and writes
`metrics/pii_redaction.json`. Recall went from 0.53 (emails and digit runs
only) to 1.00, with all 15 PII-free messages left untouched. Those cases were
written alongside the patterns, so treat 1.00 as "covers what it was designed
for", not as a measure on unseen text. This is a regex baseline: names and
street addresses have no fixed shape and would need named-entity recognition,
which is out of scope.

The file rotates at 5 MB and keeps 3 rotated copies, so raw traffic is not
retained indefinitely; the drift monitor reads the rotated copies too. All of
it is configurable under `monitoring:` in `params.yaml` (`redact`, `max_bytes`,
`backup_count`, `predictions_path`).

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
| Can I track when my card will be delivered? | card_delivery | 0.946 |
| Are both Visa and Mastercard accepted? | general | 0.975 |

The model is trained on banking support text, so it expects that domain.
Retail-style tickets about parcels or app crashes are outside its training
distribution and will be routed on surface vocabulary rather than meaning.

## Beyond the model

- **Model registry with a promotion gate** ([`models/registry.py`](src/automated_support_ticket_classification/models/registry.py), [`models/promote.py`](src/automated_support_ticket_classification/models/promote.py)): a new model only takes the MLflow `production` alias if it beats the current one on macro F1, trained on the same corpus.
- **Drift monitoring** ([`monitoring/drift.py`](src/automated_support_ticket_classification/monitoring/drift.py)): Evidently checks label and text drift, with results in [metrics/drift.json](metrics/drift.json). It stays quiet on train vs test and fires on real vs synthetic data.
- **API plus Prometheus in one command**: `docker compose up` runs the service and a Prometheus instance scraping `/metrics`.
- **Tests and CI**: `pytest` runs 140 tests across the data generators and corpus mapping, the corpus download retry, the model pipeline, the API (including input length limits and the missing-model 503), prediction-log redaction (including Indian identifiers: PAN, UPI, IFSC, Aadhaar) and rotation, drift, the registry promotion gate, and the evaluation statistics (bootstrap CIs, McNemar, the 77-way comparison, ECE). CI runs `ruff format --check .`, `ruff check .`, the tests, a Docker build and a container smoke test.

See [docs/experiments.md](docs/experiments.md) for the experiment log and [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow.

## License

The code is MIT, see [LICENSE](LICENSE).

The banking77 corpus is by PolyAI and is licensed under
[CC BY 4.0](https://github.com/PolyAI-LDN/task-specific-datasets/blob/master/LICENSE).
This repository downloads it rather than storing it, but
`reports/errors.csv` and `docs/mapping/annotation_sheet.csv` contain excerpts
of its messages. It was introduced in:

> Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson and Ivan
> Vulić. 2020. [Efficient Intent Detection with Dual Sentence
> Encoders](https://arxiv.org/abs/2003.04807). In *Proceedings of the 2nd
> Workshop on Natural Language Processing for Conversational AI*.

## Author

**Jetti Raviteja** · [Portfolio](https://jettiraviteja.vercel.app) · [GitHub](https://github.com/raviteja311) · [LinkedIn](https://www.linkedin.com/in/jettiraviteja/)
