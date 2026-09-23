# Automated Support Ticket Classification

[![CI](https://github.com/raviteja311/Automated-Support-Ticket-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/raviteja311/Automated-Support-Ticket-Classification/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end MLOps service that classifies customer support tickets into
billing, technical, account, shipping, and general categories.

## Run locally

    .\MLOps\Scripts\Activate.ps1
    uvicorn automated_support_ticket_classification.api.app:app --reload

Then open http://127.0.0.1:8000/docs

## Call the API from PowerShell

    Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
      -ContentType "application/json" -Body '{"text":"my invoice is wrong"}'

Response:

    label       confidence  all_scores
    -----       ----------  ----------
    billing     0.938       @{billing=0.938; technical=0.018; ...}

## Reproduce the pipeline

    dvc repro          # generate -> preprocess -> train -> evaluate
    dvc metrics show   # accuracy / f1

`dvc.yaml` invokes `python`, so run it from the activated environment or the
stages fail with `ModuleNotFoundError`.

## Tests

    pytest

## Experiments

See [docs/experiments.md](docs/experiments.md).
