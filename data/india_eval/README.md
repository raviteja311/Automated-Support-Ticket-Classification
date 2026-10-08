# India robustness set

Hand-written messages in the style of Indian bank customers, labelled with the
service's five queues. They test whether a model trained on banking77, which
comes from a European digital bank, transfers to UPI, IMPS, NEFT, KYC and
Hinglish.

**Write these by hand.** Do not generate them with an LLM: a set written by
people is the whole point. If you use one for ideas, say so in the project
README.

## Format

`india_messages.csv`, one message per row:

| column | values |
|---|---|
| `text` | the message, as a customer would type it |
| `label` | `billing`, `technical`, `account`, `card_delivery` or `general` |
| `language_style` | `english`, `hinglish` or `short_or_misspelled` |
| `author` | who wrote it |

Quote a message that contains a comma: `"KYC pending, account frozen",account,english,ravi`.

## Targets

- 200 to 300 messages, about 40 per queue
- roughly 60% `english`, 25% `hinglish`, 15% `short_or_misspelled`
- two or three authors if you can, for variety of style

## Run

```bash
dvc repro eval_india
```

Scores land in `metrics/india.json`, overall and by language style, with the
drop against the banking77 test score. Misclassified rows go to
`reports/india_errors.csv`. To see whether the drift monitor notices the shift:

```bash
python -m automated_support_ticket_classification.monitoring.drift --current data/india_eval/india_messages.csv --name drift_india
```
