# Mapping validation

banking77 labels customer messages with 77 fine-grained intents. This service
routes them to 5 support queues, so each intent has to be assigned to one queue.
That assignment was made by one person. This folder measures whether other
people would make the same calls.

## For annotators

Open `annotation_sheet.csv` in Excel or Google Sheets. Each row is one intent
with three real example messages. Fill in the `queue` column with exactly one
of the five queue names below, then save the file as CSV.

| queue | use it when |
|---|---|
| `billing` | money moved wrongly, or a charge needs explaining or reversing |
| `technical` | something did not work: declined, failed, unrecognised by a device |
| `account` | identity, credentials, limits, and the account's lifecycle |
| `card_delivery` | a physical card needs to arrive |
| `general` | an informational question with no fault to fix |

The question to ask for every row: **which team would need to act on this
message?**

Please:

- work alone, and do not discuss rows with anyone until you have finished
- do not look at the project's code, which contains the existing answers
- decide from the intent name and the examples; if two queues seem to fit,
  pick the one you think is better rather than leaving it blank
- expect about 20 to 30 minutes for all 77 rows

## For the project author

1. Send `annotation_sheet.csv` and this README to each annotator.
2. Save each completed sheet as `annotations/<name>.csv`.
3. Score agreement and write the majority-vote mapping:

   ```bash
   python -m automated_support_ticket_classification.data.mapping_validation agree
   ```

4. Retrain under the majority-vote mapping and measure the change:

   ```bash
   python -m automated_support_ticket_classification.data.mapping_validation sensitivity
   ```

Both commands write to `metrics/mapping_agreement.json`. Write up the result,
including which intents were disputed and why, in `docs/mapping-validation.md`.
