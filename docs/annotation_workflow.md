# Human annotation workflow

Criteria: `docs/screening_criteria.md`. Input: `data/intermediate/master_cleaned.csv` (final, after the Q014 re-run, transcript backfill and preprocessing).

| Step | Command | Result |
|---|---|---|
| 1. Pilot sample | `python scripts/make_annotation_sample.py --stage pilot` | 40 diverse videos; `pilot_annotator_A.xlsx` and `_B.xlsx` (same videos, different order); `pilot_key.csv` (private) |
| 2. Pilot labeling | Annotators A and B fill their xlsx independently and save with the same name | |
| 3. Pilot agreement | `python scripts/annotation_agreement.py --stage pilot` | kappa vs target 0.70; `pilot_disagreements.csv` to discuss. Refine the criteria; if below target, repeat with `--force --seed 43` |
| 4. Main sample | `python scripts/make_annotation_sample.py --stage main` | 400 videos (200 per platform), stratified; B gets only the 20% double subset |
| 5. Main labeling | A labels all, B labels the double subset | |
| 6. Agreement + adjudication list | `python scripts/annotation_agreement.py --stage main` | kappa on the double subset; `main_disagreements.csv` |
| 7. Adjudication | Third adjudicator fills `final_label` (and name) in `main_disagreements.csv` | |
| 8. Final labeled set | `python scripts/build_labeled_set.py` | `data/final/labeled_set.csv` |

Give annotators only their own `*_annotator_X.xlsx`. The `*_key.csv` files hold query ids, engagement and subtopics and must stay with the project owner.
The scripts refuse to overwrite an existing key (this would orphan labels) unless `--force` is used.
