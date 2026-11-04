# Safe Restraint and Immobilization Techniques for Infants and Young Children: Video Screening Pipeline

A reproducible pipeline that collects YouTube and TikTok videos about **safe restraint and immobilization techniques for infants and young children** (medical procedures, imaging, trauma and dental care), cleans them, annotates a sample by hand, screens the whole collection with a conservative machine-learning model, and assesses the relevant videos against expert clinical guidelines.

Scope is medical, dental, imaging and trauma care. Vehicle child restraints (car seats, harnesses) are out of scope. All search queries are in English.

---

## Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Results at a glance](#2-results-at-a-glance)
3. [Repository structure](#3-repository-structure)
4. [Setup](#4-setup)
5. [Running the pipeline](#5-running-the-pipeline)
6. [Stage details](#6-stage-details)
7. [Output files](#7-output-files)
8. [Limitations](#8-limitations)
9. [Documentation and references](#9-documentation-and-references)

---

## 1. Pipeline overview

```
 25 English queries
        |
 [1] Extraction ........ YouTube (yt-dlp) + TikTok (ScrapeCreators), one CSV per query
        |
 [2] Preprocessing ..... merge -> deduplicate -> clean        -> master_cleaned.csv (2,319 videos)
        |
 [3] Annotation ........ pilot (agreement check) -> 400 stratified videos labeled by humans
        |
 [4] Screening ......... text model trained on the human labels, strict thresholds
        |                -> relevant (strict) / audit / irrelevant
        |
 [5] Expert checklist .. relevant videos assessed against guideline-based points
```

Every stage reads the output of the previous one and can be re-run on its own. Extraction caches everything it fetches, so interrupted runs resume without repeating cost.

## 2. Results at a glance

| Stage | Result |
|---|---|
| Extraction and cleaning | 2,319 unique videos (TikTok 1,194; YouTube 1,125) |
| Pilot annotation | Cohen's kappa 0.815 (target 0.70): passed |
| Main annotation | 400 videos labeled (TikTok 200, YouTube 200); kappa 0.612 on the double-annotated subset, with a third adjudicator for disagreements |
| Screening model | TF-IDF + logistic regression, creator-grouped cross-validation, AUC about 0.78 |
| Screening output | **Relevant (strict): 160** (YouTube 95, TikTok 65), **Irrelevant: 389**, **Audit (needs review): 1,770** |
| Expert checklist | 13 guideline points; 50 relevant videos with model score >= 0.90, plus a 30-video TikTok comparison sample |

Of the 160 strict relevant videos, 104 are human-labeled and 56 are model-selected. The strict set is a **high-confidence subset**, not the full set of relevant videos (see [Limitations](#8-limitations)).

## 3. Repository structure

```
restraint_video_screening/
├── config/
│   ├── queries.csv                 25 English search queries (query_id, query, subtopic)
│   └── settings.yaml               paths, YouTube/TikTok limits, transcript settings
├── data/
│   ├── raw/                        one CSV per query (+ _manifest.csv)
│   ├── cache/                      saved API responses, metadata, transcripts
│   ├── intermediate/               master_raw, master_deduplicated, master_cleaned, query_video_map, dropped_records
│   ├── annotation/                 pilot/ and main/ : keys and annotator workbooks
│   └── final/
│       ├── labeled_set.csv         human labels (400)
│       ├── screening/              model results, report, relevant / irrelevant / audit lists
│       └── checklist/              expert-guideline checklist workbooks
├── docs/
│   ├── screening_criteria.md/.docx labeling criteria, grounded in published guidelines
│   ├── annotation_workflow.md      how annotators work, step by step
│   ├── extraction_design.md        tools, caching, reliability
│   └── data_dictionary.md          every column in every file
├── scripts/                        command-line entry points (section 5)
├── src/vidscreen/                  library code: extractors, preprocessing, annotation, screening
├── tests/                          unit tests (no internet needed)
├── .env.example                    copy to .env and add the TikTok API key
└── requirements.txt
```

## 4. Setup

Python 3.10 or newer. Windows PowerShell, inside the project folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and set `SCRAPECREATORS_API_KEY=your_key_here`. Never share this file. On macOS or Linux use `source .venv/bin/activate` and `cp .env.example .env`.

| Job | Tool | Requirement |
|---|---|---|
| YouTube search and metadata | `yt-dlp` | none |
| YouTube transcripts | `youtube-transcript-api` | none |
| TikTok search, metadata, captions | ScrapeCreators API | API key in `.env` |
| Tables, modeling | `pandas`, `scikit-learn`, `openpyxl` | |
| Optional embeddings | `sentence-transformers` | downloads a model on first use |

Run the unit tests before anything else (no internet, no credits):

```powershell
python -m unittest discover -s tests -t .
```

## 5. Running the pipeline

Run the commands in order, from the project folder. Each command is one line.

```powershell
# 1. Extraction
python scripts/smoke_test.py                                   # 1 query, quick check
python scripts/run_extraction.py                               # 25 queries, YouTube + TikTok
python scripts/backfill_transcripts.py --cooldown-minutes 20   # only if YouTube transcripts were blocked

# 2. Preprocessing
python scripts/run_preprocessing.py                            # merge -> dedupe -> clean

# 3. Annotation (pilot first, then the main sample)
python scripts/make_annotation_sample.py --stage pilot
python scripts/annotation_agreement.py --stage pilot
python scripts/make_annotation_sample.py --stage main
python scripts/annotation_agreement.py --stage main
python scripts/build_labeled_set.py                            # merges labels + adjudication -> labeled_set.csv

# 4. Screening
python scripts/run_screening.py                                # add --no-embeddings for a faster run

# 5. Expert-guideline checklist
# 5a. the strict relevant videos with model score >= 0.90
python scripts/build_expert_checklist.py data/final/screening/relevant_strict_videos.csv data/final/checklist/Restraint_Immobilization_Expert_Checklist.xlsx --min-score 0.90

# 5b. random TikTok comparison sample: 15 irrelevant + 15 audit videos
python scripts/make_comparison_sample.py --n 15 --platform tiktok
python scripts/build_expert_checklist.py data/final/checklist/comparison_sample.csv data/final/checklist/Restraint_Checklist_Comparison_Sample.xlsx
```

Useful options:

| Script | Option | Meaning |
|---|---|---|
| `run_extraction.py` | `--platform youtube\|tiktok`, `--queries Q001 Q014`, `--limit 3`, `--no-transcripts`, `--force` | restrict or rebuild |
| `run_screening.py` | `--stage cv`, `--precision-target`, `--high-floor`, `--zone-low`, `--repeats`, `--folds` | cross-validation only, threshold control |
| `make_annotation_sample.py` | `--n`, `--n-per-platform`, `--double-frac`, `--min-per-stratum`, `--seed` | sample sizes |
| `make_comparison_sample.py` | `--n`, `--platform tiktok\|youtube\|both`, `--seed` | comparison sample |

Re-running is safe: finished queries are skipped and everything fetched is cached.

## 6. Stage details

### 6.1 Extraction
Each of the 25 queries is searched on YouTube and TikTok. A video is stored with title, description, hashtags, creator, counts, duration, upload date, language, country (TikTok only) and transcript when available. Queries are grouped into subtopics (general, clinical holding, needle procedures, imaging, trauma and spinal, dental, safety risks, alternatives, neonatal, examination, wound care). See `docs/extraction_design.md`.

### 6.2 Preprocessing
`run_preprocessing.py` merges the per-query files, deduplicates videos found by several queries (keeping a query-to-video map), and cleans fields. Every dropped video is listed with its reason in `data/intermediate/dropped_records.csv`.

### 6.3 Annotation
Criteria in `docs/screening_criteria.md` define **Relevant** with three conditions: (A) the child is an infant or young child (birth to about 5 years), (B) the topic is holding, restricting or immobilizing movement for care, (C) the video shows or explains how it is done or how to do it safely. Positioning only, sleep swaddling, restraint collapse, music, car seats and non-medical restraint are excluded.

* **Pilot:** about 40 diverse videos labeled independently by two annotators; agreement must reach kappa 0.70 before the main stage.
* **Main sample:** 400 videos, stratified by platform and subtopic, with sample weights stored so performance can be estimated on the full collection. A random 20% is labeled by both annotators.
* **Adjudication:** a third person resolves disagreements and every Unsure item.
* Query text, search rank and engagement counts are hidden from annotators to avoid bias.

### 6.4 Screening
A strict text classifier trained on the 400 human labels:

* **Features:** TF-IDF (word and character n-grams) on title, description, hashtags and transcript, optionally combined with multilingual sentence embeddings.
* **Model:** logistic regression with balanced class weights (interpretable, stable on a small labeled set).
* **Evaluation:** repeated cross-validation grouped by creator, so videos from one creator never appear in both training and test folds; ablation across representation, feature set, rule gate and transcript use; Platt calibration.
* **Strict threshold:** chosen from out-of-fold predictions so that weighted precision reaches 0.90, with a floor of 0.80, to keep false positives low.
* **Zones:** `relevant_confident` (above the strict threshold), `irrelevant_confident` (below a data-driven low threshold) and `audit` (in between, for review). Human-labeled videos keep their human label.

The full methodology and numbers are written to `data/final/screening/screening_report.md`.

### 6.5 Expert-guideline checklist
Relevant videos are assessed against **13 points** derived from published guidance: indication and least restrictive option, preparation, distraction and pain relief, consent and assent, upright position for injections, parent comfort hold, swaddling, minimum force and time, airway and chest not obstructed, monitoring, correct device use, spinal motion restriction, and risks stated.

Each cell is **✓** (shown or said correctly), **✗** (contradicts the guideline), **?** (not verified) or **N/A** (does not apply to this video type). Coverage is the share of applicable points ticked. An automatic keyword pass pre-fills the workbook from the video text and stores the matching snippet in the Evidence sheet; the reviewer then watches the videos and corrects the grid in `Checklist_Detail`.

Workbook sheets: `Checklist_Detail` (the grid to review), `Format` (compact view), `Summary` (live counts and averages), `Checklist_Guide` (points and sources), `Evidence`, `Method_and_Limits`.

## 7. Output files

| File | Description |
|---|---|
| `data/intermediate/master_cleaned.csv` | 2,319 cleaned unique videos |
| `data/final/labeled_set.csv` | 400 human labels with sample weights |
| `data/final/screening/screening_results.csv` | every video with score, zone and label |
| `data/final/screening/relevant_strict_videos.csv` | 160 strict relevant videos |
| `data/final/screening/irrelevant_videos.csv` | 389 irrelevant videos |
| `data/final/screening/audit_videos.csv` | 1,770 videos needing review |
| `data/final/screening/screening_report.md/.json` | methods, model comparison, calibration, thresholds |
| `data/final/checklist/*.xlsx` | expert-guideline checklist workbooks |

Column definitions are in `docs/data_dictionary.md`.

## 8. Limitations

* **The strict set is a subset.** The model favors videos whose text names the technique explicitly. Videos that show restraint mainly visually are less likely to pass, and the audit zone holds an estimated several hundred more relevant videos. Percentages from the strict set describe that subset, not the platforms.
* **Scores of human-labeled videos are in-sample.** The model saw those labels in training, so their score says little; they are relevant by annotation.
* **Label noise.** Agreement on the main sample was moderate (kappa 0.612), which bounds what any model can learn.
* **Text only.** Screening and the automatic checklist pass read titles, descriptions, hashtags and transcripts; the video content itself is not analyzed. Few TikTok videos have transcripts, and non-English text is under-served by the keyword rules.
* **No single guideline covers the topic.** The criteria and checklist points combine nursing, dental, sedation, vaccination and trauma sources; the mapping to video labels is the project's own interpretation. The dental source was read from its abstract only.
* **Guidelines describe professional practice.** A parent video can be sound without citing a guideline; the criteria judge what the video shows.

## 9. Documentation and references

Project documents are in `docs/`. Main sources behind the criteria and checklist:

1. Royal College of Nursing (2019). *Restrictive physical intervention and the clinical holding of children and young people.*
2. American Academy of Pediatric Dentistry. *Use of Protective Stabilization for Pediatric Dental Patients* (Reference Manual, rev. 2024).
3. AAP and AAPD (2025). *Guidelines for Monitoring and Management of Pediatric Patients Before, During, and After Sedation.* Pediatr Dent 47(6):E100-E128.
4. Taddio A et al. (2010). *Reducing the pain of childhood vaccination: an evidence-based clinical practice guideline.* CMAJ 182(18):1989.
5. ACS COT, ACEP, NAEMSP (2018). *Spinal Motion Restriction in the Trauma Patient: a joint position statement.*
6. Two 2024 scoping reviews on procedural restraint in children (J Pediatr Nurs 2024;79:e110-e118; J Child Health Care 2024).

Full citations and links are in `docs/screening_criteria.md`, section 13.
