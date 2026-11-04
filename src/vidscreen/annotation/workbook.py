"""Annotator workbooks (xlsx): what annotators see, and reading their answers back."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

LABEL_VALUES = ["Relevant", "Irrelevant", "Unsure"]
COND_VALUES = ["Yes", "No", "Unclear"]
UNSURE_REASONS = ["age_unclear", "topic_unclear", "content_unclear", "language", "video_unavailable", "other"]
YES_NO = ["Yes", "No"]

# columns the annotator SEES (nothing about how the video was found, no engagement numbers)
SHOWN_COLUMNS = [
    "item_id", "platform", "url", "title", "description", "hashtags", "transcript",
    "creator_name", "duration", "upload_date", "platform_language",
]
# columns the annotator FILLS
FILL_COLUMNS = ["label", "age_ok", "topic_ok", "content_ok", "unsure_reason", "unsafe_demo", "note"]

TRANSCRIPT_LIMIT = 8000
_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

INSTRUCTIONS = [
    "ANNOTATION INSTRUCTIONS (full rules: docs/screening_criteria.md)",
    "",
    "Label = Relevant only if ALL three are true:",
    "  A. Age: the child is an infant or young child (birth to about 5 years). A stated age overrides appearance.",
    "  B. Topic: the video is about holding, restricting or immobilizing a child's movement for care",
    "     (clinical/comfort holding, swaddling or wrapping for a procedure, papoose board, immobilization device,",
    "     spinal motion restriction, protective stabilization, alternatives to restraint).",
    "     Test: could the child move away freely? If yes, it is only positioning -> not Relevant.",
    "  C. Content: the video shows or explains how it is done, or how to do it safely.",
    "Label = Irrelevant if any condition fails, or for: restraint collapse (parenting term), music/songs/games,",
    "  car seats, sleep swaddling, adult/martial-arts/animal restraint, behavioral restraint with no care procedure.",
    "Label = Unsure if you cannot decide after the watching procedure. Always give a reason and a note.",
    "",
    "Watching procedure: read the text first. Up to 3 min: watch fully. 3-20 min: first 60 s, then about 25%, 50%, 75%.",
    "Over 20 min: first 2 min and the same checkpoints. Never label from title, hashtags or thumbnail alone.",
    "",
    "Non-English videos: use machine translation for the text and judge from what you can see. If still unclear -> Unsure, reason = language.",
    "unsafe_demo = Yes if the video shows or recommends an unsafe practice (airway/chest blocked, child left alone, needless force),",
    "  even when the video is Relevant as safety education. Add a note.",
    "",
    "Columns to fill: label, age_ok, topic_ok, content_ok (Yes/No/Unclear), unsure_reason, unsafe_demo, note.",
    "Work alone. Do not discuss labels with the other annotator before agreement is computed.",
    "Save the file in the same folder under the same name.",
]


def _clean(v) -> str:
    return _ILLEGAL.sub("", "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))


def _fmt_duration(sec) -> str:
    try:
        s = int(float(sec))
    except (TypeError, ValueError):
        return ""
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def annotator_view(key: pd.DataFrame) -> pd.DataFrame:
    """Reduce the key table to the columns annotators may see."""
    tr = key["transcript"].fillna("").astype(str)
    cut = tr.where(tr.str.len() <= TRANSCRIPT_LIMIT, tr.str[:TRANSCRIPT_LIMIT] + " [... transcript truncated]")
    view = pd.DataFrame({
        "item_id": key["item_id"],
        "platform": key["platform"],
        "url": key["url"],
        "title": key["title"],
        "description": key["description"],
        "hashtags": key["hashtags"].fillna("").astype(str).str.replace("|", " ", regex=False),
        "transcript": cut,
        "creator_name": key["creator_name"],
        "duration": key["duration_seconds"].map(_fmt_duration),
        "upload_date": key["upload_date"],
        "platform_language": key["language_clean"],
    })
    return view[SHOWN_COLUMNS]


def write_annotator_workbook(key: pd.DataFrame, path: Path, order_seed: int) -> None:
    """Write one workbook. Rows are shuffled with ``order_seed`` (different per annotator)."""
    view = annotator_view(key).sample(frac=1.0, random_state=order_seed).reset_index(drop=True)
    wb = Workbook()
    ws0 = wb.active
    ws0.title = "Instructions"
    for i, line in enumerate(INSTRUCTIONS, start=1):
        ws0.cell(row=i, column=1, value=line)
    ws0["A1"].font = Font(bold=True, size=13)
    ws0.column_dimensions["A"].width = 130

    ws = wb.create_sheet("Annotation")
    headers = SHOWN_COLUMNS + FILL_COLUMNS
    ws.append(headers)
    head_fill = PatternFill("solid", fgColor="DDEBF7")
    fill_fill = PatternFill("solid", fgColor="FFF2CC")
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=c)
        cell.font = Font(bold=True)
        cell.fill = fill_fill if h in FILL_COLUMNS else head_fill
    for r, row in enumerate(view.itertuples(index=False), start=2):
        for c, v in enumerate(row, start=1):
            cell = ws.cell(row=r, column=c)
            cell.value = _clean(v)
            cell.data_type = "s"  # text only: a title starting with "=" must not become a formula
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    widths = {"item_id": 9, "platform": 10, "url": 30, "title": 40, "description": 50, "hashtags": 25,
              "transcript": 60, "creator_name": 18, "duration": 9, "upload_date": 12, "platform_language": 10,
              "label": 12, "age_ok": 9, "topic_ok": 9, "content_ok": 10, "unsure_reason": 16,
              "unsafe_demo": 11, "note": 40}
    for c, h in enumerate(headers, start=1):
        ws.column_dimensions[ws.cell(row=1, column=c).column_letter].width = widths.get(h, 14)
    ws.freeze_panes = "B2"
    last = len(view) + 1

    def add_list(col_name: str, values: List[str]) -> None:
        col = ws.cell(row=1, column=headers.index(col_name) + 1).column_letter
        dv = DataValidation(type="list", formula1='"' + ",".join(values) + '"', allow_blank=True, showErrorMessage=True)
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{last}")

    add_list("label", LABEL_VALUES)
    for cond in ("age_ok", "topic_ok", "content_ok"):
        add_list(cond, COND_VALUES)
    add_list("unsure_reason", UNSURE_REASONS)
    add_list("unsafe_demo", YES_NO)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def read_annotations(path: Path) -> pd.DataFrame:
    """Read a filled workbook: item_id + FILL_COLUMNS (strings, blanks as '')."""
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["Annotation"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    header = [str(h) for h in rows[0]]
    df = pd.DataFrame(rows[1:], columns=header).fillna("")
    keep = ["item_id"] + FILL_COLUMNS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    df = df[keep].astype(str).apply(lambda s: s.str.strip())
    return df[df["item_id"] != ""].reset_index(drop=True)
