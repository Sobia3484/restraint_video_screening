"""EDA features. Everything is derived from the screened dataset; no labels or model outputs are changed.

Age group, setting (subtopic), technique flags and language come from keyword rules on the text, so they are
descriptive labels and are not validated against human labels.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from ..screening.rules import _rx
from ..screening.text import fold

DURATION_BINS = [0, 60, 300, 600, 1200, np.inf]
DURATION_LABELS = ["under 1 min", "1-5 min", "5-10 min", "10-20 min", "20+ min"]
FOLLOWER_BINS = [-1, 999, 9_999, 99_999, 999_999, np.inf]
FOLLOWER_LABELS = ["under 1K", "1K-10K", "10K-100K", "100K-1M", "1M+"]

# ---- age cues in the text
NEWBORN = _rx([r"newborn\w*", r"neonat\w*", r"nicu\b", r"pre-?emie\w*", r"premie\w*", r"preterm\w*", r"recien\s*nacid\w*", r"recem[\s-]*nascid\w*"])
INFANT = _rx([r"infant\w*", r"bab(?:y|ies)\w*", r"bebe\w*", r"lactante\w*"])
TODDLER = _rx([r"toddler\w*", r"preschool\w*", r"pre-?school\w*"])
CHILD = _rx([r"child\w*", r"kids?\b", r"p(?:a)?ediatr\w*", r"peds\b", r"paeds\b", r"nin[oa]s?\b", r"crianca\w*", r"enfants?\b"])
OLDER = _rx([r"teen\w*", r"adolescen\w*", r"school[\s-]*age\w*", r"adults?\b", r"adulto\w*", r"elderly", r"patients? of all ages"])
AGE_MONTHS = re.compile(r"(\d+)\s*[- ]?\s*(?:months?|mos?|mths?)\b", re.I)
AGE_YEARS = re.compile(r"(\d+)\s*[- ]?\s*(?:years?|yrs?|yo|y/o)\b", re.I)

# ---- setting / subtopic (first match in the order of SETTING_ORDER wins)
SUB = {
    "trauma_spinal": _rx([r"spinal", r"c-?spine", r"cervical\s+collar", r"c-?collar", r"backboard", r"spine\s*board", r"splint\w*", r"fractur\w*", r"motion\s+restriction", r"\bsmr\b", r"trauma\b"]),
    "dental": _rx([r"dental", r"dentist\w*", r"odontolog\w*", r"pedodont\w*", r"papoose"]),
    "imaging": _rx([r"\bmri\b", r"x-?ray\w*", r"\bct\b", r"radiograph\w*", r"radiolog\w*", r"pigg-?o-?stat", r"resonancia", r"rayos"]),
    "needle_procedures": _rx([r"vaccin\w*", r"immuni[sz]\w*", r"inject\w*", r"needle\w*", r"blood\s+draw\w*", r"venipunct\w*", r"\biv\b", r"cannula\w*", r"\bshots?\b", r"heel\s*(?:stick|prick|lance)", r"phlebotom\w*", r"vacuna\w*", r"inyecc\w*"]),
    "wound_care": _rx([r"laceration\w*", r"suture\w*", r"stitch\w*", r"wound\w*"]),
    "examination": _rx([r"ear\s+(?:check|exam\w*|infection)", r"otoscop\w*", r"throat", r"\bexam(?:ination)?s?\b", r"check-?up", r"swab\w*"]),
    "neonatal": _rx([r"nicu\b", r"newborn\w*", r"neonat\w*", r"preterm\w*", r"pre-?emie\w*", r"premie\w*"]),
    "safety_risks": _rx([r"risks?\b", r"complication\w*", r"guideline\w*", r"polic(?:y|ies)", r"unsafe", r"restraint\s+(?:injur\w*|death\w*)", r"consent", r"assent"]),
    "alternatives": _rx([r"distract\w*", r"child\s+life", r"play\s+(?:specialist|therapy)", r"comfort\s+position\w*", r"preparation", r"non-?restraint"]),
}
SETTING_ORDER = ["trauma_spinal", "dental", "imaging", "needle_procedures", "wound_care", "examination", "neonatal", "safety_risks", "alternatives"]
SETTING_LABELS = {"trauma_spinal": "trauma / spinal / splint", "dental": "dental", "imaging": "imaging (MRI, X-ray, CT)", "needle_procedures": "needle procedures",
                  "wound_care": "wound care", "examination": "examination", "neonatal": "neonatal", "safety_risks": "safety and risks",
                  "alternatives": "alternatives to restraint", "technique_only": "holding / restraint (no setting named)", "unspecified": "not specified in text"}

# ---- technique mentions (independent flags, a video can mention several)
TECH = {
    "tech_clinical_holding": _rx([r"clinical\s+hold\w*", r"therapeutic\s+hold\w*", r"comfort\s+hold\w*", r"holding\s+(?:the\s+)?(?:child|baby|infant|toddler)", r"hold\s+(?:the\s+)?(?:child|baby|infant|toddler)\s+still", r"bear\s*hug"]),
    "tech_swaddling": _rx([r"swaddl\w*", r"wrap(?:ped|ping)?\s+(?:the\s+|your\s+)?(?:baby|infant|child)", r"burrito"]),
    "tech_papoose_device": _rx([r"papoose", r"pigg-?o-?stat", r"immobili[sz]er", r"immobili[sz]ation\s+(?:device|board)", r"restraint\s+(?:device|chair|board)", r"protective\s+stabili[sz]ation", r"vacuum\s+(?:splint|mattress)"]),
    "tech_spinal_motion": _rx([r"spinal\s+motion", r"c-?collar", r"cervical\s+collar", r"backboard", r"spine\s*board", r"scoop\s+stretcher"]),
    "tech_restraint_word": _rx([r"restrain\w*", r"immobili[sz]\w*", r"stabili[sz]ation"]),
    "tech_distraction": _rx([r"distract\w*", r"bubbles", r"child\s+life", r"play\s+(?:specialist|therapy)"]),
    "tech_comfort_position": _rx([r"comfort\s+position\w*", r"upright", r"lap\s+hold", r"skin[\s-]*to[\s-]*skin", r"kangaroo"]),
    "tech_consent_assent": _rx([r"consent", r"assent", r"permission"]),
    "tech_pain_relief": _rx([r"numbing", r"lidocaine", r"\bemla\b", r"sucrose", r"sugar\s+water", r"breast\s*feed\w*", r"anesth\w*", r"anaesth\w*", r"sedat\w*"]),
}
TECH_LABELS = {"tech_clinical_holding": "clinical / comfort holding", "tech_swaddling": "swaddling / wrapping", "tech_papoose_device": "papoose / immobilization device",
               "tech_spinal_motion": "spinal motion restriction", "tech_restraint_word": "words restraint / immobilization / stabilization",
               "tech_distraction": "distraction / child life", "tech_comfort_position": "comfort position / parent hold", "tech_consent_assent": "consent / assent",
               "tech_pain_relief": "pain relief / sedation"}

_STOP = {
    "en": "the and of to is in you for it this that with are on be your how what when can if do not".split(),
    "es": "el la los las de que y en un una para con por como es se su lo al más muy".split(),
    "pt": "o a os as de que e em um uma para com por como é se seu não mais muito você".split(),
    "fr": "le la les des et en un une pour avec par comme est se son pas plus vous que".split(),
    "id": "yang dan di ke dari untuk dengan ini itu pada adalah tidak anda bayi anak".split(),
    "tr": "ve bir bu da de için ile çok daha olan gibi nasıl bebek çocuk".split(),
}


def detect_language(text: str) -> tuple:
    """(language code, method). langdetect when installed, otherwise a script + stop-word heuristic. Short text -> 'und'."""
    t = re.sub(r"https?://\S+|[#@]\w+", " ", text or "")
    letters = sum(ch.isalpha() or unicodedata.category(ch) in ("Mn", "Mc") for ch in t)
    if letters < 25:
        return "und", "too_short"
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(t), "langdetect"
    except ImportError:
        return _heuristic_language(t), "heuristic"
    except Exception:
        return "und", "error"


def _heuristic_language(t: str) -> str:
    counts = {"hi": 0, "ar": 0, "ru": 0, "th": 0, "ko": 0, "ja": 0, "zh": 0, "latin": 0}
    for ch in t:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F: counts["hi"] += 1
        elif 0x0600 <= o <= 0x06FF: counts["ar"] += 1
        elif 0x0400 <= o <= 0x04FF: counts["ru"] += 1
        elif 0x0E00 <= o <= 0x0E7F: counts["th"] += 1
        elif 0xAC00 <= o <= 0xD7AF: counts["ko"] += 1
        elif 0x3040 <= o <= 0x30FF: counts["ja"] += 1
        elif 0x4E00 <= o <= 0x9FFF: counts["zh"] += 1
        elif ch.isalpha(): counts["latin"] += 1
    top = max(counts, key=counts.get)
    if top != "latin" and counts[top] >= counts["latin"]:
        return top
    words = re.findall(r"[^\W\d_]+", t.lower())
    scores = {lang: sum(w in set(sw) for w in words) for lang, sw in _STOP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] >= 2 else "und"


def _mask(rx, texts) -> np.ndarray:
    return np.array([bool(rx.search(t)) for t in texts])


def _stated_age_years(text: str):
    """Age in years stated as a number ('6 months old', '3 year old'); first match, months preferred for babies."""
    m = AGE_MONTHS.search(text)
    if m:
        return int(m.group(1)) / 12
    y = AGE_YEARS.search(text)
    return float(y.group(1)) if y else np.nan


def build_features(df: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    d = df.copy()
    for col in ("title", "description", "hashtags", "tags", "language_clean", "country_clean", "creator_id", "creator_name", "transcript", "transcript_status"):
        d[col] = d[col].fillna("").astype(str) if col in d else ""
    for col in ("duration_seconds", "views", "likes", "comments", "shares", "saves", "creator_followers", "query_count", "transcript_word_count"):
        d[col] = pd.to_numeric(d[col], errors="coerce") if col in d else np.nan
    d["upload_dt"] = pd.to_datetime(d["upload_date"], errors="coerce")
    ref = pd.Timestamp(reference_date) if reference_date is not None else pd.to_datetime(d["retrieved_at_utc"], errors="coerce", utc=True).max()
    ref = ref.tz_localize(None).normalize() if getattr(ref, "tzinfo", None) is not None else (ref.normalize() if pd.notna(ref) else pd.Timestamp.today().normalize())
    d["reference_date"] = ref.date().isoformat()

    d["duration_min"] = d["duration_seconds"] / 60
    d["duration_group"] = pd.cut(d["duration_seconds"], DURATION_BINS, labels=DURATION_LABELS, right=False)
    d["is_short_form"] = d["duration_seconds"] <= 60
    d["upload_year"] = d["upload_dt"].dt.year
    d["upload_month"] = d["upload_dt"].dt.month
    d["upload_year_month"] = d["upload_dt"].dt.strftime("%Y-%m")
    d["age_days"] = (ref - d["upload_dt"]).dt.days.clip(lower=0)

    ok = d["views"] >= 100          # rates of videos with very few views are unstable
    for name, col in (("like_rate", "likes"), ("comment_rate", "comments"), ("share_rate", "shares"), ("save_rate", "saves")):
        d[name] = np.where(ok & d[col].notna(), d[col] / d["views"] * 100, np.nan)
    d["follower_group"] = pd.cut(d["creator_followers"], FOLLOWER_BINS, labels=FOLLOWER_LABELS)
    d["title_chars"] = d["title"].str.len()
    d["title_words"] = d["title"].str.split().str.len()
    d["n_hashtags"] = d["hashtags"].map(lambda s: len([x for x in s.split("|") if x]))
    d["has_description"] = d["description"].str.len() > 0

    # transcript and text availability (important: the screening model sees more for videos that have a transcript)
    d["has_transcript"] = (d["transcript_status"] == "ok") & (d["transcript_word_count"].fillna(0) > 0)
    d["transcript_words"] = np.where(d["has_transcript"], d["transcript_word_count"], 0)
    meta = (d["title"] + " " + d["description"]).str.replace(r"[#@]\w+", " ", regex=True)
    d["text_words"] = meta.str.split().str.len().fillna(0)
    d["no_usable_text"] = (d["text_words"] + d["transcript_words"]) < 6

    texts = [fold(f"{t} {h.replace('|', ' ')} {g.replace('|', ' ')} {b}") for t, h, g, b in zip(d["title"], d["hashtags"], d["tags"], d["description"])]
    new, inf, tod, chi, old = (_mask(rx, texts) for rx in (NEWBORN, INFANT, TODDLER, CHILD, OLDER))
    stated = np.array([_stated_age_years(t) for t in texts])
    d["stated_age_years"] = stated
    young_num = (stated <= 5) & ~np.isnan(stated)
    old_num = (stated >= 6) & ~np.isnan(stated)
    d["age_group"] = np.select(
        [old | old_num, new, inf | (young_num & (stated < 1)), tod | (young_num & (stated >= 1)), chi],
        ["older child / adult mentioned", "newborn / neonatal", "infant", "toddler / preschool", "child (age not given)"], default="not stated in text")

    for k, rx in SUB.items():
        d[f"sub_{k}"] = _mask(rx, texts)
    prim = np.full(len(d), "unspecified", dtype=object)
    for k in reversed(SETTING_ORDER):
        prim = np.where(d[f"sub_{k}"].to_numpy(), k, prim)
    tw = _mask(TECH["tech_restraint_word"], texts) | _mask(TECH["tech_clinical_holding"], texts) | _mask(TECH["tech_swaddling"], texts) | _mask(TECH["tech_papoose_device"], texts)
    prim = np.where((prim == "unspecified") & tw, "technique_only", prim)
    d["subtopic"] = [SETTING_LABELS[x] for x in prim]
    for k, rx in TECH.items():
        d[k] = _mask(rx, texts)

    lang_text = np.where(d["platform"] == "youtube", (d["title"] + ". " + d["description"].str[:600]).str.strip(), d["title"])
    det = [detect_language(t) for t in lang_text]
    d["language_detected"] = [x[0] for x in det]
    d["language_method"] = [x[1] for x in det]
    d["non_english_text"] = ~d["language_detected"].isin(["en", "und"])
    return d.drop(columns=["upload_dt"])
