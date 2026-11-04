"""Strict relevance screening.

Same design as the earlier screening work (creator-grouped repeated CV, candidate comparison, ablation, Platt calibration,
zones) with three strictness changes:
  * gate options:  none | veto (only blocks videos with a false-match signal) | strict (topic AND child term, no false match)
  * text source ablation: title/description/hashtags/tags  vs  the same plus the (truncated) transcript
  * decision threshold = lowest calibrated probability (never below HIGH_FLOOR) at which the SAMPLE-WEIGHTED out-of-fold
    precision reaches PRECISION_TARGET; the model never labels a video relevant below it
  * only 'relevant_confident' (+ human Relevant) goes to the strict dataset; the audit zone is NOT auto-included

Inputs: title, description, hashtags, tags, simple metadata and (in the ablation) the transcript. NOT used: views, likes, query information.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from ..preprocessing.merge import read_csv_str
from .evaluate import bootstrap_ci, metrics, recall_at_precision
from .features import NUMERIC, add_features
from .models import EMBEDDING_MODEL, encode_texts
from .rules import apply_rules
from .text import build_text

log = logging.getLogger("vidscreen")
SEED = 42
TIE = 0.01
ZONE_LOW = 0.15            # calibrated probability <= this: irrelevant_confident
HIGH_FLOOR = 0.80          # the strict threshold is never lower than this
PRECISION_TARGET = 0.90    # weighted out-of-fold precision required at the strict threshold
GATES = ("none", "veto", "strict")
SOURCES = ("meta", "meta+transcript")
TRANSCRIPT_CHARS = 1500
RULE_TO_BINARY = {"relevant": 1, "potentially_relevant": 1, "irrelevant": 0}


# ---------------------------------------------------------------- data
def load_frame(settings, labeled_path: Optional[Path] = None) -> pd.DataFrame:
    master = read_csv_str(settings.intermediate_dir / "master_cleaned.csv")
    for c in ("tags", "hashtags", "description", "title", "creator_id", "transcript"):
        if c not in master:
            master[c] = ""
    lp = Path(labeled_path) if labeled_path else settings.final_dir / "labeled_set.csv"
    lab = read_csv_str(lp)
    keep = [c for c in ("video_uid", "label_final", "sample_weight", "stratum", "unsafe_demo", "label_source") if c in lab.columns]
    df = master.merge(lab[keep], on="video_uid", how="left", validate="one_to_one").fillna({"label_final": ""})
    df["text"] = df.apply(build_text, axis=1)
    df["text_tr"] = (df["text"] + " " + df["transcript"].str.replace(r"\s+", " ", regex=True).str[:TRANSCRIPT_CHARS]).str.strip()
    res = [apply_rules(r.title, r.description, r.hashtags, r.tags) for r in df.itertuples()]
    df["rule_label"] = [r.label for r in res]
    df["rule_reason"] = [r.reason for r in res]
    df["gate_topic"] = [1.0 if r.gate_topic else 0.0 for r in res]
    df["gate_strict"] = [1.0 if r.gate_strict else 0.0 for r in res]
    df["gate_veto"] = [0.0 if r.blocked else 1.0 for r in res]
    df["y"] = df["label_final"].map({"Relevant": 1, "Irrelevant": 0})
    df["w"] = pd.to_numeric(df.get("sample_weight", 1.0), errors="coerce").fillna(1.0)
    return df


@dataclass(frozen=True)
class Spec:
    rep: str      # tfidf | embed
    feats: bool   # False = text only, True = text + numeric features
    scope: str    # combined | per_platform
    gate: str     # none | veto | strict
    src: str = "meta"   # meta | meta+transcript

    @property
    def name(self) -> str:
        return f"{self.rep}|{'text+features' if self.feats else 'text'}|{self.scope}|gate_{self.gate}|{self.src}"

    @property
    def complexity(self) -> int:
        return (int(self.rep == "embed") + int(self.feats) + int(self.scope == "per_platform") + int(self.gate != "none")
                + int(self.src != "meta"))


class Data:
    def __init__(self, df: pd.DataFrame, emb: Optional[dict]):
        self.texts = {"meta": df["text"].to_numpy(), "meta+transcript": df["text_tr"].to_numpy()}
        self.num = df[NUMERIC].to_numpy(dtype=float)
        self.platform = df["platform"].to_numpy()
        self.gates = {"none": np.ones(len(df)), "veto": df["gate_veto"].to_numpy(float), "strict": df["gate_strict"].to_numpy(float)}
        self.emb = emb


class Model:
    def __init__(self, rep: str, feats: bool, data: Data, src: str = "meta", seed: int = SEED):
        self.rep, self.feats, self.data, self.src, self.seed = rep, feats, data, src, seed
        self.const = None

    def _tfidf(self, texts, fit: bool):
        if fit:
            for min_df in (2, 1):
                try:
                    self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True).fit(texts)
                    self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=min_df, sublinear_tf=True).fit(texts)
                    break
                except ValueError:
                    continue
        return [self.w.transform(texts), self.c.transform(texts)]

    def _X(self, idx, fit: bool):
        parts = self._tfidf(self.data.texts[self.src][idx], fit) if self.rep == "tfidf" else [sparse.csr_matrix(self.data.emb[self.src][idx])]
        if self.feats:
            if fit:
                self.sc = StandardScaler().fit(self.data.num[idx])
            parts.append(sparse.csr_matrix(self.sc.transform(self.data.num[idx])))
        return sparse.hstack(parts).tocsr()

    def fit(self, idx, y):
        y = np.asarray(y).astype(int)
        if len(set(y)) < 2:
            self.const = float(y.mean())
            return self
        self.lr = LogisticRegression(C=4.0 if self.rep == "tfidf" else 1.0, class_weight="balanced", max_iter=4000,
                                     random_state=self.seed).fit(self._X(idx, True), y)
        return self

    def predict(self, idx):
        if self.const is not None:
            return np.full(len(idx), self.const)
        return self.lr.predict_proba(self._X(idx, False))[:, 1]


def fit_predict(spec: Spec, data: Data, train_idx, y_train, test_idx) -> np.ndarray:
    train_idx, test_idx, y_train = np.asarray(train_idx), np.asarray(test_idx), np.asarray(y_train)
    if spec.scope == "combined":
        return Model(spec.rep, spec.feats, data, spec.src).fit(train_idx, y_train).predict(test_idx)
    out = np.zeros(len(test_idx))
    for p in np.unique(data.platform):
        tr, te = data.platform[train_idx] == p, data.platform[test_idx] == p
        if te.any():
            out[te] = Model(spec.rep, spec.feats, data, spec.src).fit(train_idx[tr], y_train[tr]).predict(test_idx[te])
    return out


def base_specs(use_embeddings: bool) -> list:
    reps = ["tfidf"] + (["embed"] if use_embeddings else [])
    return [Spec(r, f, s, g, src) for r, f, s, g, src in product(reps, (False, True), ("combined", "per_platform"), GATES, SOURCES)]


# ---------------------------------------------------------------- cross-validation and comparison
def grouped_cv(df, data, lab_idx, specs, repeats=3, folds=5, seed=SEED) -> dict:
    """Mean out-of-fold probability per labeled video over `repeats` creator-grouped, stratified CV runs."""
    y = df["y"].to_numpy()[lab_idx].astype(int)
    cid = df["creator_id"].where(df["creator_id"] != "", df["video_uid"])
    groups = (df["platform"] + ":" + cid).to_numpy()[lab_idx]
    strat = df["platform"].to_numpy()[lab_idx] + y.astype(str)
    base = sorted({(s.rep, s.feats, s.scope, s.src) for s in specs})
    acc = {b: np.zeros(len(lab_idx)) for b in base}
    cnt = np.zeros(len(lab_idx))
    for r in range(repeats):
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        for tr, te in cv.split(np.zeros(len(y)), strat, groups):
            for b in base:
                acc[b][te] += fit_predict(Spec(b[0], b[1], b[2], "none", b[3]), data, lab_idx[tr], y[tr], lab_idx[te])
            cnt[te] += 1
    oof = {}
    for s in specs:
        p = acc[(s.rep, s.feats, s.scope, s.src)] / np.maximum(cnt, 1)
        oof[s] = p * data.gates[s.gate][lab_idx]
    return oof


def best_f1_threshold(y, p) -> float:
    best, bt = -1.0, 0.5
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        f = metrics(y, p >= t)["f1"]
        if f > best + 1e-12:
            best, bt = f, float(t)
    return bt


def _auc(y, p):
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def evaluate_candidates(y, platform, w, oof: dict, target: float = PRECISION_TARGET) -> pd.DataFrame:
    rows = []
    for s, p in oof.items():
        t = best_f1_threshold(y, p)
        m = metrics(y, p >= t)
        row = {"candidate": s.name, "rep": s.rep, "features": "text+features" if s.feats else "text", "scope": s.scope, "gate": s.gate, "src": s.src,
               "complexity": s.complexity, "auc": round(_auc(y, p), 4), "pr_auc": round(float(average_precision_score(y, p)), 4),
               "brier": round(float(brier_score_loss(y, np.clip(p, 0, 1))), 4), "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
               f"recall_at_precision_{int(target * 100)}": round(recall_at_precision(y, p, target, w), 4)}
        aucs = []
        for pl in np.unique(platform):
            k = platform == pl
            a = _auc(y[k], p[k])
            aucs.append(a)
            mm = metrics(y[k], p[k] >= t)
            row.update({f"{pl}_auc": round(a, 4), f"{pl}_precision": mm["precision"], f"{pl}_recall": mm["recall"], f"{pl}_f1": mm["f1"]})
        row["macro_auc"] = round(float(np.nanmean(aucs)), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)


def select(table: pd.DataFrame) -> str:
    """Highest macro (per-platform) AUC; candidates within 0.01 are tied -> least complex, then higher AUC."""
    top = table["macro_auc"].max()
    tied = table[table["macro_auc"] >= top - TIE]
    return tied.sort_values(["complexity", "macro_auc"], ascending=[True, False]).iloc[0]["candidate"]


def ablation_summary(table: pd.DataFrame) -> list:
    """Mean change in macro AUC when one design choice is switched, over otherwise identical candidate pairs."""
    cols = ["rep", "features", "scope", "gate", "src"]
    out = []
    for factor, a, b, label in (("features", "text", "text+features", "add duration/length/signal features"),
                                ("gate", "none", "veto", "soft gate: only block videos with a false-match signal"),
                                ("gate", "veto", "strict", "tighten gate: require holding term and child term"),
                                ("gate", "none", "strict", "strict gate instead of no gate"),
                                ("src", "meta", "meta+transcript", "add transcript text"),
                                ("scope", "combined", "per_platform", "separate TikTok/YouTube models instead of one"),
                                ("rep", "tfidf", "embed", "pretrained embeddings instead of TF-IDF")):
        piv = table.pivot_table(index=[c for c in cols if c != factor], columns=factor, values="macro_auc")
        if a in piv.columns and b in piv.columns:
            d = (piv[b] - piv[a]).dropna()
            out.append({"change": label, "mean_macro_auc_change": round(float(d.mean()), 4), "pairs": int(len(d)), "pairs_improved": int((d > 0).sum())})
    return out


# ---------------------------------------------------------------- calibration and strict thresholds
def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.01, 0.99)
    return np.log(p / (1 - p))


def fit_calibrator(p, y) -> LogisticRegression:
    """Platt scaling on out-of-fold scores (monotone: ranking unchanged)."""
    return LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), np.asarray(y).astype(int))


def apply_calibrator(cal, p) -> np.ndarray:
    return cal.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def calibration(y, p, bins=5) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = [{"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int((idx == b).sum()), "mean_score": round(float(p[idx == b].mean()), 3),
             "observed_relevant_rate": round(float(y[idx == b].mean()), 3)} for b in range(bins) if (idx == b).any()]
    return pd.DataFrame(rows)


def ece(y, p, bins=5) -> float:
    c = calibration(y, p, bins)
    return round(float(sum(r["n"] * abs(r["mean_score"] - r["observed_relevant_rate"]) for r in c.to_dict("records")) / len(y)), 4)


def strict_threshold(y, p, w, target: float = PRECISION_TARGET, floor: float = HIGH_FLOOR, min_pred: int = 8) -> tuple:
    """Lowest t >= floor with weighted out-of-fold precision >= target. Returns (t, reached).
    If the target is never reached, returns the t with the best weighted precision (and reached=False)."""
    best_t, best_prec = None, -1.0
    for t in np.round(np.arange(floor, 0.995, 0.005), 3):
        pred = p >= t
        if pred.sum() < min_pred:
            break
        prec = metrics(y, pred, w)["precision"]
        if prec >= target:
            return float(t), True
        if prec > best_prec + 1e-12:
            best_t, best_prec = float(t), prec
    return (best_t if best_t is not None else float(floor)), False


def low_threshold(y, p, max_rate: float = 0.05, min_n: int = 20) -> float:
    """Highest calibrated probability t such that at most `max_rate` of labeled videos at or below t are Relevant (and >= min_n videos)."""
    for t in np.round(np.arange(0.40, 0.0095, -0.01), 2):
        sel = p <= t
        if sel.sum() >= min_n and float(y[sel].mean()) <= max_rate:
            return float(t)
    return 0.05


def threshold_sweep(y, p, w, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        pred = p >= t
        m, mw = metrics(y, pred), metrics(y, pred, w)
        rows.append({"threshold": t, "videos_predicted_relevant": int(pred.sum()), "precision": m["precision"], "recall": m["recall"],
                     "weighted_precision": mw["precision"], "weighted_recall": mw["recall"], "false_positives": m.get("fp", 0)})
    return pd.DataFrame(rows)


def zone_of(p, high: float, low: float = ZONE_LOW) -> np.ndarray:
    return np.where(p >= high, "relevant_confident", np.where(p <= low, "irrelevant_confident", "audit"))


def error_category(r) -> str:
    if r["rule_reason"].startswith("false_match"):
        return r["rule_reason"]
    if r["rule_reason"] == "sleep_context":
        return "sleep context"
    if r["rule_reason"] == "no_topic_term":
        return "no holding/restraint term in text"
    if r["rule_reason"] in ("topic_without_age", "child_and_setting_only"):
        return "age or topic unclear in text"
    if r["word_count"] < 6:
        return "very short / uninformative text"
    return "other"


def coefficient_table(data: Data, lab_idx, y) -> pd.DataFrame:
    """Interpretable TF-IDF companion: largest word coefficients (the selected model may be embedding based)."""
    m = Model("tfidf", False, data, "meta").fit(lab_idx, y)
    words = m.w.get_feature_names_out()
    return pd.DataFrame({"feature": words, "coefficient": m.lr.coef_[0][:len(words)]})


# ---------------------------------------------------------------- main
def run(settings, stage: str = "all", use_embeddings: bool = True, out_dir: Optional[Path] = None,
        encoder: Optional[Callable] = None, repeats: int = 3, folds: int = 5, embedding_model: str = EMBEDDING_MODEL,
        precision_target: float = PRECISION_TARGET, high_floor: float = HIGH_FLOOR, zone_low: Optional[float] = None,
        labeled_path: Optional[Path] = None) -> dict:
    out_dir = Path(out_dir) if out_dir else settings.final_dir / "screening"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = add_features(load_frame(settings, labeled_path))
    emb = None
    if use_embeddings:
        enc = encoder or (lambda t: encode_texts(t, settings.cache_dir, embedding_model))
        e_meta = np.asarray(enc(list(df["text"])))
        snippet = df["transcript"].str.replace(r"\s+", " ", regex=True).str[:TRANSCRIPT_CHARS]
        has_tr = (snippet.str.len() > 0).to_numpy()
        e_tr = np.zeros_like(e_meta)
        if has_tr.any():
            e_tr[has_tr] = np.asarray(enc(list(snippet[has_tr])))
        emb = {"meta": e_meta, "meta+transcript": np.hstack([e_meta, e_tr])}
    data = Data(df, emb)
    lab_idx = np.where(df["y"].notna())[0]
    if len(lab_idx) < 50:
        raise ValueError(f"Only {len(lab_idx)} labeled videos found; run scripts/build_labeled_set.py first.")
    y = df["y"].to_numpy()[lab_idx].astype(int)
    w = df["w"].to_numpy()[lab_idx]
    platform = df["platform"].to_numpy()[lab_idx]
    for pl in np.unique(platform):
        if len(set(y[platform == pl])) < 2:
            raise ValueError(f"Platform {pl} has only one class among the labels; cannot cross-validate.")

    specs = base_specs(use_embeddings)
    log.info("Creator-grouped CV: %d candidates, %d repeats x %d folds, %d labeled videos", len(specs), repeats, folds, len(lab_idx))
    oof = grouped_cv(df, data, lab_idx, specs, repeats, folds)
    table = evaluate_candidates(y, platform, w, oof, precision_target)
    chosen_name = select(table)
    chosen = next(s for s in specs if s.name == chosen_name)
    p_raw = oof[chosen]
    calibrator = fit_calibrator(p_raw, y)
    p_sel = apply_calibrator(calibrator, p_raw)
    high, reached = strict_threshold(y, p_sel, w, precision_target, high_floor)
    low_auto = zone_low is None
    zone_low = low_threshold(y, p_sel) if low_auto else float(zone_low)
    sweep = threshold_sweep(y, p_sel, w)

    pred_strict = (p_sel >= high).astype(int)
    perf = {"overall": metrics(y, pred_strict), "overall_weighted": metrics(y, pred_strict, w),
            "ci95_precision": bootstrap_ci(y, pred_strict, "precision"), "ci95_recall": bootstrap_ci(y, pred_strict, "recall")}
    for pl in np.unique(platform):
        k = platform == pl
        perf[pl] = metrics(y[k], pred_strict[k])
    rules_pred = df["rule_label"].map(RULE_TO_BINARY).to_numpy()[lab_idx]
    baselines = {"all_relevant": np.ones_like(y), "rules only (potentially -> relevant)": rules_pred,
                 "gate_topic only": df["gate_topic"].to_numpy()[lab_idx].astype(int), "gate_strict only": data.gates["strict"][lab_idx].astype(int)}
    baselines = {k: {"unweighted": metrics(y, v), "weighted": metrics(y, v, w)} for k, v in baselines.items()}

    lab = df.iloc[lab_idx].assign(y_true=y, score=p_sel, pred=pred_strict)
    err = lab[lab["y_true"] != lab["pred"]].copy()
    err["error_type"] = np.where(err["pred"] == 1, "false_positive (would enter dataset)", "false_negative (relevant video missed or sent to audit)")
    err["category"] = [error_category(r) for _, r in err.iterrows()]
    zone_oof = zone_of(p_sel, high, zone_low)
    zone_quality = [{"zone": z, "videos": int((zone_oof == z).sum()), "observed_relevant_rate": round(float(y[zone_oof == z].mean()), 3),
                     "weighted_relevant_rate": round(float(np.average(y[zone_oof == z], weights=w[zone_oof == z])), 3)}
                    for z in ("irrelevant_confident", "audit", "relevant_confident") if (zone_oof == z).any()]
    cal_raw, cal = calibration(y, p_raw), calibration(y, p_sel)
    report = {
        "seed": SEED, "repeats": repeats, "folds": folds, "n_labeled": int(len(lab_idx)), "prevalence_relevant": round(float(y.mean()), 3),
        "prevalence_relevant_weighted": round(float(np.average(y, weights=w)), 3),
        "per_platform_labeled": {p: {"relevant": int(y[platform == p].sum()), "irrelevant": int((1 - y[platform == p]).sum())} for p in np.unique(platform)},
        "selected": chosen.name, "embedding_model": embedding_model if use_embeddings else None,
        "selection_rule": "highest macro (per-platform) AUC over pooled out-of-fold scores; within 0.01 -> least complex",
        "strict": {"precision_target": precision_target, "high_floor": high_floor, "high_threshold": round(high, 3), "target_reached": bool(reached),
                   "zone_low": round(zone_low, 3), "zone_low_rule": "data-driven: at most 5% of labeled videos at or below it are Relevant" if low_auto else "set by user"},
        "threshold_sweep": sweep.to_dict("records"), "performance_at_strict_threshold": perf, "baselines": baselines,
        "selected_metrics": table[table.candidate == chosen.name].iloc[0].to_dict(), "ablation": ablation_summary(table),
        "calibration_ece_raw": ece(y, p_raw), "calibration_ece_calibrated": ece(y, p_sel), "zone_quality_oof": zone_quality,
        "error_counts": err.groupby(["error_type", "category"]).size().rename("n").reset_index().to_dict("records"),
    }
    sweep.to_csv(out_dir / "threshold_sweep.csv", index=False, encoding="utf-8-sig")
    table.to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.concat([cal_raw.assign(scale="raw score"), cal.assign(scale="calibrated")]).to_csv(out_dir / "calibration.csv", index=False, encoding="utf-8-sig")
    coefs = coefficient_table(data, lab_idx, y)
    coefs.sort_values("coefficient").to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    err[["video_uid", "platform", "title", "score", "y_true", "pred", "error_type", "category"]].to_csv(out_dir / "cv_errors.csv", index=False, encoding="utf-8-sig")

    if stage == "all":
        _final(df, data, chosen, lab_idx, y, high, zone_low, calibrator, report, out_dir)
    _write_report(report, table, cal_raw, cal, coefs, out_dir)
    (out_dir / "screening_report.json").write_text(json.dumps(_clean(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _final(df, data, chosen: Spec, lab_idx, y, high, low, calibrator, report, out_dir: Path) -> None:
    p = fit_predict(chosen, data, lab_idx, y, np.arange(len(df))) * data.gates[chosen.gate]
    p = apply_calibrator(calibrator, p)
    human = df["y"].notna().to_numpy()
    zone = zone_of(p, high, low)
    out = df.copy()
    out["score"] = np.round(p, 4)
    out["label_source"] = np.where(human, "human", "model")
    out["zone"] = np.where(human, "human_label", zone)
    model_label = np.where(zone == "relevant_confident", "relevant", np.where(zone == "irrelevant_confident", "irrelevant", "audit"))
    out["screening_label"] = np.where(human, df["label_final"].str.lower(), model_label)
    keep = ["video_uid", "platform", "video_id", "url", "title", "screening_label", "label_source", "zone", "score", "rule_label", "rule_reason",
            "query_ids", "query_subtopics"]
    out[keep].to_csv(out_dir / "screening_results.csv", index=False, encoding="utf-8-sig")
    drop = {"text", "text_tr", "rule_label", "rule_reason", "gate_topic", "gate_strict", "gate_veto", "y", "w", "label_final", "sample_weight", "stratum", "word_count",
            "screening_label", "label_source", "zone", "score", "unsafe_demo"} | set(NUMERIC)
    base = [c for c in out.columns if c not in drop]
    extra = ["screening_label", "label_source", "zone", "score"]
    strict = out[out["screening_label"] == "relevant"]
    strict[base + extra].to_csv(out_dir / "relevant_strict_videos.csv", index=False, encoding="utf-8-sig")
    out[out["screening_label"] == "irrelevant"][base + extra].to_csv(out_dir / "irrelevant_videos.csv", index=False, encoding="utf-8-sig")
    audit = out[out["zone"] == "audit"].assign(dist=lambda d: (d["score"] - high).abs()).sort_values("dist")
    audit[base + extra].to_csv(out_dir / "audit_videos.csv", index=False, encoding="utf-8-sig")
    unl = out[~human]
    report["funnel"] = {
        "cleaned_videos": int(len(df)), "human_labeled": int(human.sum()), "model_scored": int((~human).sum()),
        "model_zones": unl["zone"].value_counts().to_dict(),
        "strict_relevant_total": int(len(strict)), "strict_relevant_human": int((strict["label_source"] == "human").sum()),
        "strict_relevant_model": int((strict["label_source"] == "model").sum()),
        "strict_relevant_by_platform": strict["platform"].value_counts().to_dict(), "audit_videos": int(len(audit)),
        "audit_by_platform": audit["platform"].value_counts().to_dict(),
    }


def _write_report(report, table, cal_raw, cal, coefs, out_dir: Path) -> None:
    s, st = report["selected_metrics"], report["strict"]
    pk = int(st["precision_target"] * 100)
    perf = report["performance_at_strict_threshold"]
    L = ["# Screening report: restraint and immobilization videos (strict)", "",
         f"Seed {report['seed']}. {report['n_labeled']} human-labeled videos ({report['prevalence_relevant'] * 100:.0f}% Relevant; "
         f"{report['prevalence_relevant_weighted'] * 100:.0f}% after sample weights). {report['repeats']} x {report['folds']}-fold CV, stratified by label and "
         "platform, grouped by creator (a creator never appears in both training and validation).", "",
         "Inputs: title, description, hashtags, tags, duration (z-scored within platform), length and signal features; the ablation also tests adding the "
         f"transcript (first {TRANSCRIPT_CHARS} characters). **Not used:** views, likes, query information.", "",
         "## Why this screening is strict", "",
         (f"- The model labels a video Relevant only at calibrated probability >= **{st['high_threshold']}**, the lowest value (not below {st['high_floor']}) at which "
          f"the sample-weighted out-of-fold precision reaches {pk}%." if st["target_reached"] else
          f"- **WARNING: the {pk}% precision target was NOT reached.** The threshold **{st['high_threshold']}** is the value with the best weighted out-of-fold "
          "precision available. Treat model-relevant videos with caution and send more of them to the audit."),
         f"- Videos between {st['zone_low']} ({st['zone_low_rule']}) and {st['high_threshold']} go to an **audit** zone for a human. They are never added to the strict dataset automatically.",
         "- Gate options include a strict gate that needs both a holding/restraint term and a child term, and no false-match signal (restraint collapse, music, "
         "car seats, sleep swaddling, adult or non-medical restraint, animals).", "",
         "## Baselines (all labeled videos)", "", "| baseline | precision | recall | F1 | weighted precision | weighted recall |", "|---|---|---|---|---|---|"]
    for k, m in report["baselines"].items():
        u, w_ = m["unweighted"], m["weighted"]
        L.append(f"| {k} | {u['precision']} | {u['recall']} | {u['f1']} | {w_['precision']} | {w_['recall']} |")
    cols = ["candidate", "macro_auc", "auc", "pr_auc", "brier", f"recall_at_precision_{pk}", "youtube_auc", "tiktok_auc"]
    L += ["", f"## Candidate comparison ({len(table)} candidates, pooled out-of-fold scores)", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in table.to_dict("records"):
        L.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    L += ["", f"**Selected:** `{report['selected']}` (rule: {report['selection_rule']}). Macro AUC {s['macro_auc']}, PR-AUC {s['pr_auc']}.", "",
          "## What the design choices did (mean change in macro AUC over otherwise identical candidate pairs)", "",
          "| change | mean macro-AUC change | pairs | pairs that improved |", "|---|---|---|---|"]
    for r in report["ablation"]:
        L.append(f"| {r['change']} | {r['mean_macro_auc_change']:+.3f} | {r['pairs']} | {r['pairs_improved']} |")
    L += ["", "## Threshold sweep (calibrated probability, out-of-fold, labeled videos)", "",
          "| threshold | videos called relevant | precision | recall | weighted precision | weighted recall | false positives |", "|---|---|---|---|---|---|---|"]
    for r in report["threshold_sweep"]:
        L.append(f"| {r['threshold']} | {r['videos_predicted_relevant']} | {r['precision']} | {r['recall']} | {r['weighted_precision']} | {r['weighted_recall']} | {r['false_positives']} |")
    o, ow = perf["overall"], perf["overall_weighted"]
    L += ["", "## Performance at the strict threshold (out-of-fold)", "",
          "| group | precision | recall | F1 | videos | tp | fp | fn |", "|---|---|---|---|---|---|---|---|",
          f"| all (unweighted) | {o['precision']} | {o['recall']} | {o['f1']} | {o['n']} | {o['tp']} | {o['fp']} | {o['fn']} |",
          f"| all (sample-weighted) | {ow['precision']} | {ow['recall']} | {ow['f1']} | {ow['n']} | | | |"]
    for pl in report["per_platform_labeled"]:
        m = perf[pl]
        L.append(f"| {pl} | {m['precision']} | {m['recall']} | {m['f1']} | {m['n']} | {m['tp']} | {m['fp']} | {m['fn']} |")
    L += ["", f"95% bootstrap interval: precision {perf['ci95_precision']}, recall {perf['ci95_recall']}. With about 100 Relevant labels these intervals are wide.", "",
          "## Calibration (out-of-fold)", "", f"Raw: ECE {report['calibration_ece_raw']}. After Platt scaling: ECE {report['calibration_ece_calibrated']} "
          "(fitted on the same out-of-fold scores, so slightly optimistic).", "", "| scale | bin | videos | mean score | observed relevant rate |", "|---|---|---|---|---|"]
    for name, tab in (("raw", cal_raw), ("calibrated", cal)):
        for r in tab.to_dict("records"):
            L.append(f"| {name} | {r['bin']} | {r['n']} | {r['mean_score']} | {r['observed_relevant_rate']} |")
    L += ["", "## Zones on the labeled videos (out-of-fold)", "", "| zone | videos | observed relevant rate | weighted rate |", "|---|---|---|---|"]
    for r in report["zone_quality_oof"]:
        L.append(f"| {r['zone']} | {r['videos']} | {r['observed_relevant_rate']} | {r['weighted_relevant_rate']} |")
    w = coefs.sort_values("coefficient")
    L += ["", "## Patterns (interpretable TF-IDF companion; largest word coefficients)", "",
          "**Towards relevant:** " + ", ".join(w.tail(15).iloc[::-1].feature), "", "**Towards irrelevant:** " + ", ".join(w.head(15).feature)]
    if report["error_counts"]:
        L += ["", "## Error categories (out-of-fold, strict threshold)", "", "| error | category | n |", "|---|---|---|"]
        for r in report["error_counts"]:
            L.append(f"| {r['error_type']} | {r['category']} | {r['n']} |")
    if "funnel" in report:
        f = report["funnel"]
        L += ["", "## Screening flow", "",
              f"{f['cleaned_videos']} cleaned videos: {f['human_labeled']} human-labeled, {f['model_scored']} scored by the model "
              f"(zones {f['model_zones']}).", f"**Strict relevant dataset: {f['strict_relevant_total']} videos** ({f['strict_relevant_human']} human Relevant + "
              f"{f['strict_relevant_model']} model relevant_confident). By platform: {f['strict_relevant_by_platform']}.",
              f"**Audit queue: {f['audit_videos']} videos** {f['audit_by_platform']}, to be labeled by a human; sorted by closeness to the threshold."]
    L += ["", "## Limitations", "",
          "- About 100 Relevant labels: all metrics and the threshold have wide uncertainty; 314 of 400 labels come from a single annotator.",
          "- The threshold is tuned on the same out-of-fold scores, so reported precision at the threshold is slightly optimistic.",
          "- Models see title, description, hashtags and tags only; annotators also saw the video and, where available, the transcript.",
          "- Sample weights correct for stratified sampling but add variance; weighted and unweighted numbers are both shown.",
          "- The rule lexicon is written from the criteria (not tuned on labels); it is mostly English, so non-English videos rely more on embeddings."]
    (out_dir / "screening_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")