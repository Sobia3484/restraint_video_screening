"""EDA sections. Each returns a Section with auto-generated findings (numbers come from the data, never typed by hand)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import plots
from .features import DURATION_LABELS
from .stats import compare_two, describe, fmt_p, gini

P = ("youtube", "tiktok")
NAME = {"youtube": "YouTube", "tiktok": "TikTok"}


@dataclass
class Section:
    title: str
    intro: str = ""
    findings: list = field(default_factory=list)
    figures: list = field(default_factory=list)
    tables: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def _n(x) -> str:
    return f"{x:,.0f}"


def _pct(x: float) -> str:
    return f"{x:.1f}%" if 0 < x < 1 else f"{x:.0f}%"


def _csv(df: pd.DataFrame, out: Path, name: str, index=False) -> str:
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables" / name, index=index, encoding="utf-8-sig")
    return name


def _fig(out: Path, name: str) -> Path:
    return out / "figures" / name


def _by(d, p):
    return d[d["platform"] == p]


def _cmp_line(label, c, unit="", dec=1) -> str:
    if "p_value" not in c:
        return f"{label}: not enough data to compare."
    return (f"{label}: YouTube median {c['median_x']:,.{dec}f}{unit} vs TikTok median {c['median_y']:,.{dec}f}{unit} "
            f"(difference {c['median_diff']:+,.{dec}f}{unit}, 95% CI {c['median_diff_ci_low']:+,.{dec}f} to {c['median_diff_ci_high']:+,.{dec}f}); "
            f"Mann-Whitney {fmt_p(c['p_value'])}; Cliff's delta {c['cliffs_delta']:+.2f} (95% CI {c['delta_ci_low']:+.2f} to {c['delta_ci_high']:+.2f}), "
            f"{c['magnitude']} effect (positive = YouTube tends to be larger).")


# ------------------------------------------------------------------------------------------------ 0 data quality
def quality(d: pd.DataFrame, out: Path) -> Section:
    s = Section("0. Data quality checks", "Checks run before any analysis, so that charts and tests are not built on broken data.")
    rows = []

    def add(check, platform, value):
        rows.append({"check": check, "platform": platform, "value": value})

    add("rows", "all", len(d)); add("unique video_uid", "all", int(d["video_uid"].nunique())); add("duplicate video_uid", "all", int(d["video_uid"].duplicated().sum()))
    ref = pd.Timestamp(d["reference_date"].iloc[0])
    for p in P:
        x = _by(d, p)
        add("videos", p, len(x))
        add("negative numeric values (views, likes, comments, shares, saves, duration)", p, int((x[["views", "likes", "comments", "shares", "saves", "duration_seconds"]] < 0).sum().sum()))
        add("zero views", p, int((x["views"] == 0).sum()))
        add("likes greater than views", p, int((x["likes"] > x["views"]).sum()))
        add("comments greater than views", p, int((x["comments"] > x["views"]).sum()))
        add("upload date after the collection date", p, int((pd.to_datetime(x["upload_date"], errors="coerce") > ref).sum()))
        add("upload date missing", p, int(x["upload_date"].isna().sum() + (x["upload_date"].astype(str) == "").sum()))
        add("duration over 3 hours", p, int((x["duration_seconds"] > 10_800).sum()))
        add("duplicate title and creator", p, int(x.duplicated(["title", "creator_id"]).sum()))
    q = pd.DataFrame(rows)
    s.tables.append(_csv(q, out, "00_data_quality.csv"))

    cols = ["title", "description", "hashtags", "tags", "creator_followers", "duration_seconds", "upload_date", "views", "likes", "comments", "shares", "saves", "language_clean", "country_clean"]
    miss = []
    for c in cols:
        for p in P:
            x = _by(d, p)[c]
            m = int((x.isna() | (x.astype(str) == "") | (x.astype(str).str.lower().isin(["unknown", "nan"]))).sum())
            miss.append({"column": c, "platform": p, "missing": m, "videos": len(x), "missing_pct": round(m / max(len(x), 1) * 100, 1)})
    miss = pd.DataFrame(miss)
    s.tables.append(_csv(miss.pivot(index="column", columns="platform", values="missing_pct").reset_index(), out, "00_missingness_pct.csv"))

    ext = []
    for p in P:
        x = _by(d, p)
        for col, k in (("views", "most viewed"), ("duration_seconds", "longest")):
            for r in x.nlargest(5, col).itertuples():
                ext.append({"platform": p, "kind": k, "value": getattr(r, col), "title": r.title[:90], "creator": r.creator_name, "url": r.url})
    s.tables.append(_csv(pd.DataFrame(ext), out, "00_extremes.csv"))

    dup = int(d["video_uid"].duplicated().sum())
    s.findings.append(f"{_n(len(d))} videos ({', '.join(f'{NAME[p]} {_n((d.platform == p).sum())}' for p in P)}); duplicate video ids: {dup}.")
    for p in P:
        m = miss[miss.platform == p].set_index("column")["missing_pct"]
        s.findings.append(f"{NAME[p]}: missing likes {m['likes']}%, comments {m['comments']}%, followers {m['creator_followers']}%, description {m['description']}%, tags {m['tags']}%, hashtags {m['hashtags']}%.")
    s.findings.append("TikTok has no separate description or tags (the caption is stored as the title), so 100% missing there is by design, not a data problem.")
    bad = q[(q.check.str.contains("negative|greater|after the collection")) & (q.value > 0)]
    s.findings.append("No impossible values found (negative counts, likes or comments above views, future upload dates)." if bad.empty else
                      "Impossible values found: " + "; ".join(f"{r.platform}: {r.check} = {r.value}" for r in bad.itertuples()))
    long_ = q[(q.check == "duration over 3 hours") & (q.value > 0)]
    if len(long_):
        s.findings.append("Very long videos (over 3 hours): " + ", ".join(f"{NAME[r.platform]} {r.value}" for r in long_.itertuples()) + " (kept; analyses use medians and log scales).")
    s.notes.append("Likes and comments can be hidden or disabled by the creator, so 'missing' there is a platform setting, not a data error. Shares and saves exist only on TikTok.")
    return s


# ------------------------------------------------------------------------------------------------ features summary
def features_summary(d: pd.DataFrame, out: Path) -> Section:
    s = Section("1. Derived features", "Columns created for the analysis (duration groups, rates, dates, age group, subtopic, detected language). They come from keyword rules and text, are not human-validated, and are saved in the features file.")
    for col, name in (("age_group", "01_age_group_by_platform.csv"), ("subtopic", "01_subtopic_by_platform.csv"), ("language_detected", "01_language_by_platform.csv"), ("follower_group", "01_followers_by_platform.csv")):
        t = pd.crosstab(d[col].astype(str), d["platform"]).reindex(columns=list(P), fill_value=0)
        t["total"] = t.sum(axis=1)
        s.tables.append(_csv(t.sort_values("total", ascending=False).reset_index().rename(columns={col: col}), out, name))
    ag = pd.crosstab(d["platform"], d["age_group"])
    s.figures.append(plots.stacked_share(ag, _fig(out, "01_age_group_share.png"), "Age group mentioned in the text, by platform"))
    s.findings.append("Age group in the text: " + "; ".join(f"{NAME[p]}: " + ", ".join(f"{k} {v / ag.loc[p].sum() * 100:.0f}%" for k, v in ag.loc[p].sort_values(ascending=False).head(3).items()) for p in P) + ".")
    m = d["language_method"].value_counts()
    s.findings.append("Language detection method: " + ", ".join(f"{k} {v}" for k, v in m.items()) + ". Texts with fewer than 25 letters are labeled 'und' (undetermined).")
    s.notes.append("Subtopic and age group come from keyword matching on title, hashtags, tags and description. They are descriptive labels, not validated against human labels. The setting (subtopic) is the first one found in the order trauma, dental, imaging, needle, wound care, examination, neonatal, safety, alternatives, so a video that names two settings is counted once. Language detection on short social-media captions is error-prone, and Roman Urdu or Hindi written in Latin letters cannot be told apart from other Latin-script languages.")
    return s


# ------------------------------------------------------------------------------------------------ 2 overview
def overview(d: pd.DataFrame, out: Path) -> Section:
    s = Section("2. Dataset overview", "Who and what is in the final dataset.")
    rows = []
    for p in P:
        x = _by(d, p)
        per = x.groupby("creator_id").size() if len(x) else pd.Series(dtype=int)
        top10 = per.sort_values(ascending=False).head(10).sum() / max(len(x), 1) * 100
        rows.append({"platform": p, "videos": len(x), "share_of_dataset_pct": round(len(x) / len(d) * 100, 1), "distinct_creators": int(x["creator_id"].nunique()),
                     "videos_per_creator_mean": round(per.mean(), 2), "videos_per_creator_median": float(per.median()), "videos_per_creator_max": int(per.max()),
                     "top10_creators_share_of_videos_pct": round(top10, 1), "gini_videos_per_creator": round(gini(per), 3),
                     "total_duration_hours": round(x["duration_seconds"].sum() / 3600, 1), "median_duration_min": round(x["duration_min"].median(), 2),
                     "videos_with_transcript": int((x["has_transcript"].astype(str) == "True").sum()),
                     "found_by_2plus_queries": int((x["query_count"] >= 2).sum()), "human_labeled": int((x["label_source"] == "human").sum()),
                     "model_scored": int((x["label_source"] == "model").sum())})
    t = pd.DataFrame(rows)
    s.tables.append(_csv(t, out, "02_overview_by_platform.csv"))
    top = (d.groupby(["platform", "creator_id", "creator_name"]).size().rename("videos").reset_index().sort_values(["platform", "videos"], ascending=[True, False]).groupby("platform").head(10))
    s.tables.append(_csv(top, out, "02_top_creators.csv"))
    zone = pd.crosstab(d["platform"], d["zone"])
    s.tables.append(_csv(zone.reset_index(), out, "02_zone_by_platform.csv"))
    s.figures.append(plots.stacked_share(zone, _fig(out, "02_zone_share.png"), "How confident the screening was (zone), by platform"))
    for r in rows:
        p = NAME[r["platform"]]
        s.findings.append(f"{p}: {_n(r['videos'])} videos ({r['share_of_dataset_pct']}% of the dataset) from {_n(r['distinct_creators'])} creators; "
                          f"median {r['videos_per_creator_median']:.0f} and maximum {r['videos_per_creator_max']} videos per creator; the top 10 creators contribute "
                          f"{r['top10_creators_share_of_videos_pct']}% of the videos (Gini {r['gini_videos_per_creator']}); total length {r['total_duration_hours']:,.1f} hours.")
    qc = d.groupby("platform")["query_count"].apply(lambda x: (x >= 2).mean() * 100)
    s.findings.append("Videos found by two or more of the 25 queries: " + ", ".join(f"{NAME[p]} {qc[p]:.0f}%" for p in P) + ".")
    hs = d["label_source"].value_counts()
    s.findings.append(f"Decision source: {_n(hs.get('human', 0))} videos have a human label and {_n(hs.get('model', 0))} were scored by the model.")
    s.notes.append("These are the videos the platforms returned for the 25 queries on one day, not a random sample of all videos on the topic.")
    return s


# ------------------------------------------------------------------------------------------------ 3 duration
def duration(d: pd.DataFrame, out: Path) -> Section:
    s = Section("3. Video duration", "How long are the videos, and do the platforms differ?")
    summ = pd.DataFrame([{"platform": p, **describe(_by(d, p)["duration_seconds"])} for p in P])
    s.tables.append(_csv(summ.round(2), out, "03_duration_summary_seconds.csv"))
    g = pd.crosstab(d["platform"], d["duration_group"]).reindex(columns=DURATION_LABELS, fill_value=0)
    s.tables.append(_csv(g.reset_index(), out, "03_duration_groups.csv"))
    c = compare_two(_by(d, "youtube")["duration_seconds"], _by(d, "tiktok")["duration_seconds"])
    s.tables.append(_csv(pd.DataFrame([c]), out, "03_duration_test.csv"))
    s.figures += [
        plots.hist_by_platform(d, "duration_seconds", _fig(out, "03_duration_hist.png"), "Video duration (log scale, dashed line = median)", "Duration in seconds (log scale)", log_x=True),
        plots.box_by_platform(d, {"Duration (minutes, log scale)": "duration_min"}, _fig(out, "03_duration_box.png"), "Video duration by platform", log_y=True),
        plots.stacked_share(g, _fig(out, "03_duration_groups.png"), "Duration groups, by platform"),
    ]
    for p in P:
        r = summ[summ.platform == p].iloc[0]
        x = _by(d, p)
        s.findings.append(f"{NAME[p]}: median {r['median']:.0f} s ({r['median'] / 60:.1f} min); middle half of videos between {r['q1']:.0f} and {r['q3']:.0f} s; shortest {r['min']:.0f} s, longest {r['max'] / 60:.0f} min; "
                          f"{_pct((x['duration_seconds'] <= 60).mean() * 100)} are 60 seconds or shorter and {_pct((x['duration_seconds'] > 600).mean() * 100)} are longer than 10 minutes.")
    s.findings.append(_cmp_line("Duration", c, " s", 0))
    s.notes.append("Duration is strongly right-skewed, so medians, quartiles and a log scale are used instead of means. A YouTube video of 60 seconds or less may be a Short; the data do not say.")
    return s


# ------------------------------------------------------------------------------------------------ 4 views
def views(d: pd.DataFrame, out: Path) -> Section:
    s = Section("4. Views", "How often are the videos watched, and how concentrated are the views?")
    summ = pd.DataFrame([{"platform": p, **describe(_by(d, p)["views"])} for p in P])
    s.tables.append(_csv(summ.round(1), out, "04_views_summary.csv"))
    conc = []
    for p in P:
        v = _by(d, p)["views"].dropna().sort_values(ascending=False)
        tot = v.sum()
        conc.append({"platform": p, "videos": len(v), "total_views": tot, "top10_videos_share_pct": round(v.head(10).sum() / tot * 100, 1) if tot else np.nan,
                     "top10pct_videos_share_pct": round(v.head(max(1, len(v) // 10)).sum() / tot * 100, 1) if tot else np.nan, "gini_views": round(gini(v), 3),
                     "videos_under_1000_views_pct": round((v < 1000).mean() * 100, 1), "videos_over_1m_views": int((v >= 1_000_000).sum())})
    conc = pd.DataFrame(conc)
    s.tables.append(_csv(conc, out, "04_views_concentration.csv"))
    c = compare_two(_by(d, "youtube")["views"], _by(d, "tiktok")["views"])
    s.tables.append(_csv(pd.DataFrame([c]), out, "04_views_test.csv"))
    s.figures += [
        plots.hist_by_platform(d, "views", _fig(out, "04_views_hist.png"), "Views per video (log scale, dashed line = median)", "Views (log scale)", log_x=True),
        plots.ecdf_by_platform(d, "views", _fig(out, "04_views_ecdf.png"), "Cumulative share of videos by number of views", "Views (log scale)"),
        plots.box_by_platform(d, {"Views (log scale)": "views"}, _fig(out, "04_views_box.png"), "Views by platform", log_y=True),
    ]
    for p in P:
        r, k = summ[summ.platform == p].iloc[0], conc[conc.platform == p].iloc[0]
        s.findings.append(f"{NAME[p]}: median {r['median']:,.0f} views (middle half {r['q1']:,.0f} to {r['q3']:,.0f}); maximum {r['max']:,.0f}. {k['videos_under_1000_views_pct']}% of videos have fewer than 1,000 views and {k['videos_over_1m_views']} have a million or more. "
                          f"The 10 most viewed videos account for {k['top10_videos_share_pct']}% of all views and the top 10% of videos for {k['top10pct_videos_share_pct']}% (Gini {k['gini_views']}).")
    s.findings.append(_cmp_line("Views", c, "", 0))
    s.notes.append("Views are highly skewed; a few videos dominate the totals. The platforms count a 'view' differently and the data were collected on one day, so older videos have had longer to gather views. The videos are what search returned and platforms favour popular videos, so these figures describe search results, not all videos.")
    return s


# ------------------------------------------------------------------------------------------------ 5 engagement
def engagement(d: pd.DataFrame, out: Path) -> Section:
    s = Section("5. Engagement", "How much do viewers react? Rates are in percent of views and are computed only for videos with at least 100 views (rates of videos with very few views are unstable).")
    rows, tests = [], []
    for name, col in (("like rate", "like_rate"), ("comment rate", "comment_rate")):
        for p in P:
            x = _by(d, p)
            rows.append({"measure": name, "platform": p, "videos_with_rate": int(x[col].notna().sum()), "videos_with_100plus_views": int((x["views"] >= 100).sum()),
                         "videos_with_zero_rate": int((x[col] == 0).sum()), **{k: v for k, v in describe(x[col]).items() if k != "n"}})
        c = compare_two(_by(d, "youtube")[col], _by(d, "tiktok")[col])
        tests.append({"measure": name, **c})
    summ = pd.DataFrame(rows)
    s.tables.append(_csv(summ.round(3), out, "05_engagement_summary.csv"))
    s.tables.append(_csv(pd.DataFrame(tests), out, "05_engagement_tests.csv"))
    tt = _by(d, "tiktok")
    extra = pd.DataFrame([{"measure": m, **{k: v for k, v in describe(tt[c]).items()}} for m, c in (("share rate", "share_rate"), ("save rate", "save_rate"))]).round(3)
    s.tables.append(_csv(extra, out, "05_tiktok_share_save_rates.csv"))
    corr = []
    for p in P:
        x = _by(d, p)
        for col in ("like_rate", "comment_rate"):
            z = x[["views", col]].dropna()
            z = z[z["views"] > 0]
            if len(z) > 5:
                rho, pv = sps.spearmanr(np.log10(z["views"]), z[col])
                corr.append({"platform": p, "rate": col, "n": len(z), "spearman_rho_vs_log_views": round(float(rho), 3), "p_value": float(pv)})
    s.tables.append(_csv(pd.DataFrame(corr), out, "05_rate_vs_views_correlation.csv"))
    s.figures += [
        plots.box_by_platform(d, {"Like rate (% of views, log scale)": "like_rate", "Comment rate (% of views, log scale)": "comment_rate"}, _fig(out, "05_rates_box.png"), "Engagement rates by platform", log_y=True),
        plots.hist_by_platform(d, "like_rate", _fig(out, "05_like_rate_hist.png"), "Like rate (log scale, dashed line = median)", "Likes as % of views (log scale)", log_x=True),
    ]
    for name in ("like rate", "comment rate"):
        for p in P:
            r = summ[(summ.measure == name) & (summ.platform == p)].iloc[0]
            s.findings.append(f"{NAME[p]} {name}: median {r['median']:.2f}% (middle half {r['q1']:.2f}% to {r['q3']:.2f}%), available for {_n(r['videos_with_rate'])} of {_n(r['videos_with_100plus_views'])} videos with at least 100 views.")
    for name, col in (("Like rate", "like_rate"), ("Comment rate", "comment_rate")):
        s.findings.append(_cmp_line(name, compare_two(_by(d, "youtube")[col], _by(d, "tiktok")[col]), "%", 2))
    for r in corr:
        if r["rate"] == "like_rate":
            s.findings.append(f"{NAME[r['platform']]}: Spearman correlation between log views and like rate rho = {r['spearman_rho_vs_log_views']:+.2f} (n = {r['n']}).")
    if len(extra):
        s.findings.append("TikTok only: median share rate " + f"{extra.loc[extra.measure == 'share rate', 'median'].iloc[0]:.2f}% and median save rate {extra.loc[extra.measure == 'save rate', 'median'].iloc[0]:.2f}% of views.")
    zero = {p: int((_by(d, p)["comment_rate"] == 0).sum()) for p in P}
    s.notes.append(f"Videos with a rate of exactly zero (for example no comments: YouTube {zero['youtube']}, TikTok {zero['tiktok']}) cannot be drawn on a log scale, so they are left out of the box plots only; the statistics include them.")
    s.notes.append("YouTube creators can hide likes or disable comments, and those videos are excluded from the rates, so the YouTube rates describe the videos where the counts are visible. The platforms differ in how people react (TikTok has shares and saves, YouTube has long watch sessions), so rates are compared with care and only descriptively.")
    return s


# ------------------------------------------------------------------------------------------------ 6 upload trends
def upload_trends(d: pd.DataFrame, out: Path) -> Section:
    s = Section("6. Upload trends", "When were the videos uploaded, and how does the observed dataset vary over time?")
    x = d.dropna(subset=["upload_year_month"]).copy()
    monthly = pd.crosstab(x["upload_year_month"], x["platform"]).reindex(columns=list(P), fill_value=0).sort_index()
    monthly["total"] = monthly.sum(axis=1)
    s.tables.append(_csv(monthly.reset_index(), out, "06_uploads_by_month.csv"))
    yearly = pd.crosstab(x["upload_year"], x["platform"]).reindex(columns=list(P), fill_value=0).sort_index()
    yearly["total"] = yearly.sum(axis=1)
    s.tables.append(_csv(yearly.reset_index(), out, "06_uploads_by_year.csv"))
    fig = plots.bar_counts(monthly["total"], _fig(out, "06_uploads_by_month.png"), "Videos by upload month", "Videos", top=24)
    s.figures.append(fig)
    if len(yearly):
        latest = yearly.index.max()
        first = yearly.index.min()
        s.findings.append(f"Upload years in the dataset range from {int(first)} to {int(latest)}; {int(yearly.loc[latest, 'total']) if latest in yearly.index else 0:,} videos were uploaded in {int(latest)}.")
    if len(monthly):
        top = monthly["total"].nlargest(3)
        s.findings.append("Highest observed upload months: " + ", ".join(f"{idx} ({int(v):,})" for idx, v in top.items()) + ".")
    s.notes.append("These are counts among videos returned by the 25 queries on the collection date; they are not a complete measure of how much content was published on each date.")
    return s


# ------------------------------------------------------------------------------------------------ 7 language and country
def language_country(d: pd.DataFrame, out: Path) -> Section:
    s = Section("7. Language and country", "What languages and countries are represented in the final relevant dataset?")
    lang = pd.crosstab(d["language_detected"].astype(str), d["platform"]).reindex(columns=list(P), fill_value=0)
    lang["total"] = lang.sum(axis=1)
    lang = lang.sort_values("total", ascending=False)
    s.tables.append(_csv(lang.reset_index().rename(columns={"language_detected": "language"}), out, "07_language_detected.csv"))
    top_lang = lang.head(10).drop(columns=["total"])
    s.figures.append(plots.bar_counts(top_lang.sum(axis=1), _fig(out, "07_language_top10.png"), "Top detected languages", "Videos"))
    for p in P:
        vals = lang[p].sort_values(ascending=False)
        if len(vals):
            top = vals.index[0]
            s.findings.append(f"{NAME[p]}: most common detected language is {top} ({int(vals.iloc[0]):,} videos).")
    if "country_clean" in d:
        c = d.loc[d["country_clean"].fillna("").astype(str).str.lower().isin(["", "unknown", "nan"]) == False].copy()
        if len(c):
            country = pd.crosstab(c["country_clean"].astype(str), c["platform"]).reindex(columns=list(P), fill_value=0)
            country["total"] = country.sum(axis=1)
            country = country.sort_values("total", ascending=False)
            s.tables.append(_csv(country.reset_index().rename(columns={"country_clean": "country"}), out, "07_country.csv"))
            topc = country["total"].head(15)
            s.figures.append(plots.bar_counts(topc, _fig(out, "07_country_top15.png"), "Top reported countries", "Videos"))
            s.findings.append("Country metadata are available mainly where the platform reports a country; the top reported country is " + f"{country.index[0]} ({int(country.iloc[0]['total']):,} videos).")
        else:
            s.notes.append("No non-missing country metadata were available for this dataset.")
    s.notes.append("Detected language is text-based and can be uncertain for short captions. Roman Urdu/Hindi written in Latin script cannot be reliably separated by this method. Country is platform-reported metadata, not a verified creator residence.")
    return s


# ------------------------------------------------------------------------------------------------ 8 creator analysis
def creators(d: pd.DataFrame, out: Path) -> Section:
    s = Section("8. Creator analysis", "How concentrated is the relevant content among creators?")
    rows = []
    top_rows = []
    for p in P:
        x = _by(d, p)
        counts = x.groupby(["creator_id", "creator_name"], dropna=False).size().sort_values(ascending=False)
        for (cid, name), n in counts.head(20).items():
            top_rows.append({"platform": p, "creator_id": cid, "creator_name": name, "videos": int(n)})
        if len(counts):
            rows.append({"platform": p, "creators": int(len(counts)), "videos": int(len(x)), "median_videos_per_creator": float(counts.median()), "max_videos_per_creator": int(counts.max()), "top10_share_pct": round(counts.head(10).sum() / len(x) * 100, 1)})
    s.tables.append(_csv(pd.DataFrame(rows), out, "08_creator_summary.csv"))
    s.tables.append(_csv(pd.DataFrame(top_rows), out, "08_top_creators.csv"))
    if top_rows:
        tc = pd.DataFrame(top_rows).assign(label=lambda z: z["platform"].str.title() + ": " + z["creator_name"].fillna("unknown").astype(str).str[:45])
        tc = tc.sort_values("videos").tail(15).set_index("label")["videos"]
        s.figures.append(plots.bar_counts(tc, _fig(out, "08_top_creators.png"), "Top creator video counts", "Videos"))
    for r in rows:
        s.findings.append(f"{NAME[r['platform']]}: {r['creators']:,} creators for {r['videos']:,} videos; median {r['median_videos_per_creator']:.0f} video per creator and maximum {r['max_videos_per_creator']:,}; top 10 creators contribute {r['top10_share_pct']:.1f}% of videos.")
    s.notes.append("Creator concentration is descriptive. The dataset is search-result based, so it does not represent the complete creator population.")
    return s


# ------------------------------------------------------------------------------------------------ 9 subtopics
def subtopics(d: pd.DataFrame, out: Path) -> Section:
    s = Section("9. Subtopics", "Which clinical settings appear most often in the videos?")
    tab = pd.crosstab(d["subtopic"].astype(str), d["platform"]).reindex(columns=list(P), fill_value=0)
    tab["total"] = tab.sum(axis=1)
    tab = tab.sort_values("total", ascending=False)
    s.tables.append(_csv(tab.reset_index().rename(columns={"subtopic": "subtopic"}), out, "09_subtopics.csv"))
    s.figures.append(plots.bar_counts(tab["total"], _fig(out, "09_subtopics.png"), "Detected subtopics", "Videos"))
    if len(tab):
        s.findings.append("Most common detected subtopics: " + ", ".join(f"{idx} ({int(v):,})" for idx, v in tab["total"].head(5).items()) + ".")
    s.notes.append("Subtopics are keyword-derived from title, hashtags, tags and description. They describe text signals and do not prove that every listed procedure is actually taught in the video.")
    return s


# ------------------------------------------------------------------------------------------------ 10 query analysis
def query_analysis(d: pd.DataFrame, out: Path) -> Section:
    s = Section("10. Query analysis", "Which search queries and query counts are associated with the final relevant videos?")
    if "query_count" in d:
        qc = d.groupby("platform")["query_count"].agg(["count", "median", "mean", "max"]).reset_index()
        s.tables.append(_csv(qc.round(2), out, "10_query_count_summary.csv"))
        for p in P:
            x = d.loc[d["platform"] == p, "query_count"].dropna()
            if len(x):
                s.findings.append(f"{NAME[p]}: median query count per relevant video is {x.median():.0f}; maximum is {x.max():.0f}.")
    if "query_subtopics" in d:
        rows = []
        for r in d[["platform", "query_subtopics"]].itertuples(index=False):
            for q in str(r.query_subtopics).split("|"):
                q = q.strip()
                if q and q.lower() not in {"nan", "unknown"}:
                    rows.append({"platform": r.platform, "query_subtopic": q})
        if rows:
            qt = pd.DataFrame(rows).value_counts(["platform", "query_subtopic"]).reset_index(name="videos")
            s.tables.append(_csv(qt, out, "10_query_subtopic_counts.csv"))
            total = qt.groupby("query_subtopic")["videos"].sum().sort_values(ascending=False).head(15)
            s.figures.append(plots.bar_counts(total, _fig(out, "10_query_subtopics.png"), "Query-subtopic occurrences", "Occurrences"))
            s.findings.append("Most frequent query-subtopic labels among relevant rows: " + ", ".join(f"{idx} ({int(v):,})" for idx, v in total.head(5).items()) + ".")
    s.notes.append("Query/subtopic fields record how videos were discovered; they are not video-level human topic labels.")
    return s


# ------------------------------------------------------------------------------------------------ 11 text analysis
def text_analysis(d: pd.DataFrame, out: Path) -> Section:
    s = Section("11. Text analysis", "What simple text characteristics are visible in titles and hashtags?")
    cols = [c for c in ["title_words", "title_chars", "n_hashtags"] if c in d]
    if cols:
        summary = []
        for p in P:
            x = _by(d, p)
            for c in cols:
                z = pd.to_numeric(x[c], errors="coerce").dropna()
                if len(z):
                    summary.append({"platform": p, "measure": c, "median": z.median(), "q1": z.quantile(.25), "q3": z.quantile(.75), "max": z.max()})
        s.tables.append(_csv(pd.DataFrame(summary).round(2), out, "11_text_length_summary.csv"))
        if "title_words" in d:
            s.figures.append(plots.box_by_platform(d, {"Title words": "title_words", "Title characters": "title_chars"}, _fig(out, "11_title_length_box.png"), "Title length by platform"))
        for r in summary:
            if r["measure"] == "title_words":
                s.findings.append(f"{NAME[r['platform']]} title length: median {r['median']:.0f} words (middle half {r['q1']:.0f}-{r['q3']:.0f}).")
    s.notes.append("This is a descriptive text analysis of available metadata, not a semantic analysis of the video content itself.")
    return s


# ------------------------------------------------------------------------------------------------ transcripts and usable text
def transcripts(d: pd.DataFrame, out: Path) -> Section:
    s = Section("12. Transcripts and usable text", "The screening model reads title, description, hashtags and the transcript. How much text do the videos have, and is a transcript linked to being selected as relevant?")
    rows = []
    for p in P:
        x = _by(d, p)
        w = x.loc[x["has_transcript"], "transcript_words"]
        rows.append({"platform": p, "videos": len(x), "with_transcript": int(x["has_transcript"].sum()), "with_transcript_pct": round(x["has_transcript"].mean() * 100, 1),
                     "no_usable_text": int(x["no_usable_text"].sum()), "no_usable_text_pct": round(x["no_usable_text"].mean() * 100, 1),
                     "median_transcript_words": float(w.median()) if len(w) else np.nan, "median_title_description_words": float(x["text_words"].median())})
    t = pd.DataFrame(rows)
    s.tables.append(_csv(t, out, "12_text_availability_by_platform.csv"))
    st = d["transcript_status"].replace("", "unknown")
    stat = pd.crosstab(st, d["platform"]).reindex(columns=list(P), fill_value=0)
    s.tables.append(_csv(stat.reset_index().rename(columns={"transcript_status": "status"}), out, "12_transcript_status_by_platform.csv"))
    s.figures.append(plots.stacked_share(pd.crosstab(d["platform"], st), _fig(out, "12_transcript_status.png"), "Transcript status, by platform"))
    for r in rows:
        s.findings.append(f"{NAME[r['platform']]}: {r['with_transcript']:,} of {r['videos']:,} videos ({r['with_transcript_pct']}%) have a usable transcript; {r['no_usable_text']:,} ({r['no_usable_text_pct']}%) have almost no text at all (fewer than 6 words in title, description and transcript).")
    if "screening_label" in d and d["screening_label"].nunique() > 1:
        ct = pd.crosstab(d["has_transcript"], d["screening_label"])
        s.tables.append(_csv(ct.reset_index(), out, "12_transcript_vs_screening_label.csv"))
        rel = d.groupby("has_transcript")["screening_label"].apply(lambda z: (z == "relevant").mean() * 100)
        chi, pv, _, _ = sps.chi2_contingency(pd.crosstab(d["has_transcript"], d["screening_label"] == "relevant"))
        yes, no = rel.get(True, np.nan), rel.get(False, np.nan)
        s.findings.append(f"Share labeled relevant: {yes:.1f}% among videos with a transcript vs {no:.1f}% among videos without (chi-square {fmt_p(pv)}). "
                          + ("Videos with a transcript were not favoured by the screening." if yes <= no else "Videos with a transcript were favoured by the screening, probably because the model can read more text for them.")
                          + " (A gap of this kind would be a bias in the selection, not a property of the topic.)")
    s.notes.append("Most TikTok videos carry only a caption; YouTube transcripts can be missing because of platform blocks or disabled captions. Missing text lowers what the model and the keyword checklist can see; it does not mean the video is irrelevant.")
    return s


# ------------------------------------------------------------------------------------------------ techniques
def techniques(d: pd.DataFrame, out: Path) -> Section:
    s = Section("13. Techniques mentioned", "Which holding, restraint and safety terms appear in the text? A video can mention several. These are word matches in the title, hashtags, tags and description, not proof that the technique is shown.")
    cols = [c for c in d.columns if c.startswith("tech_")]
    from .features import TECH_LABELS

    t = pd.DataFrame({TECH_LABELS[c]: {p: _by(d, p)[c].mean() * 100 for p in P} for c in cols}).T
    t["total_videos_mentioning"] = [int(d[c].sum()) for c in cols]
    s.tables.append(_csv(t.round(1).reset_index().rename(columns={"index": "technique"}), out, "13_technique_mentions_by_platform.csv"))
    s.figures.append(plots.grouped_bars(t[list(P)], _fig(out, "13_technique_mentions.png"), "Videos mentioning each technique (% of the platform's videos)"))
    top = t["total_videos_mentioning"].sort_values(ascending=False)
    s.findings.append("Most mentioned: " + ", ".join(f"{k} ({int(v):,})" for k, v in top.head(4).items()) + ".")
    s.findings.append(f"{(d[cols].any(axis=1)).mean() * 100:.0f}% of the videos mention at least one of these techniques; the rest are described only in general terms.")
    if "screening_label" in d and d["screening_label"].nunique() > 1:
        g = pd.DataFrame({TECH_LABELS[c]: d.groupby("screening_label")[c].mean() * 100 for c in cols}).T.round(1)
        s.tables.append(_csv(g.reset_index().rename(columns={"index": "technique"}), out, "13_technique_mentions_by_label.csv"))
        anyt = d[cols].any(axis=1).groupby(d["screening_label"]).mean() * 100
        s.findings.append("Share of videos mentioning at least one technique, by screening group: " + ", ".join(f"{k} {v:.0f}%" for k, v in anyt.items()) + " (expected to differ, because the screening rules use the same kind of words).")
    s.notes.append("The same vocabulary drives the screening rules, so this section describes the data and is not an independent check of the screening.")
    return s


# ------------------------------------------------------------------------------------------------ screening outcome
def outcome(d: pd.DataFrame, out: Path) -> Section:
    s = Section("14. Screening outcome", "How did the screening sort the videos (relevant, audit, irrelevant), and how do the groups differ? Only meaningful on the dataset that holds all three groups.")
    if "screening_label" not in d or d["screening_label"].nunique() < 2:
        s.notes.append("This dataset holds a single screening group, so there is nothing to compare. Run the 'all' dataset for this section.")
        return s
    order = [g for g in ("relevant", "audit", "irrelevant") if g in set(d["screening_label"])]
    by_p = pd.crosstab(d["platform"], d["screening_label"]).reindex(columns=order, fill_value=0)
    s.tables.append(_csv(by_p.reset_index(), out, "14_label_by_platform.csv"))
    s.tables.append(_csv(pd.crosstab([d["screening_label"], d["zone"]], d["platform"]).reset_index(), out, "14_zone_by_label_platform.csv"))
    s.figures.append(plots.stacked_share(by_p, _fig(out, "14_label_share_by_platform.png"), "Screening outcome, by platform"))
    for p in P:
        r = by_p.loc[p]
        s.findings.append(f"{NAME[p]}: " + ", ".join(f"{g} {int(r[g]):,} ({r[g] / r.sum() * 100:.0f}%)" for g in order) + ".")
    def rate(col, name):
        g = d.groupby(col).agg(videos=("video_uid", "size"), relevant=("screening_label", lambda z: int((z == "relevant").sum())), audit=("screening_label", lambda z: int((z == "audit").sum())),
                               irrelevant=("screening_label", lambda z: int((z == "irrelevant").sum())))
        g["relevant_pct"] = (g["relevant"] / g["videos"] * 100).round(1)
        g = g.sort_values("videos", ascending=False)
        s.tables.append(_csv(g.reset_index(), out, name))
        return g
    st = rate("subtopic", "14_label_by_setting.csv")
    rate("age_group", "14_label_by_age_group.csv"); rate("language_detected", "14_label_by_language.csv")
    top = st[st["videos"] >= 20].sort_values("relevant_pct", ascending=False)
    if len(top):
        s.findings.append("Share relevant by setting (settings with at least 20 videos): " + "; ".join(f"{i} {r.relevant_pct:.0f}% of {int(r.videos):,}" for i, r in top.head(4).iterrows()) + ".")
    s.figures.append(plots.hist_by_group(d, "score", "screening_label", order, _fig(out, "14_score_by_label.png"), "Screening score by group", "Calibrated score"))
    rows = []
    for col, lab, lg in (("duration_seconds", "duration (s)", True), ("views", "views", True), ("text_words", "title+description words", False), ("transcript_words", "transcript words (videos with a transcript)", False)):
        base = d[d["has_transcript"]] if col == "transcript_words" else d   # transcript length only among videos that have one
        groups = [base.loc[base["screening_label"] == g, col].dropna() for g in order]
        kw = sps.kruskal(*[x for x in groups if len(x) > 1]) if sum(len(x) > 1 for x in groups) > 1 else None
        rows.append({"measure": lab, **{f"median_{g}": float(x.median()) if len(x) else np.nan for g, x in zip(order, groups)}, "kruskal_p": float(kw.pvalue) if kw else np.nan})
        if col in ("duration_seconds", "views"):
            s.figures.append(plots.box_by_group(d, col, "screening_label", order, _fig(out, f"14_{col}_by_label.png"), f"{lab.capitalize()} by screening group", log_y=lg, ylabel=lab))
    cmp_ = pd.DataFrame(rows)
    s.tables.append(_csv(cmp_.round(3), out, "14_group_comparison.csv"))
    for r in rows:
        s.findings.append(f"{r['measure'].capitalize()}: " + ", ".join(f"{g} median {r['median_' + g]:,.0f}" for g in order) + f" (Kruskal-Wallis {fmt_p(r['kruskal_p']) if r['kruskal_p'] == r['kruskal_p'] else 'n/a'}).")
    hum = d.groupby("screening_label")["label_source"].apply(lambda z: (z == "human").mean() * 100).round(0)
    s.notes.append("Videos labeled by humans (" + ", ".join(f"{g} {hum[g]:.0f}%" for g in order) + " of each group) keep their human label; their score is in-sample. The audit group is undecided, not irrelevant, so group differences describe how the model sorted the videos, not ground truth.")
    return s


SECTIONS = {
    "quality": quality, "features": features_summary, "overview": overview,
    "duration": duration, "views": views, "engagement": engagement,
    "upload_trends": upload_trends, "language_country": language_country,
    "creators": creators, "subtopics": subtopics, "query_analysis": query_analysis,
    "text_analysis": text_analysis, "transcripts": transcripts, "techniques": techniques, "outcome": outcome,
}
