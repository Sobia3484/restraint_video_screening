"""Build the expert-guideline checklist workbook for the restraint/immobilization project.

Input : relevant_strict_videos.csv (output of screening)
Output: Restraint_Immobilization_Expert_Checklist.xlsx
Usage : python scripts/build_expert_checklist.py <relevant_csv> <out_xlsx> [--min-score 0.90]
Sheets mirror the CPR/choking checklist workbook; all summary numbers are live formulas.
"""
import re, sys, argparse
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter as L
from openpyxl.formatting.rule import CellIsRule
from openpyxl.worksheet.datavalidation import DataValidation

NAVY = "1F3864"
HDR_FILL = PatternFill("solid", fgColor=NAVY)
HDR_FONT = Font(bold=True, color="FFFFFF")
WRAP = Alignment(wrap_text=True, vertical="top")
HC = Alignment(wrap_text=True, vertical="center", horizontal="center")

# ---------------------------------------------------------------- checklist points
# (short name, what the video should show, applies to, guideline basis, keyword hint)
POINTS = [
 ("Holding only when needed / least restrictive option", "Says holding or restraint is used only when necessary, after alternatives, and in the least restrictive way", "All", "RCN 2019: holding without assent is a last resort; use the least restrictive option.", "last resort, only if necessary, least restrictive"),
 ("Preparation and information", "Prepares and informs the child and parent before the procedure (what will happen, play or child-life preparation)", "All", "RCN 2019: need for holding is reduced through information and preparation.", "prepare, explain to the child, child life, what to expect"),
 ("Distraction / pain relief", "Uses distraction, comfort measures or analgesia (e.g. toys, bubbles, sucrose, topical anaesthetic, feeding)", "All", "RCN 2019 (distraction, analgesia, sedation); CMAJ 2010 (pain-reduction measures).", "distract, bubbles, toy, numbing cream, sucrose, breastfeed"),
 ("Consent / assent / permission", "Obtains or mentions parent consent and the child's assent; explains before holding", "All", "RCN 2019: holding without assent is a last resort; AAPD: consent and assent.", "consent, assent, permission, ask the child"),
 ("Upright (not lying flat) for injections", "Child held upright or semi-upright instead of lying flat for needle procedures", "Needle procedures", "CMAJ 2010 (HELPinKIDS): hold or position upright instead of supine for injections.", "upright, sitting up, instead of lying down"),
 ("Parent / caregiver comfort hold", "Parent holds the child in a comfort hold (bear hug, lap hold, skin-to-skin) instead of a stranger forcing the child", "Non-dental, non-trauma", "CMAJ 2010: comfortable parent hold such as a bear hug or lap.", "bear hug, comfort hold, lap, cuddle, skin-to-skin"),
 ("Swaddling / wrapping for the procedure", "Infant wrapped so the limbs are contained for the procedure (not sleep swaddling)", "Infant procedures", "RCN 2019: age-appropriate techniques such as wrapping and splinting.", "swaddle, wrap, blanket, papoose, burrito"),
 ("Minimum force", "Uses only the force needed, gentle contact, not forceful pinning", "All", "RCN 2019: limited force, least restrictive option.", "gentle, light, minimum force, no force"),
 ("Minimum time / release when done", "Holds for the shortest time and releases the child as soon as the procedure is over", "All", "RCN 2019: minimum time; AAP/AAPD: child not left restrained.", "briefly, shortest, release, as soon as done"),
 ("Airway and chest not obstructed", "Device or hold does not cover the face or restrict chest movement; head position keeps the airway open", "All", "AAP/AAPD 2025: avoid airway obstruction and chest restriction; check head position.", "airway, chest, face uncovered, breathe freely"),
 ("Monitoring / never left unattended", "Watches breathing and colour during restraint; child never left alone", "All", "AAP/AAPD 2025: respiratory movement checked frequently; never unattended.", "monitor, watch breathing, never leave, unattended"),
 ("Hand or foot exposed / circulation", "Keeps a hand or foot visible and checks circulation; wrap or strap not too tight", "Devices and wraps", "AAP/AAPD 2025: a hand or foot stays exposed.", "hand exposed, circulation, not too tight"),
 ("Correct use of device (papoose / immobiliser)", "Shows or explains correct fitting and use of the stabilization or immobilization device", "Device videos", "AAPD: indications, equipment and technique for protective stabilization.", "papoose, Pigg-O-Stat, immobiliser, strap, stabilization"),
 ("Spinal motion restriction (collar, limited board time)", "Fitted collar, scoop or board used for trauma, with limited time on the backboard (term: spinal motion restriction)", "Trauma / spinal", "ACS COT, ACEP, NAEMSP 2018: SMR with properly fitted collar, limited backboard time.", "spinal motion restriction, c-collar, backboard"),
 ("Child-size padding (large head)", "Pads under the child's shoulders or torso because of the proportionally larger head", "Trauma / spinal", "ACS COT, ACEP, NAEMSP 2018: padding for children (larger head).", "padding under shoulders, towel under back"),
 ("Risks of restraint / what to avoid stated", "Explains risks, complications or unsafe practices to avoid (force, prone pinning, chest or face pressure)", "All", "2024 scoping reviews; RCN 2019; AAP/AAPD 2025.", "risks, complications, unsafe, never, avoid forcing"),
]
NP = len(POINTS)
assert NP == 16
_P = POINTS
POINTS = [_P[0], _P[1], _P[2], _P[3], _P[4], _P[5], _P[6],
 ("Minimum force and minimum time", "Uses only the force needed (gentle contact, not forceful pinning) and releases the child as soon as the procedure is over", "All", "RCN 2019: limited force, least restrictive option, minimum time.", "gentle, minimum force, shortest, release when done"),
 _P[9],
 ("Monitoring, hand/foot visible, never left alone", "Watches breathing and colour during restraint, keeps a hand or foot visible, wrap or strap not too tight, child never left alone", "All", "AAP/AAPD 2025: respiratory movement checked frequently, a hand or foot stays exposed, never unattended.", "monitor, watch breathing, hand exposed, never leave"),
 _P[12],
 ("Spinal motion restriction (collar, limited board time, padding)", "Fitted collar, scoop or board for trauma, limited time on the backboard, and padding under a child's shoulders or torso because of the larger head (term: spinal motion restriction)", "Trauma / spinal", "ACS COT, ACEP, NAEMSP 2018: SMR with fitted collar, limited backboard time, padding for children.", "spinal motion restriction, c-collar, backboard, padding under shoulders"),
 _P[15]]
NP = len(POINTS)

I = re.I
def rx(p): return re.compile(p, I)
TICK = [
 rx(r"last resort|only (when|if) (it'?s |it is )?(necessary|needed)|if (really )?necessary|when necessary|unavoidable|least restrictive|before (holding|restrain)|avoid(ing)? (restraint|holding)|alternatives? to (restraint|holding)"),
 rx(r"\bprepar(e|ing|ation)\b|child life|play (specialist|therapy)|what to expect|social story|explain(s|ed|ing)? (to )?(the |your )?(child|kid|them|him|her|what)|talk(ing)? (to|with) (the |your )?(child|kid)|preparaci[oó]n"),
 rx(r"distract|bubbles|\btoys?\b|\btablet\b|sing(ing)? |numbing|lidocaine|\bemla\b|sucrose|sugar water|breast ?feed|topical (an[a]?esth)|\bbuzzy\b|shotblocker|comfort position|distrae|\bice\b"),
 rx(r"\bconsent|\bassent|permission|ask(ing)? (the |your )?(child|parent)|consentimiento"),
 rx(r"upright|sit(ting)? up|\bseated\b|sit(ting)? (on|in) (a |the |your )?lap|instead of (lying|laying)|not (lying|laying) (down|flat)|vertical|sentad[oa]"),
 rx(r"bear ?hug|comfort(ing)? (hold|position)|cuddl|lap hold|\bhugg?(ing|ed)?\b|(parent|mom|mum|dad|mother|father)'?s? (can )?(hold|holds|holding|lap)|kangaroo|skin[- ]to[- ]skin|en brazos|abrazo|brazos de"),
 rx(r"swaddl|wrap(ped|ping|s)? (the |your |a |in a |him|her|baby|infant)|blanket wrap|burrito|papoose|envuelt|arrop"),
 rx(r"\bgentl(e|y)\b|least restrictive|minimum (force|restraint)|without force|no force|limited force|don'?t (force|squeeze|pin)|firm but (gentle|kind)|\bsuave"),
 rx(r"as (short|briefly) as possible|shortest|release (the )?(hold|him|her|child|baby)|until (the )?procedure (is )?(done|over|finished)|few seconds|minimum time|only (for )?(as long as|during)"),
 rx(r"\bairway\b|chest (restrict|obstruct|pressure)|obstruct|suffocat|asphyx|head position|neck (flexion|position)|breathe freely|free to breathe|nose and mouth|v[ií]a a[eé]rea"),
 rx(r"\bmonitor(ing|ed)?\b|watch(ing)? (the |your )?(child|baby|breathing|colou?r)|never leave|don'?t leave|unattended|pulse ox|\bsupervis|check(ing)? (the )?(breathing|colou?r|pulse|oxygen|sats?)|nunca (dej|solo)|vigila"),
 rx(r"(hands?|foot|feet|fingers|toes|limbs?) (exposed|free|visible)|circulation|capillary|too tight|not too tight|neurovascular|colou?r of (the )?(skin|fingers|toes)"),
 rx(r"papoose|pigg-?o-?stat|immobili[sz]er|immobili[sz](ation|ing|e)? (device|board|chair)|vacuum (splint|mattress)|protective stabili[sz]ation|restraint (device|chair|board)|\bvelcro|\bstraps?\b|\bsandbags?\b|inmovilizador|dispositivo de inmovilizaci[oó]n|inmovilizaci[oó]n"),
 rx(r"spinal motion|spine motion|c-?spine|cervical collar|c-?collar|neck collar|\bcollar\b|backboard|spine ?board|long ?board|scoop stretcher|log ?roll|spinal (immobili|precaution|restriction)|\bsmr\b"),
 rx(r"padding (under|beneath)|pad(ded)? (under|beneath)|(towel|blanket|roll)s? (under|beneath)|under (the |his |her )?(shoulders?|torso)|larger head|proportionally|occiput|neutral (position|alignment)"),
 rx(r"\brisks?\b|danger|complication|\bunsafe\b|harm(ful)?\b|injur(y|ies)|warning|mistake|never (restrain|hold|force|leave)|avoid (force|forcing|restrain|pinning|holding)|traumatic|riesgo|peligro"),
]
CROSS = {
 8: rx(r"(cover(ing|s)?|block(ing|s)?|press(ing)? on|lean(ing)? on) (the |his |her )?(face|nose|mouth|airway|chest)"),
 12: rx(r"hold(ing|s)? (him|her|them|the (baby|child|kid|toddler)) down|pin(ning|ned)? (him|her|them|the (baby|child|kid))? ?down|force(d)? (him|her|them) (to )?(hold|stay|still)|restrain(ed|ing)? (tightly|hard)|sit(ting)? on (him|her|the child)"),
}
def _merge(T):
    j = lambda *ix: rx("|".join(T[i].pattern for i in ix))
    return [T[0], T[1], T[2], T[3], T[4], T[5], T[6], j(7, 8), T[9], j(10, 11), T[12], j(13, 14), T[15]]
# applicability by video type: indices (0-based) that are N/A when no keyword is found
NA = {
 "Needle procedure": {10, 11},
 "Imaging": {4, 11},
 "Dental": {4, 11},
 "Trauma / spinal / splint": {4, 5, 6},
 "Neonatal": {4, 10, 11},
 "Examination": {10, 11},
 "General / other": {11},
}
TICK = _merge(TICK)
assert len(TICK) == NP
TYPE_RX = [
 ("Trauma / spinal / splint", rx(r"spinal|c-?spine|collar|backboard|spine ?board|splint|fractur|\btrauma\b|\bems\b|ambulance|paramedic")),
 ("Dental", rx(r"dental|dentist|odont|pediatric dentistry")),
 ("Imaging", rx(r"\bmri\b|x-?ray|\bct\b|radiograph|radiolog|pigg|rayos|resonancia")),
 ("Needle procedure", rx(r"vaccin|inject|needle|blood draw|venipunct|\biv\b|cannula|\bshots?\b|heel ?(stick|prick)|phleb|inyecc|vacuna|\bpedia\b")),
 ("Neonatal", rx(r"\bnicu\b|newborn|neonat|preterm|premie")),
 ("Examination", rx(r"\bear (check|exam)|otoscop|throat|\bexam(ination)?\b|check-?up|swab")),
]
NEG = rx(r"never|don'?t|do not|avoid|without|\bno\b|\bnot\b|instead of|research|stud(y|ies)|stress|trauma|causes?|rather than|alternative")
HEALTH_RX = rx(r"hospital|clinic|medical|health|nurs|\bdr\.?\b|\bmd\b|children'?s|pediatric|paediatric|dental|dentist|radiolog|\bnhs\b|\brn\b|\bpa\b|physician|\bered?\b")

def clean_txt(s): return "" if pd.isna(s) else str(s)

def snippet(text, m, w=75):
    a, b = max(0, m.start() - w), min(len(text), m.end() + w)
    s = re.sub(r"\s+", " ", text[a:b]).strip()
    return ("…" if a > 0 else "") + s + ("…" if b < len(text) else "")

def analyse(row):
    title, desc = clean_txt(row.title), clean_txt(row.description)
    tags = " ".join([clean_txt(row.hashtags), clean_txt(row.tags)])
    tr = clean_txt(row.transcript) if str(row.transcript_status) == "ok" else ""
    meta = " . ".join(x for x in (title, desc, tags) if x)
    words = len(re.sub(r"[#@]\w+", " ", meta).split()) + len(tr.split())
    has_tr = len(tr.split()) >= 20
    typ_text = meta + " " + tr
    typ = "General / other"
    for name, r in TYPE_RX:
        if r.search(typ_text): typ = name; break
    # age group
    low = typ_text.lower()
    age = "Not stated in text"
    mm = re.search(r"(\d+)\s*(?:-| )?(?:months?|mos?|mths?)\b", low)
    yy = re.search(r"(\d+)\s*(?:-| )?(?:years?|yrs?|yo|y/o)\b", low)
    if re.search(r"infant|newborn|\bbaby\b|\bbabies\b|beb[eé]|neonat|\bnicu\b|\bpreterm", low) or mm: age = "Infant"
    elif yy and int(yy.group(1)) <= 5: age = "Young child (1-5 y)"
    elif re.search(r"toddler|preschool|ni[nñ]o|\bkids?\b|\bchild(ren)?\b|\bpedia", low): age = "Child (age not given)"
    d = row.duration_seconds
    length = "Short (<60s)" if d < 60 else ("Medium (60-180s)" if d <= 180 else "Long (>180s)")
    if re.search(r"step|how to|tutorial|demonstrat|training|class|learn|technique", low): style = "Training / demonstration style"
    elif re.search(r"my (son|daughter|baby|boy|girl|toddler|kid)|mi (beb[eé]|hij)|vlog|\bday in\b|\bpov\b", low): style = "Personal / parent clip"
    elif re.search(r"tips?|why|did you know|explain|facts?", low): style = "Short educational tip"
    else: style = "Not determinable from text (needs viewing)"
    src = "Health organisation / clinician (auto-guess)" if HEALTH_RX.search(clean_txt(row.creator_name)) else "Individual / other (auto-guess)"
    level = ("Transcript + title/description" if has_tr else ("Title/description only" if words >= 6 else "Not analysable"))
    vals, ev = [], []
    for i, rg in enumerate(TICK):
        res, where, snip = None, "", ""
        if level != "Not analysable":
            for src_name, txt in (("transcript", tr), ("title/description", meta)):
                if not txt: continue
                if i in CROSS:
                    mc = next((m for m in CROSS[i].finditer(txt) if not NEG.search(txt[max(0, m.start() - 60):m.start() + 5])), None)
                    if mc: res, where, snip = "✗", src_name, snippet(txt, mc); break
                m = rg.search(txt)
                if m: res, where, snip = "✓", src_name, snippet(txt, m); break
        if res is None:
            res = "N/A" if i in NA[typ] else "?"
        vals.append(res)
        if res in ("✓", "✗"): ev.append((i, res, where, snip))
    return dict(typ=typ, age=age, length=length, style=style, src=src, level=level, vals=vals, ev=ev)

# ---------------------------------------------------------------- workbook helpers
def hdr(ws, row, headers, widths=None, height=45):
    for c, h in enumerate(headers, 1):
        cell = ws.cell(row, c, h); cell.fill = HDR_FILL; cell.font = HDR_FONT; cell.alignment = HC
    ws.row_dimensions[row].height = height
    if widths:
        for c, w in enumerate(widths, 1): ws.column_dimensions[L(c)].width = w

def title(ws, text, sub=None, sub_h=60):
    ws["A1"] = text; ws["A1"].font = Font(bold=True, size=14, color=NAVY)
    if sub:
        ws["A2"] = sub; ws["A2"].alignment = WRAP; ws["A2"].font = Font(italic=True, color="404040")
        ws.row_dimensions[2].height = sub_h

def cf_marks(ws, rng):
    for val, bg, fg in (("✓", "C6EFCE", "006100"), ("✗", "FFC7CE", "9C0006"), ("?", "FFEB9C", "7F6000"), ("N/A", "EDEDED", "7F7F7F")):
        ws.conditional_formatting.add(rng, CellIsRule(operator="equal", formula=[f'"{val}"'],
            fill=PatternFill("solid", bgColor=bg, fgColor=bg), font=Font(color=fg)))

def marks_dv(ws, rng):
    dv = DataValidation(type="list", formula1='"✓,✗,?,N/A"', allow_blank=False, showErrorMessage=True,
                        errorStyle="warning", errorTitle="Check value", error="Use only ✓  ✗  ?  N/A")
    ws.add_data_validation(dv); dv.add(rng)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("out")
    ap.add_argument("--min-score", type=float, default=0.90)
    a = ap.parse_args()
    thr = a.min_score
    full = pd.read_csv(a.csv)
    df = full[full.score >= thr].sort_values("score", ascending=False).reset_index(drop=True)
    N = len(df); R0 = 5; R1 = R0 + N - 1
    res = [analyse(r) for r in df.itertuples()]

    wb = Workbook()
    ws_fmt = wb.active; ws_fmt.title = "Format"
    ws_sum = wb.create_sheet("Summary"); ws_det = wb.create_sheet("Checklist_Detail")
    ws_gd = wb.create_sheet("Checklist_Guide"); ws_ev = wb.create_sheet("Evidence")
    ws_ml = wb.create_sheet("Method_and_Limits")
    pts = [f"{i+1}. {p[0]}" for i, p in enumerate(POINTS)]
    PC0 = 15
    PCOL = [L(PC0 + i) for i in range(NP)]
    pfirst, plast = PCOL[0], PCOL[-1]
    cOK, cX, cQ, cAP, cCOV, cNOTE, cSRC = (L(PC0 + NP + k) for k in range(7))

    # ---------------- Checklist_Detail
    title(ws_det, f"Checklist detail - one row per video (model score ≥ {thr:.2f}, {N} videos), one column per expert-guideline point",
          "Legend:  ✓ = shown/said correctly in the video   ✗ = video contradicts the guideline (e.g. covers the face or chest, forceful pinning)   ? = not verified / not shown   N/A = point does not apply to this video type.   "
          "Cells in the point columns are PRE-FILLED by an automatic keyword pass (text only). Watch each video and overwrite wrong cells.", 48)
    heads = ["S.No", "Title", "Platform", "URL", "Creator", "Source type (auto-guess)", "Upload date", "Model score", "Age group (auto)",
             "Video type (auto)", "Duration (s)", "Length category", "Presentation style (auto-guess)", "Evidence level"] + pts + \
            ["✓ count", "✗ count", "? count", "Applicable points", "Coverage (✓ / applicable)", "Reviewer notes", "Label source"]
    hdr(ws_det, 4, heads, [6, 34, 10, 40, 22, 26, 11, 9, 14, 20, 9, 14, 28, 24] + [14] * NP + [8, 8, 8, 10, 14, 28, 11], 75)
    for k, (r, an) in enumerate(zip(df.itertuples(), res)):
        row = R0 + k
        vals = [k + 1, clean_txt(r.title)[:300], clean_txt(r.platform).capitalize(), r.url, clean_txt(r.creator_name), an["src"],
                clean_txt(r.upload_date), round(float(r.score), 4), an["age"], an["typ"], round(float(r.duration_seconds)), an["length"], an["style"], an["level"]] + an["vals"]
        for c, v in enumerate(vals, 1):
            cell = ws_det.cell(row, c, v); cell.alignment = Alignment(vertical="top", wrap_text=(c in (2, 5, 6, 13)))
            if c >= PC0: cell.alignment = Alignment(horizontal="center", vertical="top")
        rg = f"{pfirst}{row}:{plast}{row}"
        ws_det[f"{cOK}{row}"] = f'=COUNTIF({rg},"✓")'
        ws_det[f"{cX}{row}"] = f'=COUNTIF({rg},"✗")'
        ws_det[f"{cQ}{row}"] = f'=SUMPRODUCT(--({rg}="?"))'
        ws_det[f"{cAP}{row}"] = f'=COUNTA({rg})-COUNTIF({rg},"N/A")'
        ws_det[f"{cCOV}{row}"] = f'=IF({cAP}{row}=0,0,{cOK}{row}/{cAP}{row})'; ws_det[f"{cCOV}{row}"].number_format = "0%"
        ws_det[f"{cSRC}{row}"] = clean_txt(r.label_source)
        ws_det.cell(row, 8).number_format = "0.000"
    ws_det.freeze_panes = "C5"; ws_det.auto_filter.ref = f"A4:{cSRC}{R1}"
    cf_marks(ws_det, f"{pfirst}{R0}:{plast}{R1}"); marks_dv(ws_det, f"{pfirst}{R0}:{plast}{R1}")

    # ---------------- Evidence
    title(ws_ev, "Evidence for every ✓ / ✗ decision (text snippet that triggered it) - check before trusting a ✓",
          "Snippets come from the automatic first pass. Keyword hits can be negated or off-topic: a snippet is a lead to check in the video, not proof.", 32)
    hdr(ws_ev, 3, ["S.No", "Video title", "Checklist point", "Result", "Found in", "Text snippet"], [6, 40, 34, 8, 18, 100], 30)
    er = 4
    for k, an in enumerate(res):
        for (i, rs, where, sn) in an["ev"]:
            for c, v in enumerate([k + 1, clean_txt(df.title[k])[:120], POINTS[i][0], rs, where, sn], 1):
                ws_ev.cell(er, c, v).alignment = WRAP
            er += 1
    ws_ev.freeze_panes = "A4"; ws_ev.auto_filter.ref = f"A3:F{max(er-1,4)}"; cf_marks(ws_ev, f"D4:D{max(er-1,4)}")

    # ---------------- Format
    title(ws_fmt, f"Expert-guideline checklist - safe restraint and immobilization videos for infants and young children (model score ≥ {thr:.2f}, {N} videos)",
          f"Checklist column: ✓ = point shown/said correctly, ✗ = video contradicts the guideline, ? = not verified, numbers refer to points 1-{NP} in sheet Checklist_Guide. Updates automatically from Checklist_Detail.", 40)
    hdr(ws_fmt, 4, ["S.No", "Title", "Platform", "Checklist", "URL", "Presentation"], [6, 40, 10, 54, 34, 36], 30)
    def part(sym, row):
        return "&".join(f'IF(Checklist_Detail!{PCOL[i]}{row}="{sym}","{i+1} ","")' for i in range(NP))
    for k in range(N):
        row = R0 + k
        ws_fmt.cell(row, 1, f"=Checklist_Detail!A{row}"); ws_fmt.cell(row, 2, f"=Checklist_Detail!B{row}"); ws_fmt.cell(row, 3, f"=Checklist_Detail!C{row}")
        ws_fmt.cell(row, 4, f'="✓ "&TRIM({part("✓",row)})&"  |  ✗ "&TRIM({part("✗",row)})&"  |  ? "&TRIM({part("?",row)})&CHAR(10)&"Coverage: "&TEXT(Checklist_Detail!{cCOV}{row},"0%")&" of applicable points ticked  ("&Checklist_Detail!N{row}&")"')
        ws_fmt.cell(row, 5, f"=Checklist_Detail!D{row}")
        ws_fmt.cell(row, 6, f'=Checklist_Detail!M{row}&" · "&Checklist_Detail!L{row}&" · "&Checklist_Detail!J{row}&" / "&Checklist_Detail!I{row}')
        for c in range(1, 7): ws_fmt.cell(row, c).alignment = WRAP
        ws_fmt.row_dimensions[row].height = 44
    ws_fmt.freeze_panes = "C5"

    # ---------------- Summary
    D = lambda col: f"Checklist_Detail!{col}{R0}:{col}{R1}"
    title(ws_sum, "Summary (all numbers are live formulas on Checklist_Detail)")
    rows = [
        (f"Videos in this file (model score ≥ {thr:.2f})", f"=COUNTA({D('A')})"),
        ("YouTube", f'=COUNTIF({D("C")},"Youtube")'),
        ("TikTok", f'=COUNTIF({D("C")},"Tiktok")'),
        ("Human-labelled (score is in-sample, see Method_and_Limits)", f'=COUNTIF({D(cSRC)},"human")'),
        ("Model-selected", f'=COUNTIF({D(cSRC)},"model")'),
        ("With transcript analysed", f'=COUNTIF({D("N")},"Transcript*")'),
        ("Title/description only", f'=COUNTIF({D("N")},"Title/description only")'),
        ("Not analysable (almost no text)", f'=COUNTIF({D("N")},"Not analysable")'),
        ("Average checklist coverage - all videos", f"=AVERAGE({D(cCOV)})"),
        ("Average coverage - YouTube", f'=AVERAGEIF({D("C")},"Youtube",{D(cCOV)})'),
        ("Average coverage - TikTok", f'=AVERAGEIF({D("C")},"Tiktok",{D(cCOV)})'),
        ("Average coverage - videos WITH transcript", f'=IFERROR(AVERAGEIF({D("N")},"Transcript*",{D(cCOV)}),0)'),
        ("Average coverage - title/description only", f'=IFERROR(AVERAGEIF({D("N")},"Title/description only",{D(cCOV)}),0)'),
    ]
    for k, (lab, f) in enumerate(rows):
        r = 3 + k
        ws_sum.cell(r, 1, lab); c = ws_sum.cell(r, 2, f)
        if "coverage" in lab.lower(): c.number_format = "0.0%"
    note_r = 3 + len(rows)
    ws_sum.cell(note_r, 1, 'Coverage is a LOWER BOUND: a video may show a point without saying it, so "?" does not mean the video is wrong.').alignment = WRAP
    ws_sum.row_dimensions[note_r].height = 32
    th = note_r + 2
    hdr(ws_sum, th, ["Checklist point", "Applicable videos", "✓ found", "✗ contradicts", "? not verified", "% ✓ of applicable", "% ✗ of applicable"], [58, 16, 12, 14, 14, 16, 16], 32)
    for i in range(NP):
        r = th + 1 + i; col = PCOL[i]
        ws_sum.cell(r, 1, pts[i])
        ws_sum.cell(r, 2, f'=COUNTA({D(col)})-COUNTIF({D(col)},"N/A")')
        ws_sum.cell(r, 3, f'=COUNTIF({D(col)},"✓")'); ws_sum.cell(r, 4, f'=COUNTIF({D(col)},"✗")')
        ws_sum.cell(r, 5, f'=SUMPRODUCT(--({D(col)}="?"))')
        ws_sum.cell(r, 6, f"=IF(B{r}=0,0,C{r}/B{r})").number_format = "0%"
        ws_sum.cell(r, 7, f"=IF(B{r}=0,0,D{r}/B{r})").number_format = "0%"

    # ---------------- Checklist_Guide
    title(ws_gd, "Checklist points and the expert guideline behind each point")
    hdr(ws_gd, 2, ["#", "Point", "What the video should show / say", "Applies to", "Expert guideline basis", "Auto-detection (text keywords)"], [5, 36, 56, 20, 62, 44], 30)
    for i, p in enumerate(POINTS):
        for c, v in enumerate([i + 1, p[0], p[1], p[2], p[3], p[4]], 1): ws_gd.cell(3 + i, c, v).alignment = WRAP
        ws_gd.row_dimensions[3 + i].height = 52
    r = 3 + NP + 1
    notes = [
        f"Symbols: ✓ = shown or said correctly. ✗ = video contradicts the guideline (only with positive evidence). ? = not verified (NOT a failure of the video). N/A = point does not apply to this kind of video (e.g. point {NP-1} only for trauma / spinal videos, point 5 only for needle procedures). If unsure whether a point applies, use ? not N/A.",
        "Not every point applies to every video type, so the number of applicable points differs per video. Coverage = ✓ / applicable points.",
        "Sources (details in docs/screening_criteria.md, sections 2 and 13):",
        "1. Royal College of Nursing (2019). Restrictive physical intervention and the clinical holding of children and young people. https://www.rcn.org.uk/-/media/Royal-College-Of-Nursing/Documents/Publications/2019/October/007-746.pdf",
        "2. American Academy of Pediatric Dentistry. Use of Protective Stabilization for Pediatric Dental Patients (Reference Manual, rev. 2024). https://www.aapd.org/research/oral-health-policies--recommendations/protective-stabilization-for-the-pediatric-dental-patients/",
        "3. AAP and AAPD (2025). Guidelines for Monitoring and Management of Pediatric Patients Before, During, and After Sedation. Pediatr Dent 47(6):E100-E128. https://www.aapd.org/globalassets/media/policies_guidelines/bp_monitoringsedation25.pdf",
        "4. Taddio A et al. (2010). Reducing the pain of childhood vaccination (HELPinKIDS). CMAJ 182(18):1989. https://www.cmaj.ca/content/182/18/1989",
        "5. ACS COT, ACEP, NAEMSP (2018). Spinal Motion Restriction in the Trauma Patient: a joint position statement. https://naemsp.org/2020-7-25-article-bites-21-spinal-motion-restriction-in-trauma-a-joint-position-statement/",
        "6. Two 2024 scoping reviews on procedural restraint in children (J Pediatr Nurs 2024;79:e110-e118; J Child Health Care 2024).",
        "Limit: the points are the project's own reading of these sources. The dental source was read from its abstract only; check the device point against the full AAPD text before finalizing.",
    ]
    for t in notes:
        ws_gd.cell(r, 1, t).alignment = WRAP; ws_gd.merge_cells(start_row=r, start_column=1, end_row=r, end_column=6)
        ws_gd.row_dimensions[r].height = 34; r += 1

    # ---------------- Method_and_Limits
    n_tr = sum(1 for x in res if x["level"].startswith("Transcript")); n_na = sum(1 for x in res if x["level"] == "Not analysable")
    n_hum = int((df.label_source == "human").sum()); n_all = len(full)
    title(ws_ml, "How this file was made")
    lines = [
        f"1. Source: relevant_strict_videos.csv ({n_all} videos kept as relevant by the screening stage). Only the {N} videos with model score ≥ {thr:.2f} are used here.",
        f"2. Checklist: {NP} points built from the sources in sheet Checklist_Guide. Points that cover similar guideline text were merged to keep the list short.",
        "3. For every video the text fields (title, description, hashtags, tags and transcript when available) were searched for keywords for each point. The matching snippet is stored in sheet Evidence.",
        "4. ✗ is only given when the text positively contradicts a guideline (e.g. covering the face or chest, forceful pinning) and is not negated. Everything else is ? (not verified) and is NOT a failure of the video.",
        "5. N/A is set automatically from the video type guessed from the text (needle, imaging, dental, trauma, neonatal, examination, general) when no keyword for the point was found. Please check these guesses.",
        "Honest limitations",
        "- The videos were NOT watched by the automatic pass; only text was searched. A video can show a point visually without saying it, so ? is common and coverage is a lower bound.",
        f"- Only {n_tr} of the {N} videos have a usable transcript; the others are judged from title, description and hashtags ({n_na} have almost no text).",
        "- Keyword matching can be fooled by negation or context and by non-English text; non-English videos are under-credited.",
        "- Video type, age group, presentation style and source type are auto-guessed from text and can be wrong.",
        f"- Model score: {n_hum} of the {N} videos were labelled by human annotators and the model had seen those labels in training, so their score is in-sample. They are relevant by annotation. Only the {N-n_hum} model-selected videos are chosen by score.",
        "- This is a high-confidence subset of the relevant videos, not all of them. Percentages describe this subset only and must not be read as prevalence on the platform.",
        "- Guidelines describe professional practice. A parent clip can be sound without citing any guideline; the checklist judges what the video shows or says.",
    ]
    for i, t in enumerate(lines):
        c = ws_ml.cell(3 + i, 1, t); c.alignment = WRAP
        if t == "Honest limitations": c.font = Font(bold=True, size=12, color=NAVY)
        ws_ml.row_dimensions[3 + i].height = 18 if t == "Honest limitations" else 34
    ws_ml.column_dimensions["A"].width = 150

    for ws in (ws_fmt, ws_det, ws_sum, ws_gd, ws_ev, ws_ml): ws.sheet_view.zoomScale = 90
    wb.save(a.out)
    print(f"saved {a.out}: {N} videos (score >= {thr}), {NP} points; transcript {n_tr}, not analysable {n_na}; evidence rows {er-4}")

if __name__ == "__main__":
    main()