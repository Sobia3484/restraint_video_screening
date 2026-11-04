"""Transparent keyword rules built from docs/screening_criteria.md (not tuned on labels).

Three jobs:
  1. a rule label (relevant / potentially_relevant / irrelevant) used as a baseline,
  2. two gates for the hybrid models:  gate_topic  = topic term present and no false-match signal,
                                       gate_strict = gate_topic AND a child/infant term present,
  3. signal flags used as numeric features.
Text is matched after folding Latin diacritics. English first, with a few Spanish/Portuguese terms.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text import fold

W = r"(?<![^\W_])"  # not preceded by a letter or digit


def _rx(terms) -> re.Pattern:
    return re.compile("|".join(f"{W}(?:{t})" for t in terms), re.IGNORECASE | re.UNICODE)


# ---- condition A: infant / young child
CHILD = _rx([
    r"infant\w*", r"bab(?:y|ies)\w*", r"newborn\w*", r"neonat\w*", r"nicu\b", r"pre-?emie\w*", r"premie\w*", r"child\w*",
    r"kids?\b", r"toddler\w*", r"p(?:a)?ediatr\w*", r"peds\b", r"paeds\b", r"preschool\w*", r"little\s+ones?\b",
    r"\d+[\s-]*(?:year|yr|month|mo|week|wk)s?[\s-]*old", r"bebe\w*", r"nin[oa]s?\b", r"lactante\w*", r"crianca\w*", r"enfants?\b",
])

# ---- condition B: holding / restraining / immobilizing (strong terms)
TOPIC = _rx([
    r"restrain\w*", r"immobili[sz]\w*", r"papoose\w*", r"swaddl\w*", r"mummy\s+(?:wrap|restraint)\w*", r"pedi[\s-]*wrap\w*",
    r"clinical\s+hold\w*", r"therapeutic\s+hold\w*", r"comfort\s+hold\w*", r"(?:safe|proper|gentle|secure|correct)\s+hold\w*",
    r"hold(?:ing)?\s+(?:a\s+|the\s+|your\s+|their\s+|him\s+|her\s+)?(?:\w+\s+){0,2}(?:still|down|steady|securely)",
    r"hold(?:ing)?\s+(?:a\s+|the\s+|your\s+)?(?:baby|child|infant|toddler|kid|newborn)\w*\s+(?:for|during)",
    r"bear\s+hug", r"lap\s+hold", r"facilitated\s+tucking", r"containment",
    r"protective\s+stabili[sz]\w*", r"stabili[sz]ation", r"spinal\s+(?:motion|immobili\w*|precaution\w*)", r"smr\b",
    r"c[\s-]*collar\w*", r"cervical\s+(?:collar|immobili\w*)", r"backboard\w*", r"spine\s*board\w*", r"scoop\s+stretcher",
    r"vacuum\s+(?:splint|immobili\w*|mattress)", r"splint\w*", r"stabili[sz]e\s+(?:the\s+)?(?:head|neck|spine|limb|arm|leg)",
    r"inmoviliz\w*", r"estabilizacion", r"sujec\w*", r"imobiliz\w*", r"contencao",
])
# ---- care settings where holding happens (weak terms)
SETTING = _rx([
    r"needle\w*", r"inject\w*", r"vaccin\w*", r"immuni[sz]\w*", r"shots?\b", r"blood\s+(?:draw|sample|test)\w*", r"venipuncture", r"heel\s*(?:stick|prick)\w*",
    r"\biv\b", r"cannula\w*", r"phlebotom\w*", r"mri\b", r"ct\s+scan\w*", r"x-?ray\w*", r"imaging", r"radiolog\w*", r"dentist\w*", r"dental", r"fractur\w*",
    r"\bcast\b", r"sutur\w*", r"stitch\w*", r"lacerat\w*", r"wound\w*", r"ear\s+exam\w*", r"otoscop\w*", r"throat\s+exam\w*", r"trauma\w*",
    r"hospital\w*", r"clinic\w*", r"procedur\w*", r"sedat\w*", r"anesthe\w*", r"anaesthe\w*", r"lumbar\s+puncture",
])
INSTRUCT = _rx([
    r"how\s+to", r"step[\s-]*by[\s-]*step", r"tutorial\w*", r"demonstrat\w*", r"technique\w*", r"procedure\w*", r"guideline\w*", r"guide\b",
    r"what\s+to\s+do", r"learn\w*", r"training", r"explain\w*", r"best\s+practice\w*", r"safely", r"safe\b", r"tips?\b", r"skills?\b",
])
PROMO = _rx([
    r"enrol\w*", r"regist(?:er|ration)\w*", r"sign\s*up", r"book\s+(?:now|your|a)", r"join\s+us", r"workshop", r"promo\s*code", r"discount",
    r"coupon", r"link\s+in\s+bio", r"dm\s+(?:us|me)", r"call\s+us", r"visit\s+our", r"order\s+now", r"shop\s+now", r"for\s+sale",
])
ADULT = _rx([r"adults?\b", r"adulto\w*", r"elderly", r"geriatric\w*"])

# ---- false matches (criteria section 5)
EXCLUDE = {
    "restraint_collapse": _rx([r"restraint\s+collapse", r"after[\s-]*school\s+restraint"]),
    "car_seat": _rx([r"car\s*seat\w*", r"carseat\w*", r"(?:child|infant|baby)\s+(?:safety\s+)?seat\w*", r"booster\s+seat\w*",
                     r"child\s+restraint\s+system\w*", r"seat\s*belt\w*", r"strollers?\b", r"isofix"]),
    "music_media": _rx([r"lyrics?\b", r"official\s+(?:music\s+)?video", r"music\s+video", r"remix\b", r"\bfeat\.?\s", r"\bft\.\s", r"\balbum\b",
                       r"\bsong\b", r"karaoke", r"gameplay", r"minecraft", r"roblox", r"let'?s\s+play", r"\basmr\b", r"\bmv\b"]),
    "non_medical_restraint": _rx([r"bondage", r"bdsm", r"handcuff\w*", r"straitjacket", r"strait\s*jacket", r"jiu[\s-]*jitsu", r"\bbjj\b",
                                  r"martial\s+arts?", r"self[\s-]*defen[cs]e", r"police", r"arrested", r"taser", r"wrestl\w*"]),
    "pet_animal": _rx([r"dogs?\b", r"pupp(?:y|ies)\b", r"cats?\b(?!\s*scan)", r"kittens?\b", r"pets?\b(?!\s*(?:scan|ct|mri|imaging))", r"canine\w*", r"horses?\b", r"vet\b", r"veterinar\w*"]),
}
SLEEP = _rx([r"sleep\w*", r"bedtime", r"naps?\b", r"nap\s*time", r"sids\b", r"lullab\w*", r"soothe\w*", r"fussy"])


@dataclass
class RuleResult:
    label: str
    reason: str
    child: list = field(default_factory=list)
    topic: list = field(default_factory=list)
    excluded: str = ""
    gate_topic: bool = False
    gate_strict: bool = False
    blocked: bool = False      # false-match signal or sleep-only context


def _hits(rx: re.Pattern, text: str) -> list:
    return sorted({m.group(0).strip() for m in rx.finditer(text)})[:8]


def exclusion_category(text: str) -> str:
    for name, rx in EXCLUDE.items():
        if rx.search(text):
            return name
    return ""


def apply_rules(title: str, description: str, hashtags: str, tags: str) -> RuleResult:
    head = fold(f"{title} {hashtags.replace('|', ' ')} {tags.replace('|', ' ')}")
    allt = f"{head} {fold(description or '')}"
    child, topic = _hits(CHILD, allt), _hits(TOPIC, allt)
    excl = exclusion_category(allt)
    setting = bool(SETTING.search(allt))
    sleep_only = bool(SLEEP.search(allt)) and not setting
    blocked = bool(excl) or sleep_only
    gate_topic = bool(topic) and not blocked
    gate_strict = gate_topic and bool(child)
    base = dict(child=child, topic=topic, excluded=excl, gate_topic=gate_topic, gate_strict=gate_strict, blocked=blocked)
    if excl:
        return RuleResult("irrelevant", f"false_match:{excl}", **base)
    if not topic:
        if child and setting:
            return RuleResult("potentially_relevant", "child_and_setting_only", **base)
        return RuleResult("irrelevant", "no_topic_term", **base)
    if sleep_only:
        return RuleResult("irrelevant", "sleep_context", **base)
    if child:
        if PROMO.search(allt) and not INSTRUCT.search(allt):
            return RuleResult("potentially_relevant", "promo_without_instruction", **base)
        return RuleResult("relevant", "child_and_topic", **base)
    if ADULT.search(allt):
        return RuleResult("irrelevant", "adult_only", **base)
    return RuleResult("potentially_relevant", "topic_without_age", **base)