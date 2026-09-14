from __future__ import annotations

import re

_STT_FIXES = (
    (re.compile(r"\bmhp\b", re.I), "Mbps"),
    (re.compile(r"\bmb/s\b", re.I), "Mbps"),
    (re.compile(r"\bmbps\b", re.I), "Mbps"),
    (re.compile(r"\bmégas?\b", re.I), "Mbps"),
    (re.compile(r"\bmegas?\b", re.I), "Mbps"),
    (re.compile(r"\bcent\s+méga", re.I), "100 Mbps"),
)

_NEW_TOPIC = re.compile(
    r"offre|forfait|tarif|prix|fibre|mbps|mhp|méga|mega|résili|factur|déménag|"
    r"paiement|routeur|upgrade|100\s|50\s|300\s",
    re.I,
)

_FOLLOWUP = re.compile(
    r"^(oui|non|ok|okay|d['’ ]?accord|c['’ ]est fait|fait|ensuite|et après|"
    r"toujours pas|ça (ne )?marche pas|merci|bon|allo)[\s.!?]*$",
    re.I,
)


def normalize_query(text: str) -> str:
    out = " ".join((text or "").split())
    for pat, repl in _STT_FIXES:
        out = pat.sub(repl, out)
    return out


def is_followup(text: str) -> bool:
    stripped = (text or "").strip()
    return bool(_FOLLOWUP.match(stripped)) and not _NEW_TOPIC.search(stripped)


_FILLER = re.compile(r"^(euh+|hum+|hmm+|bah+|allo|allô)[\s.!?]*$", re.I)


def transcript_needs_clarification(text: str, extra: dict | None = None) -> bool:
    if extra and int(extra.get("stt_unclear") or 0) == 1:
        return True
    stripped = (text or "").strip()
    if not stripped:
        return True
    if is_followup(stripped):
        return False
    if _FILLER.match(stripped):
        return True
    return False


def expand_retrieval_query(text: str) -> str:
    """Keep the user wording; add catalog synonyms only for retrieval."""
    query = normalize_query(text)
    lowered = query.lower()
    if re.search(r"\boffres?\b", lowered) and not re.search(r"forfait|fibre|tarif|prix", lowered):
        return f"{query} forfaits fibre tarifs"
    return query


def lexical_needles(text: str) -> list[str]:
    t = normalize_query(text).lower()
    needles: list[str] = []
    if re.search(r"\boffres?\b", t):
        needles.append("Forfaits Fibre")
    if "forfait" in t or "tarif" in t or "prix" in t or re.search(r"\boffres?\b", t):
        needles.append("Forfait")
    if "fibre" in t:
        needles.append("Fibre")
    if "mbps" in t or "méga" in t or "mega" in t:
        needles.append("Mbps")
    if re.search(r"\b100\b", t):
        needles.extend(["100 Mbps", "Fibre 100"])
    if re.search(r"\b50\b", t) and any(x in t for x in ("forfait", "fibre", "mbps", "méga")):
        needles.append("Fibre 50")
    if re.search(r"\b300\b", t):
        needles.append("Fibre 300")
    seen: set[str] = set()
    unique: list[str] = []
    for needle in needles:
        key = needle.lower()
        if key not in seen:
            seen.add(key)
            unique.append(needle)
    return unique
