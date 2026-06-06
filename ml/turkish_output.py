"""Isaret gloss -> ekranda gosterilecek ham ve duzeltilmis Turkce metin."""

# Tek kelimelik isaretlerde dogal kisa cumle sablonlari
_SINGLE_WORD_HINTS: dict[str, str] = {
    "merhaba": "Merhaba.",
    "tesekkur": "Teşekkür ederim.",
    "evet": "Evet.",
    "hayir": "Hayır.",
    "lutfen": "Lütfen.",
    "su": "Su.",
    "anne": "Anne.",
    "baba": "Baba.",
    "arkadas": "Arkadaş.",
    "aile": "Aile.",
}


def gloss_to_display(gloss: str) -> str:
    g = (gloss or "").strip().replace("_", " ")
    if not g:
        return ""
    return g.upper()


def gloss_to_sentence(gloss: str, low_confidence: bool = False) -> str:
    g = (gloss or "").strip().replace("_", " ")
    if not g:
        return "İşaret net algılanamadı; videoyu tekrar deneyin."
    if low_confidence:
        word = g.split()[0].lower()
        hint = _SINGLE_WORD_HINTS.get(word)
        base = hint or (" ".join(w[:1].upper() + w[1:] for w in g.split()) + ".")
        return f"Tahmin zayıf; en olası: {base} (farklı açıdan tekrar deneyin.)"

    key = g.replace(" ", "_").lower()
    if key in _SINGLE_WORD_HINTS:
        return _SINGLE_WORD_HINTS[key]
    words = g.split()
    out = " ".join(w[:1].upper() + w[1:] if w else "" for w in words)
    if out and out[-1] not in ".!?":
        out += "."
    return out
