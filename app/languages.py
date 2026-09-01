LANGUAGE_NAMES = {
    "ar": "Arabic",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
}


def language_instruction(language: str | None) -> str:
    if not language:
        return (
            "Use exactly the same language as the latest utterance. Preserve unavoidable proper "
            "names, acronyms, and technical terms intact in their original form."
        )
    code = language.casefold().split("-", 1)[0]
    name = LANGUAGE_NAMES.get(code, f"the language identified by code '{language}'")
    return (
        f"The required output language is {name} (code {language}). "
        "Use that language for the entire card, regardless of languages in older context "
        "or previous cards. Keep only unavoidable proper names, acronyms, and technical terms "
        "in their original form, preserve them intact, and integrate them naturally into the "
        "surrounding sentence. Do not append an unrelated translation or second-language clause."
    )
