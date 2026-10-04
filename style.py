"""
Анализ стиля общения по messages.json.

messages.json используется ТОЛЬКО как STYLE REFERENCE:
из него извлекаются статистические признаки стиля (длина, пунктуация,
эмодзи, скобки, регистр, сленг и т.п.). Сами тексты сообщений в промпт
НЕ передаются и AI не получает инструкции копировать их.
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field, asdict

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F000-\U0001F2FF\u2600-\u26FF]"
)
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё']+")

LATIN_RE = re.compile(r"[A-Za-z]")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

# Сленг и слова-маркеры, которые считаем «характерными», если они встречаются.
SLANG_CANDIDATES = [
    "лол", "кек", "ахах", "жиза", "норм", "ок", "окей", "бля", "блин",
    "короче", "типа", "щас", "чё", "че", "ну", "го", "пон", "кста",
    "lol", "lmao", "bro", "ok", "kek", "ngl", "tbh", "yeah", "nah",
]


@dataclass
class StyleProfile:
    sample_count: int = 0
    avg_length: float = 0.0
    median_length: float = 0.0
    short_ratio: float = 0.0       # доля сообщений до 30 символов
    long_ratio: float = 0.0        # доля сообщений больше 150 символов
    lowercase_start_ratio: float = 0.0
    no_final_punct_ratio: float = 0.0
    emoji_per_message: float = 0.0
    emoji_usage_ratio: float = 0.0
    exclamation_ratio: float = 0.0
    question_ratio: float = 0.0
    ellipsis_ratio: float = 0.0
    parentheses_ratio: float = 0.0
    dash_ratio: float = 0.0
    multi_punct_ratio: float = 0.0
    all_caps_word_ratio: float = 0.0
    cyrillic_ratio: float = 0.0
    latin_ratio: float = 0.0
    top_emojis: list[str] = field(default_factory=list)
    top_slang: list[str] = field(default_factory=list)
    top_words: list[str] = field(default_factory=list)

    def describe(self) -> str:
        """Текстовое описание стиля для системного промпта (без текстов сообщений)."""
        lines = []
        if self.sample_count == 0:
            return "Стиль: нет данных, пиши коротко, неформально и естественно."

        if self.avg_length < 40:
            lines.append("обычно пишешь очень коротко (несколько слов)")
        elif self.avg_length < 100:
            lines.append("пишешь сообщения средней длины (1–2 предложения)")
        else:
            lines.append("иногда пишешь длинно, но в целом обходишься короткими фразами")

        if self.lowercase_start_ratio > 0.6:
            lines.append("обычно начинаешь сообщения со строчной буквы")
        if self.no_final_punct_ratio > 0.6:
            lines.append("чаще не ставишь точку в конце")
        if self.emoji_usage_ratio > 0.3:
            lines.append(f"часто используешь эмодзи (в среднем {self.emoji_per_message:.1f} на сообщение)")
        elif self.emoji_usage_ratio < 0.05:
            lines.append("почти не используешь эмодзи")
        if self.parentheses_ratio > 0.1:
            lines.append("иногда используешь скобки")
        if self.multi_punct_ratio > 0.05:
            lines.append("иногда дублируешь знаки препинания (!!, ??, ...)")
        if self.ellipsis_ratio > 0.05:
            lines.append("любишь многоточие")
        if self.all_caps_word_ratio > 0.05:
            lines.append("иногда пишешь слова капсом для акцента")
        if self.cyrillic_ratio > 0.8:
            lines.append("пишешь в основном по-русски")
        elif self.latin_ratio > 0.3:
            lines.append("часто смешиваешь русский и английский")
        if self.top_slang:
            lines.append("характерный сленг: " + ", ".join(self.top_slang[:6]))
        if self.top_emojis:
            lines.append("любимые эмодзи: " + " ".join(self.top_emojis[:5]))
        return "Стиль автора: " + "; ".join(lines) + "."


def _load_messages(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    texts: list[str] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                texts.append(item.strip())
    return texts


def build_style_profile(messages_path: str, sample_size: int = 40) -> StyleProfile:
    texts = _load_messages(messages_path)
    if not texts:
        return StyleProfile()

    # Берём ограниченную выборку, чтобы статистика была стабильной и быстрой.
    sample = texts[-sample_size:] if len(texts) > sample_size else texts
    n = len(sample)
    lengths = sorted(len(t) for t in sample)

    all_words: list[str] = []
    all_emojis: list[str] = []
    slang_counter: Counter = Counter()
    lower_start = no_final = emoji_msgs = excl = quest = ell = paren = dash = multi = 0
    caps_words = total_words = 0
    cyr_chars = lat_chars = 0

    for t in sample:
        words = WORD_RE.findall(t)
        all_words.extend(w.lower() for w in words)
        total_words += len(words)
        caps_words += sum(1 for w in words if len(w) > 2 and w.isupper())

        emojis = EMOJI_RE.findall(t)
        all_emojis.extend(emojis)
        if emojis:
            emoji_msgs += 1

        if t and t[0].islower():
            lower_start += 1
        if t and t[-1] not in ".!?…)":
            no_final += 1
        if "!" in t:
            excl += 1
        if "?" in t:
            quest += 1
        if "..." in t or "…" in t:
            ell += 1
        if "(" in t or ")" in t:
            paren += 1
        if " - " in t or " — " in t or "–" in t:
            dash += 1
        if re.search(r"([!?.])\1", t):
            multi += 1

        cyr_chars += len(CYRILLIC_RE.findall(t))
        lat_chars += len(LATIN_RE.findall(t))

        lowered = t.lower()
        for s in SLANG_CANDIDATES:
            if re.search(rf"(?<!\w){re.escape(s)}(?!\w)", lowered):
                slang_counter[s] += 1

    word_counter = Counter(w for w in all_words if len(w) > 2)
    emoji_counter = Counter(all_emojis)
    total_chars = cyr_chars + lat_chars or 1

    return StyleProfile(
        sample_count=n,
        avg_length=sum(lengths) / n,
        median_length=float(lengths[n // 2]),
        short_ratio=sum(1 for x in lengths if x < 30) / n,
        long_ratio=sum(1 for x in lengths if x > 150) / n,
        lowercase_start_ratio=lower_start / n,
        no_final_punct_ratio=no_final / n,
        emoji_per_message=len(all_emojis) / n,
        emoji_usage_ratio=emoji_msgs / n,
        exclamation_ratio=excl / n,
        question_ratio=quest / n,
        ellipsis_ratio=ell / n,
        parentheses_ratio=paren / n,
        dash_ratio=dash / n,
        multi_punct_ratio=multi / n,
        all_caps_word_ratio=(caps_words / total_words) if total_words else 0.0,
        cyrillic_ratio=cyr_chars / total_chars,
        latin_ratio=lat_chars / total_chars,
        top_emojis=[e for e, _ in emoji_counter.most_common(5)],
        top_slang=[s for s, _ in slang_counter.most_common(6)],
        top_words=[w for w, _ in word_counter.most_common(10)],
    )


def profile_to_dict(profile: StyleProfile) -> dict:
    return asdict(profile)
