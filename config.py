"""
Конфигурация проекта.

Все секреты читаются ТОЛЬКО из переменных окружения.
Никаких значений секретов здесь быть не должно.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, required: bool = True, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        if required:
            raise RuntimeError(
                f"Не задана переменная окружения {name}. "
                f"Заполни её в файле .env (см. README.md)."
            )
        value = default
    return value


def _get_int(name: str, required: bool = True, default: int | None = None) -> int | None:
    raw = _get_env(name, required=required, default=None if default is None else str(default))
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Переменная {name} должна быть целым числом.") from exc


def _get_float(name: str, default: float) -> float:
    raw = _get_env(name, required=False, default=str(default))
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Переменная {name} должна быть числом.") from exc


@dataclass(frozen=True)
class Settings:
    # Telegram
    telegram_api_id: int
    telegram_api_hash: str
    playerok_chat_id: int
    owner_id: int
    session_name: str
    auth_method: str  # "qr" (по умолчанию) или "phone"

    # AI
    ai_api_key: str
    ai_base_url: str
    ai_model: str
    ai_timeout: float

    # Поведение
    reply_probability: float
    min_reply_delay: float
    max_reply_delay: float
    batch_window: float
    context_size: int
    style_sample_size: int

    # Пути
    messages_file: str
    log_file: str


def load_settings(strict: bool = True) -> Settings:
    """
    strict=True  — полный режим работы бота, все обязательные переменные обязательны.
    strict=False — режим --list-chats: AI-ключ и PLAYEROK_CHAT_ID не нужны.
    """
    auth_method = (_get_env("AUTH_METHOD", required=False, default="qr") or "qr").strip().lower()
    if auth_method not in ("qr", "phone"):
        raise RuntimeError("AUTH_METHOD должен быть 'qr' или 'phone'.")

    return Settings(
        telegram_api_id=_get_int("TELEGRAM_API_ID"),
        telegram_api_hash=_get_env("TELEGRAM_API_HASH"),
        playerok_chat_id=_get_int("PLAYEROK_CHAT_ID", required=strict, default=0),
        owner_id=_get_int("OWNER_ID", required=strict, default=0),
        session_name=_get_env("TELEGRAM_SESSION_NAME", required=False, default="twink_session"),
        auth_method=auth_method,

        ai_api_key=_get_env("AI_API_KEY", required=strict, default="") or "",
        ai_base_url=_get_env("AI_BASE_URL", required=False, default="https://api.openai.com/v1"),
        ai_model=_get_env("AI_MODEL", required=False, default="gpt-4o-mini"),
        ai_timeout=_get_float("AI_TIMEOUT", default=30.0),

        # Вероятность ответа — дополнительный механизм, а не основной.
        reply_probability=_get_float("REPLY_PROBABILITY", default=0.35),
        # MIN_REPLY_DELAY = 5 — минимальный интервал между отправками (cooldown).
        min_reply_delay=_get_float("MIN_REPLY_DELAY", default=5.0),
        max_reply_delay=_get_float("MAX_REPLY_DELAY", default=7.0),
        # Сбор пачки сообщений, пришедших почти одновременно.
        batch_window=_get_float("BATCH_WINDOW", default=3.0),
        # Сколько последних сообщений группы передавать в AI (20–40).
        context_size=int(_get_float("CONTEXT_SIZE", default=30)),
        # Сколько примеров из messages.json использовать для анализа стиля.
        style_sample_size=int(_get_float("STYLE_SAMPLE_SIZE", default=40)),

        messages_file=_get_env("MESSAGES_FILE", required=False, default="messages.json"),
        log_file=_get_env("LOG_FILE", required=False, default="userbot.log"),
    )
