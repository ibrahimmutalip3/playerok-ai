"""
Логирование с маскировкой секретов.

В логи никогда не попадают: API hash, API keys, содержимое session,
пароли и коды авторизации. Все записи проходят через фильтр маскировки.
"""

import logging
import re
import sys

_SECRET_PATTERNS: list[re.Pattern] = []


def register_secret(value: str | None) -> None:
    """Регистрирует секрет, который нужно замаскировать в логах."""
    if value and len(str(value)) >= 4:
        _SECRET_PATTERNS.append(re.compile(re.escape(str(value))))


class SecretMaskingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for pattern in _SECRET_PATTERNS:
            message = pattern.sub("***", message)
        # Дополнительно маскируем типичный формат API-ключей.
        message = re.sub(r"(sk-[A-Za-z0-9_\-]{8,})", "***", message)
        record.msg = message
        record.args = ()
        return True


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("playerok")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    mask = SecretMaskingFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(mask)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(mask)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
