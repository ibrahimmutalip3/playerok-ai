"""
Playerok AI userbot — точка входа.

Аккаунт-твинк слушает ТОЛЬКО одну группу (PLAYEROK_CHAT_ID).
Все остальные чаты, каналы и личные сообщения игнорируются.

Запуск:
    python main.py                # обычный запуск
    python main.py --list-chats   # показать ID групп аккаунта и выйти
"""

import argparse
import asyncio
import difflib
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

from telethon import events

from ai import AIClient, ChatLine, describe_error
from config import Settings, load_settings
from logger import setup_logger
from style import build_style_profile
from telegram_client import TelegramBot, TelegramBotError


@dataclass
class Stats:
    started_at: float
    received: int = 0       # всего сообщений из разрешённого чата
    evaluated: int = 0      # сообщений, по которым запускался анализ
    replied: int = 0        # сообщений, на которые был отправлен ответ
    skipped: int = 0        # пропущено (не нужно / пауза / повтор / лимиты)
    ai_errors: int = 0      # ошибки AI-модели
    sent: int = 0           # отправлено сообщений
    failed_sends: int = 0   # неудачные отправки


def _fmt_uptime(seconds: float) -> str:
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _author_label(sender) -> str:
    if sender is None:
        return "unknown"
    username = getattr(sender, "username", None)
    if username:
        return f"@{username}"
    return getattr(sender, "first_name", None) or "user"


class Userbot:
    def __init__(self, settings: Settings, log):
        self.s = settings
        self.log = log
        self.bot = TelegramBot(
            api_id=settings.telegram_api_id,
            api_hash=settings.telegram_api_hash,
            session_name=settings.session_name,
            min_reply_delay=settings.min_reply_delay,
            log=log,
        )
        self.ai = AIClient(
            api_key=settings.ai_api_key,
            base_url=settings.ai_base_url,
            model=settings.ai_model,
            timeout=settings.ai_timeout,
        )
        self.stats = Stats(started_at=time.time())
        self.paused = False
        self.style_hint = ""
        self.style_samples = 0
        self._load_style()

        self.recent_replies: deque[str] = deque(maxlen=20)
        self.recent_triggers: deque[str] = deque(maxlen=10)

        self._pending_trigger = None
        self._last_incoming_at = 0.0
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

    # ------------------------------------------------------------ setup

    def _load_style(self) -> None:
        profile = build_style_profile(self.s.messages_file, self.s.style_sample_size)
        self.style_samples = profile.sample_count
        self.style_hint = profile.describe()
        if profile.sample_count == 0:
            self.log.warning("messages.json не найден или пуст — стиль будет нейтральным.")
        else:
            self.log.info(
                "Стиль загружен из messages.json (%d примеров). "
                "Тексты примеров не выводятся и не передаются в промпт.",
                profile.sample_count,
            )

    async def start(self) -> None:
        await self.bot.connect()
        await self.bot.ensure_authorized(use_qr=self.s.auth_method != "phone")
        await self.bot.resolve_playerok_chat(self.s.playerok_chat_id)

        ai_ok, ai_info = await self.ai.health_check()
        if ai_ok:
            self.log.info("AI проверен: %s.", ai_info)
        else:
            self.log.error(
                "AI недоступен при старте: %s. Бот запущен, но ответы будут пропускаться, "
                "пока проблема не будет исправлена.",
                ai_info,
            )

        client = self.bot.client
        # Обработчик группы: фильтр по конкретному чату на уровне Telethon...
        client.add_event_handler(
            self.on_group_message,
            events.NewMessage(chats=self.bot.chat),
        )
        # ...и команды владельца: только личные сообщения от OWNER_ID.
        client.add_event_handler(
            self.on_owner_command,
            events.NewMessage(
                incoming=True,
                from_users=self.s.owner_id,
                func=lambda e: e.is_private,
            ),
        )

        self._tasks.append(asyncio.create_task(self._worker()))
        self._tasks.append(asyncio.create_task(self._terminal_loop()))
        self.log.info(
            "Userbot запущен. Реагирую ТОЛЬКО на чат «%s» (ID %s).",
            getattr(self.bot.chat, "title", "без названия"),
            self.bot.chat_id,
        )

    async def run(self) -> None:
        await self.start()
        disconnect_task = asyncio.create_task(self.bot.client.run_until_disconnected())
        stop_task = asyncio.create_task(self._stop.wait())
        done, pending = await asyncio.wait(
            {disconnect_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await self.shutdown()

    async def shutdown(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._print_stats()
        await self.ai.close()
        await self.bot.disconnect()
        self.log.info("Userbot остановлен.")

    # ------------------------------------------------------------ incoming

    async def on_group_message(self, event) -> None:
        # Жёсткая двойная проверка: сообщение должно быть из разрешённого чата.
        if event.chat_id != self.bot.chat_id:
            return
        # Свои собственные сообщения не обрабатываем как триггер.
        if event.out:
            return
        text = (event.raw_text or "").strip()
        if not text or text.startswith("/"):
            return

        self.stats.received += 1
        try:
            sender = await event.get_sender()
        except Exception:  # noqa: BLE001 — не даём падать из-за отдельного сообщения
            sender = None
        if sender is not None and getattr(sender, "bot", False):
            return

        self.log.info("New message from %s", _author_label(sender))
        self._pending_trigger = event.message
        self._last_incoming_at = time.monotonic()
        self._wake.set()

    async def on_owner_command(self, event) -> None:
        if event.sender_id != self.s.owner_id:
            return
        text = (event.raw_text or "").strip().lower()
        if not text.startswith("/"):
            return
        command = text[1:].split()[0] if len(text) > 1 else ""
        answer = self.handle_command(command)
        try:
            await event.reply(answer)
        except Exception as exc:  # noqa: BLE001
            self.log.error("Не удалось ответить на команду: %s", type(exc).__name__)

    # ------------------------------------------------------------ commands

    def handle_command(self, command: str) -> str:
        title = getattr(self.bot.chat, "title", "без названия")
        if command == "status":
            state = "на паузе" if self.paused else "активен"
            flood = int(self.bot.flood_pause_remaining())
            return (
                f"Статус: {state}.\n"
                f"Чат: «{title}» (ID {self.bot.chat_id}).\n"
                f"Модель: {self.s.ai_model}.\n"
                f"Примеров стиля: {self.style_samples}.\n"
                f"FloodWait: {flood} сек."
            )
        if command == "stop":
            self.paused = True
            self.log.info("Команда /stop: ответы приостановлены.")
            return "Ответы приостановлены. Чтение чата продолжается."
        if command == "start":
            self.paused = False
            self.log.info("Команда /start: ответы включены.")
            return "Ответы включены."
        if command == "reload":
            self._load_style()
            self.log.info("Команда /reload: стиль перечитан из messages.json.")
            return f"Стиль перечитан. Примеров: {self.style_samples}."
        if command == "stats":
            return self._stats_text()
        return "Команды: /status, /stop, /start, /reload, /stats"

    def _stats_text(self) -> str:
        s = self.stats
        return (
            f"Время работы: {_fmt_uptime(time.time() - s.started_at)}\n"
            f"Получено из чата: {s.received}\n"
            f"Проанализировано: {s.evaluated}\n"
            f"Отвечено: {s.replied}\n"
            f"Пропущено: {s.skipped}\n"
            f"Ошибок AI: {s.ai_errors}\n"
            f"Отправлено сообщений: {s.sent}\n"
            f"Неудачных отправок: {s.failed_sends}"
        )

    def _print_stats(self) -> None:
        for line in self._stats_text().splitlines():
            self.log.info("%s", line)

    async def _terminal_loop(self) -> None:
        """Локальные команды из терминала: status, stop, start, reload, stats, quit."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def reader() -> None:
            for line in sys.stdin:
                loop.call_soon_threadsafe(queue.put_nowait, line)
            loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=reader, daemon=True).start()

        while not self._stop.is_set():
            line = await queue.get()
            if line is None:  # stdin закрыт — локальные команды недоступны
                return
            command = line.strip().lower().lstrip("/")
            if not command:
                continue
            if command in ("quit", "exit"):
                self.log.info("Завершаю работу по команде из терминала.")
                self._stop.set()
                return
            print(self.handle_command(command))

    # ------------------------------------------------------------ pipeline

    async def _worker(self) -> None:
        """Обрабатывает сообщения по одному, собирая пачки (batching)."""
        while not self._stop.is_set():
            await self._wake.wait()
            # Ждём, пока поток сообщений не утихнет (BATCH_WINDOW).
            while not self._stop.is_set():
                idle = time.monotonic() - self._last_incoming_at
                if idle >= self.s.batch_window:
                    break
                await asyncio.sleep(self.s.batch_window - idle)

            self._wake.clear()
            trigger = self._pending_trigger
            self._pending_trigger = None
            if trigger is None:
                continue
            try:
                await self._handle_trigger(trigger)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self.log.error("Внутренняя ошибка обработки: %s", type(exc).__name__)

    async def _build_context(self, trigger) -> tuple[list[ChatLine], ChatLine | None]:
        raw = await self.bot.fetch_context(self.s.context_size)
        own_ids = {m.id for m in raw if m.out}
        lines: list[ChatLine] = []
        trigger_line: ChatLine | None = None

        for m in raw:
            text = (m.raw_text or "").strip()
            if not text:
                continue
            sender = m.sender
            if sender is not None and getattr(sender, "bot", False):
                continue
            is_own = bool(m.out)
            line = ChatLine(
                author="ты (твинк)" if is_own else _author_label(sender),
                text=text[:600],
                is_own=is_own,
                is_reply_to_own=bool(m.reply_to_msg_id and m.reply_to_msg_id in own_ids),
                mentions_own=bool(getattr(m, "mentioned", False)) and not is_own,
            )
            lines.append(line)
            if m.id == trigger.id:
                trigger_line = line

        return lines, trigger_line

    async def _handle_trigger(self, trigger) -> None:
        if self.paused:
            self.stats.skipped += 1
            self.log.info("Бот на паузе — сообщение пропущено.")
            return
        if self.bot.flood_pause_remaining() > 0:
            self.stats.skipped += 1
            self.log.info("Действует ожидание FloodWait — сообщение пропущено.")
            return

        trigger_text = (trigger.raw_text or "").strip()
        if trigger_text in self.recent_triggers:
            self.stats.skipped += 1
            self.log.info("Повторяющееся сообщение — пропущено.")
            return
        self.recent_triggers.append(trigger_text)
        self.stats.evaluated += 1

        lines, trigger_line = await self._build_context(trigger)
        if not lines or trigger_line is None:
            self.stats.skipped += 1
            self.log.info("Нет подходящего контекста — пропущено.")
            return

        direct = trigger_line.is_reply_to_own or trigger_line.mentions_own

        self.log.info("Checking whether to reply…")
        should, reason = await self.ai.should_reply(lines, self.style_hint)
        if reason.startswith("ai_error"):
            self.stats.ai_errors += 1
            self.log.warning(
                "AI недоступен: %s — сообщение пропущено.", describe_error(self.ai.last_error)
            )
            return
        if not should:
            self.stats.skipped += 1
            self.log.info("Ответ не нужен — ничего не делаю.")
            return

        # REPLY_PROBABILITY — дополнительный механизм. Прямые обращения его не блокируют.
        if not direct and random.random() > self.s.reply_probability:
            self.stats.skipped += 1
            self.log.info("Решил промолчать (вероятностный фильтр).")
            return

        candidates = await self.ai.generate_reply(lines, self.style_hint, list(self.recent_replies))
        if candidates is not None and any(self._too_similar(c) for c in candidates):
            candidates = await self.ai.generate_reply(lines, self.style_hint, list(self.recent_replies))
        if candidates is None:
            self.stats.ai_errors += 1
            self.log.warning(
                "AI не вернул ответ: %s — сообщение пропущено.", describe_error(self.ai.last_error)
            )
            return
        if any(self._too_similar(c) for c in candidates):
            self.stats.skipped += 1
            self.log.info("Ответ слишком похож на недавний — пропущено.")
            return

        self.log.info("AI generated response")

        total_len = sum(len(c) for c in candidates)
        await asyncio.sleep(self._natural_delay(total_len))

        if self.paused or self.bot.flood_pause_remaining() > 0:
            self.stats.skipped += 1
            self.log.info("Ответ отменён (пауза или FloodWait).")
            return

        sent_any = False
        for idx, text in enumerate(candidates):
            await self.bot.send_typing(self._typing_duration(text))
            reply_to = trigger.id if (direct and idx == 0) else None
            ok = await self.bot.send_text(text, reply_to=reply_to)
            if not ok:
                self.stats.failed_sends += 1
                break
            sent_any = True
            self.stats.sent += 1
            self.recent_replies.append(text)
            self.log.info("Message sent")

        if sent_any:
            self.stats.replied += 1

    # ------------------------------------------------------------ helpers

    def _too_similar(self, text: str) -> bool:
        norm = text.lower().strip()
        for prev in self.recent_replies:
            prev_norm = prev.lower().strip()
            if norm == prev_norm:
                return True
            if difflib.SequenceMatcher(None, norm, prev_norm).ratio() >= 0.85:
                return True
        return False

    @staticmethod
    def _natural_delay(length: int) -> float:
        """Задержка перед ответом: случайная, чуть больше для длинных ответов (3–10 сек)."""
        base = random.uniform(3.0, 6.0)
        bonus = min(length / 150.0, 3.0)
        return round(min(base + bonus + random.uniform(0.0, 1.0), 10.0), 2)

    @staticmethod
    def _typing_duration(text: str) -> float:
        return round(min(1.0 + len(text) / 25.0 + random.uniform(0.0, 1.0), 6.0), 2)


# ---------------------------------------------------------------- entry points


async def run_bot(settings: Settings, log) -> None:
    bot = Userbot(settings, log)
    await bot.run()


async def list_chats(settings: Settings, log) -> None:
    bot = TelegramBot(
        api_id=settings.telegram_api_id,
        api_hash=settings.telegram_api_hash,
        session_name=settings.session_name,
        min_reply_delay=settings.min_reply_delay,
        log=log,
    )
    await bot.connect()
    try:
        await bot.ensure_authorized(use_qr=settings.auth_method != "phone")
        chats = await bot.list_chats()
        print("\nГруппы и каналы аккаунта (ID нужно вставить в PLAYEROK_CHAT_ID):")
        for name, chat_id in chats:
            print(f"  {chat_id:>20}   {name}")
        print()
    finally:
        await bot.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Playerok AI userbot")
    parser.add_argument(
        "--list-chats",
        action="store_true",
        help="Показать группы аккаунта с их ID и выйти (чтобы узнать PLAYEROK_CHAT_ID)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        settings = load_settings(strict=not args.list_chats)
    except RuntimeError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        return 1

    log = setup_logger(settings.log_file)
    try:
        if args.list_chats:
            asyncio.run(list_chats(settings, log))
        else:
            asyncio.run(run_bot(settings, log))
    except KeyboardInterrupt:
        log.info("Остановлено пользователем (Ctrl+C).")
    except TelegramBotError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
