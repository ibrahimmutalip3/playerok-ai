"""
Работа с Telegram через Telethon.

Здесь находится всё, что касается аккаунта-твинка:
- авторизация (QR-код или номер телефона) и сохранение .session;
- проверка, что PLAYEROK_CHAT_ID существует и это именно нужная группа;
- получение контекста последних сообщений;
- отправка сообщений с cooldown и корректной обработкой FloodWait.
"""

import asyncio
import getpass
import time

from telethon import TelegramClient, errors
from telethon.tl.types import Channel, Chat, User
from telethon.utils import get_peer_id

from logger import register_secret


class TelegramBotError(Exception):
    pass


class TelegramBot:
    def __init__(
        self,
        api_id: int,
        api_hash: str,
        session_name: str,
        min_reply_delay: float,
        log,
    ):
        register_secret(api_hash)
        register_secret(str(api_id))
        self._log = log
        self._min_reply_delay = min_reply_delay
        self._last_sent_at = 0.0
        self._flood_until = 0.0  # monotonic time, до которого нельзя отправлять
        self._session_path = f"{session_name}.session"

        # flood_sleep_threshold=0: мы сами обрабатываем FloodWait, без скрытых ожиданий.
        self.client = TelegramClient(
            session_name,
            api_id,
            api_hash,
            flood_sleep_threshold=0,
        )
        self.me = None
        self.chat = None
        self.chat_id: int | None = None

    # ------------------------------------------------------------------ connect

    async def connect(self) -> None:
        await self.client.connect()

    async def disconnect(self) -> None:
        await self.client.disconnect()

    async def ensure_authorized(self, use_qr: bool = True) -> None:
        """Если сессии нет или она недействительна — проводит авторизацию."""
        if await self.client.is_user_authorized():
            self.me = await self.client.get_me()
            self._log.info("Сессия найдена, аккаунт авторизован.")
            return

        self._log.info("Сессия не найдена. Запускаю авторизацию аккаунта-твинка.")
        if use_qr:
            try:
                await self._login_qr()
            except TelegramBotError as exc:
                self._log.error("QR-авторизация не удалась: %s. Пробую по номеру телефона.", exc)
                await self._login_phone()
        else:
            await self._login_phone()

        self.me = await self.client.get_me()
        self._log.info("Авторизация успешна. Сессия сохранена в файл %s (содержимое не выводится).",
                       self._session_path)

    async def _login_qr(self, attempts: int = 5) -> None:
        """Вход по QR-коду. QR-код выводится только в терминал, не в лог."""
        try:
            import qrcode  # type: ignore
        except ImportError as exc:
            raise TelegramBotError("не установлен пакет qrcode") from exc

        for attempt in range(1, attempts + 1):
            qr_login = await self.client.qr_login()
            print("\n=== Отсканируйте QR-код в Telegram: Настройки -> Устройства -> Подключить устройство ===")
            qr = qrcode.QRCode(border=1)
            qr.add_data(qr_login.url)
            qr.print_ascii(invert=True)
            print("Ожидаю сканирования (QR обновляется примерно каждые 30 секунд)...\n")

            try:
                await qr_login.wait(timeout=30)
                return
            except asyncio.TimeoutError:
                self._log.info("QR-код истёк, генерирую новый (попытка %d из %d).", attempt, attempts)
                continue
            except errors.SessionPasswordNeededError:
                await self._submit_2fa()
                return
            except errors.RPCError as exc:
                raise TelegramBotError(type(exc).__name__) from exc

        raise TelegramBotError("QR-код не был отсканирован вовремя")

    async def _submit_2fa(self) -> None:
        """Запрашивает облачный пароль 2FA. Пароль нигде не сохраняется и не логируется."""
        password = getpass.getpass("Введите облачный пароль Telegram (2FA), ввод скрыт: ")
        await self.client.sign_in(password=password)

    async def _login_phone(self) -> None:
        phone = input("Введите номер телефона аккаунта-твинка в формате +71234567890: ").strip()
        await self.client.send_code_request(phone)
        code = input("Введите код из Telegram: ").strip()
        try:
            await self.client.sign_in(phone=phone, code=code)
        except errors.SessionPasswordNeededError:
            await self._submit_2fa()
        except errors.RPCError as exc:
            raise TelegramBotError(f"вход по телефону не удался ({type(exc).__name__})") from exc

    # ------------------------------------------------------------------ chat

    async def resolve_playerok_chat(self, expected_chat_id: int) -> None:
        """Находит группу по ID и проверяет, что это именно разрешённый чат."""
        try:
            entity = await self.client.get_entity(expected_chat_id)
        except (ValueError, errors.RPCError) as exc:
            raise TelegramBotError(
                "Не удалось найти чат по PLAYEROK_CHAT_ID. Убедитесь, что аккаунт "
                "состоит в группе и ID верный (см. команду --list-chats)."
            ) from exc

        if isinstance(entity, User):
            raise TelegramBotError("PLAYEROK_CHAT_ID указывает на личный чат, а не на группу.")
        if isinstance(entity, Channel) and not entity.megagroup:
            raise TelegramBotError("PLAYEROK_CHAT_ID указывает на канал, а не на группу.")
        if not isinstance(entity, (Chat, Channel)):
            raise TelegramBotError("PLAYEROK_CHAT_ID указывает не на группу.")

        real_id = get_peer_id(entity)
        if real_id != expected_chat_id:
            raise TelegramBotError(
                f"Несовпадение ID чата: ожидался {expected_chat_id}, найден {real_id}. "
                "Используйте ID в том виде, который выводит --list-chats."
            )

        self.chat = entity
        self.chat_id = real_id
        title = getattr(entity, "title", "без названия")
        self._log.info("Разрешённый чат найден: «%s» (ID %s). Работаю только с ним.", title, real_id)

    async def list_chats(self, limit: int = 200) -> list[tuple[str, int]]:
        result = []
        async for dialog in self.client.iter_dialogs(limit=limit):
            if dialog.is_group or dialog.is_channel:
                result.append((dialog.name or "без названия", dialog.id))
        return result

    async def fetch_context(self, limit: int) -> list:
        """Возвращает последние сообщения группы от старых к новым."""
        messages = await self.client.get_messages(self.chat, limit=limit)
        return list(reversed(messages))

    # ------------------------------------------------------------------ sending

    def flood_pause_remaining(self) -> float:
        return max(0.0, self._flood_until - time.monotonic())

    async def wait_cooldown(self) -> None:
        """Соблюдает минимальный интервал между отправками (MIN_REPLY_DELAY)."""
        elapsed = time.monotonic() - self._last_sent_at
        if elapsed < self._min_reply_delay:
            await asyncio.sleep(self._min_reply_delay - elapsed)

    async def send_text(self, text: str, reply_to: int | None = None) -> bool:
        """Отправляет сообщение. Возвращает False, если отправка не состоялась."""
        if self.flood_pause_remaining() > 0:
            self._log.info("Идёт ожидание FloodWait, сообщение пропущено.")
            return False

        await self.wait_cooldown()
        try:
            await self.client.send_message(self.chat, text, reply_to=reply_to)
        except errors.FloodWaitError as exc:
            self._flood_until = time.monotonic() + exc.seconds
            self._log.warning("Telegram FloodWait: ожидание %d сек. Все отправки приостановлены до конца ожидания.",
                              exc.seconds)
            return False
        except errors.SlowModeWaitError as exc:
            self._flood_until = time.monotonic() + exc.seconds
            self._log.warning("Включён slow mode в группе: ожидание %d сек.", exc.seconds)
            return False
        except errors.RPCError as exc:
            self._log.error("Ошибка отправки Telegram: %s", type(exc).__name__)
            return False

        self._last_sent_at = time.monotonic()
        return True

    async def send_typing(self, duration: float) -> None:
        """Имитирует набор текста на заданное время (без ошибок, если действие не поддержано)."""
        try:
            async with self.client.action(self.chat, "typing"):
                await asyncio.sleep(duration)
        except errors.RPCError:
            await asyncio.sleep(duration)
