import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.telegram import TelegramAdapter
from telegram.constants import ChatType as _ChatType


@pytest.fixture
def adapter():
    config = PlatformConfig(
        enabled=True,
        token="fake-token",
        extra={
            "group_topics": [
                {
                    "chat_id": -1003924391517,
                    "topics": [
                        {"name": "Task", "thread_id": 39},
                    ],
                }
            ],
            "prefix_topic_routes": [
                {
                    "chat_id": -1003924391517,
                    "prefix": "nhac toi",
                    "thread_id": 39,
                    "topic_name": "Task",
                    "strip_prefix": True,
                }
            ],
        },
    )
    a = TelegramAdapter(config)
    a.handle_message = AsyncMock()
    return a


def _make_message(text: str, *, chat_id: int = -1003924391517, thread_id=None):
    chat = SimpleNamespace(id=chat_id, type=_ChatType.SUPERGROUP, title="My Team")
    user = SimpleNamespace(id=123, full_name="Lê Sơn")
    return SimpleNamespace(
        chat=chat,
        from_user=user,
        text=text,
        caption=None,
        message_thread_id=thread_id,
        message_id=777,
        reply_to_message=None,
        date=None,
    )


def _make_update(message):
    return SimpleNamespace(message=message)


@pytest.mark.asyncio
async def test_nhac_toi_prefix_routes_to_task_topic(adapter):
    update = _make_update(_make_message("nhac toi mai 8h hop standup"))

    await adapter._handle_text_message(update, None)
    await asyncio.sleep(adapter._text_batch_delay_seconds + 0.2)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "mai 8h hop standup"
    assert event.source.thread_id == "39"
    assert event.source.chat_topic == "Task"
    assert event.message_type == MessageType.TEXT


class _FrozenChat:
    """Simulate telegram.Chat objects that reject post-init title writes."""

    def __init__(self, chat_id: int = -1003924391517):
        object.__setattr__(self, "id", chat_id)
        object.__setattr__(self, "type", _ChatType.SUPERGROUP)
        object.__setattr__(self, "title", "My Team")

    def __copy__(self):
        return self

    def __setattr__(self, name, value):
        if name == "title" and hasattr(self, "title"):
            raise AttributeError("Attribute `title` of class `Chat` can't be set!")
        object.__setattr__(self, name, value)


class _FrozenMessage:
    """Simulate telegram.Message refusing post-init writes to message_thread_id."""

    def __init__(self, text: str, *, chat_id: int = -1003924391517, thread_id=None):
        object.__setattr__(self, "chat", _FrozenChat(chat_id))
        object.__setattr__(self, "from_user", SimpleNamespace(id=123, full_name="Lê Sơn"))
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "caption", None)
        object.__setattr__(self, "message_thread_id", thread_id)
        object.__setattr__(self, "message_id", 888)
        object.__setattr__(self, "reply_to_message", None)
        object.__setattr__(self, "date", None)

    def __setattr__(self, name, value):
        if name == "message_thread_id" and hasattr(self, "message_thread_id"):
            raise AttributeError("Attribute `message_thread_id` of class `Message` can't be set!")
        object.__setattr__(self, name, value)


@pytest.mark.asyncio
async def test_prefix_route_does_not_mutate_immutable_telegram_message(adapter):
    update = _make_update(_FrozenMessage("nhac toi mai 8h hop standup"))

    await adapter._handle_text_message(update, None)
    await asyncio.sleep(adapter._text_batch_delay_seconds + 0.2)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "mai 8h hop standup"
    assert event.source.thread_id == "39"
    assert event.source.chat_topic == "Task"


@pytest.mark.asyncio
async def test_non_matching_text_stays_in_current_thread(adapter):
    update = _make_update(_make_message("hello world", thread_id=157))

    await adapter._handle_text_message(update, None)
    await asyncio.sleep(adapter._text_batch_delay_seconds + 0.2)

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello world"
    assert event.source.thread_id == "157"
