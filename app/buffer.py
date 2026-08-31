from collections import deque
from datetime import datetime, timedelta, timezone

from app.models import TranscriptMessage


class TranscriptBuffer:
    def __init__(self, max_items: int = 40, window_seconds: int = 90) -> None:
        self._items: deque[TranscriptMessage] = deque(maxlen=max_items)
        self._window = timedelta(seconds=window_seconds)

    def add(self, message: TranscriptMessage) -> None:
        if message.is_final:
            self._items.append(message)
        self._prune()

    def _prune(self) -> None:
        cutoff = datetime.now(timezone.utc) - self._window
        while self._items and self._items[0].timestamp < cutoff:
            self._items.popleft()

    def text(self) -> str:
        self._prune()
        return "\n".join(
            f"{item.speaker or 'unknown'}: {item.text}" for item in self._items
        )

    def latest_text(self, count: int = 3) -> str:
        self._prune()
        return " ".join(item.text for item in list(self._items)[-count:])

    def __len__(self) -> int:
        self._prune()
        return len(self._items)

