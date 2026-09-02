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
        # Preserve utterance boundaries so the detector can reason about a
        # question followed by another speaker's possible answer.
        return "\n".join(item.text for item in list(self._items)[-count:])

    def __len__(self) -> int:
        self._prune()
        return len(self._items)


class PartialTranscriptAssembler:
    """Reconstruct STT utterances whose cumulative partial text becomes a sliding tail."""

    def __init__(self, max_characters: int = 8_000) -> None:
        self.max_characters = max_characters
        # Providers may issue a fresh event ID for every partial revision. This
        # object belongs to one WebSocket speech stream, so continuity must not
        # depend on provider-specific event-ID behavior.
        self._text = ""

    def update(self, message: TranscriptMessage) -> TranscriptMessage:
        merged = self._merge(self._text, message.text)[-self.max_characters :]
        self._text = merged
        return message.model_copy(update={"text": merged})

    def finalize(self, message: TranscriptMessage) -> TranscriptMessage:
        assembled = self.update(message)
        self._text = ""
        return assembled

    @staticmethod
    def _merge(previous: str, current: str) -> str:
        if not previous or current.startswith(previous):
            return current
        if previous == current or previous.endswith(current):
            return previous

        # A capped Soniox partial drops characters from the left. Find the exact
        # previous-suffix/current-prefix overlap and append only genuinely new text.
        maximum = min(len(previous), len(current))
        for overlap in range(maximum, 11, -1):
            if previous.endswith(current[:overlap]):
                return previous + current[overlap:]

        # Normal STT revisions can rewrite the latest word. Prefer the corrected
        # cumulative version instead of accidentally concatenating two alternatives.
        shared_prefix = 0
        for left, right in zip(previous, current):
            if left != right:
                break
            shared_prefix += 1
        if shared_prefix >= min(len(previous), len(current)) // 2:
            return current
        return current
