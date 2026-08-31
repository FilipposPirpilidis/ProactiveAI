import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.memory import MemoryEngine


@dataclass(frozen=True)
class IssuedToken:
    access_token: str
    expires_at: datetime


class AuthService:
    def __init__(
        self,
        memory: MemoryEngine,
        username: str,
        password: str,
        token_ttl_seconds: int,
    ) -> None:
        self.memory = memory
        self.username = username
        self.password = password
        self.token_ttl = timedelta(seconds=token_ttl_seconds)

    @property
    def configured(self) -> bool:
        return bool(self.username and self.password)

    async def sign_in(self, username: str, password: str) -> IssuedToken | None:
        if not self.configured:
            return None
        username_matches = secrets.compare_digest(
            username.encode("utf-8"), self.username.encode("utf-8")
        )
        password_matches = secrets.compare_digest(
            password.encode("utf-8"), self.password.encode("utf-8")
        )
        if not (username_matches and password_matches):
            return None

        now = datetime.now(timezone.utc)
        expires_at = now + self.token_ttl
        token = secrets.token_urlsafe(32)
        await self.memory.delete_expired_auth_tokens(now)
        await self.memory.store_auth_token(self._hash(token), username, now, expires_at)
        return IssuedToken(token, expires_at)

    async def validate(self, token: str) -> str | None:
        if not token:
            return None
        record = await self.memory.auth_token(self._hash(token))
        if not record:
            return None
        username, expires_at = record
        if expires_at <= datetime.now(timezone.utc):
            await self.memory.delete_auth_token(self._hash(token))
            return None
        return username

    async def sign_out(self, token: str) -> bool:
        return await self.memory.delete_auth_token(self._hash(token))

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()
