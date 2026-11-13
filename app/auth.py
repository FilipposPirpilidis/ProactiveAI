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
        static_token: str = "",
    ) -> None:
        self.memory = memory
        self.username = username
        self.password = password
        self.token_ttl = timedelta(seconds=token_ttl_seconds)
        self._static_token_hash = self._hash(static_token) if static_token else None

    @property
    def configured(self) -> bool:
        return self.password_signin_configured or self._static_token_hash is not None

    @property
    def password_signin_configured(self) -> bool:
        return bool(self.username and self.password)

    async def sign_in(self, username: str, password: str) -> IssuedToken | None:
        if not self.password_signin_configured:
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
        token_hash = self._hash(token)
        if self._static_token_hash and secrets.compare_digest(
            token_hash, self._static_token_hash
        ):
            return "static-token"

        record = await self.memory.auth_token(token_hash)
        if not record:
            return None
        username, expires_at = record
        if expires_at <= datetime.now(timezone.utc):
            await self.memory.delete_auth_token(token_hash)
            return None
        return username

    async def sign_out(self, token: str) -> bool:
        token_hash = self._hash(token)
        if self._static_token_hash and secrets.compare_digest(
            token_hash, self._static_token_hash
        ):
            return False
        return await self.memory.delete_auth_token(token_hash)

    @staticmethod
    def _hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()
