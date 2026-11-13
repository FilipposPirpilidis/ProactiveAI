from app.auth import AuthService
from app.memory import MemoryEngine


async def test_token_is_stored_hashed_and_can_be_revoked(tmp_path) -> None:
    memory = MemoryEngine(str(tmp_path / "auth.db"))
    await memory.initialize()
    try:
        auth = AuthService(memory, "homebuddy", "secret", 3_600)

        assert await auth.sign_in("homebuddy", "wrong") is None
        issued = await auth.sign_in("homebuddy", "secret")
        assert issued is not None
        assert await auth.validate(issued.access_token) == "homebuddy"

        cursor = await memory._db.execute("SELECT token_hash FROM auth_tokens")  # type: ignore[union-attr]
        stored_hash = str((await cursor.fetchone())["token_hash"])
        assert stored_hash != issued.access_token
        assert issued.access_token not in stored_hash

        assert await auth.sign_out(issued.access_token) is True
        assert await auth.validate(issued.access_token) is None
    finally:
        await memory.close()


async def test_expired_token_is_deleted_on_validation(tmp_path) -> None:
    memory = MemoryEngine(str(tmp_path / "expired.db"))
    await memory.initialize()
    try:
        auth = AuthService(memory, "homebuddy", "secret", -1)
        issued = await auth.sign_in("homebuddy", "secret")
        assert issued is not None

        assert await auth.validate(issued.access_token) is None
        assert await auth.sign_out(issued.access_token) is False
    finally:
        await memory.close()


async def test_static_token_works_without_password_signin_and_cannot_be_revoked(
    tmp_path,
) -> None:
    memory = MemoryEngine(str(tmp_path / "static-token.db"))
    await memory.initialize()
    try:
        auth = AuthService(memory, "", "", 3_600, "shared-secret-token")

        assert auth.configured is True
        assert auth.password_signin_configured is False
        assert await auth.sign_in("", "") is None
        assert await auth.validate("shared-secret-token") == "static-token"
        assert await auth.validate("shared-secret-tokeN") is None
        assert await auth.sign_out("shared-secret-token") is False
        assert await auth.validate("shared-secret-token") == "static-token"
    finally:
        await memory.close()
