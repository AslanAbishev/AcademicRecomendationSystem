from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.schemas import (
    AccountRegistrationRequest,
    AuthSessionResponse,
    AuthenticatedUser,
    ResearcherRegistrationRequest,
    UserLoginRequest,
)
from app.services.registration_service import RegistrationService


HASH_ITERATIONS = 120_000


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_email(email: str) -> str:
    return email.strip().lower()


class AccountService:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        payload: AccountRegistrationRequest,
        registration_service: RegistrationService,
    ) -> AuthSessionResponse:
        email = _normalize_email(payload.email)
        users = self._read_all()
        if any(item.get("email") == email for item in users):
            raise ValueError("An account with this email already exists")

        profile_payload = ResearcherRegistrationRequest.model_validate(
            payload.model_dump(exclude={"password"})
        )
        profile = registration_service.register(profile_payload)
        salt = secrets.token_hex(16)
        token = secrets.token_urlsafe(32)
        created_at = _utc_now()
        user = {
            "user_id": str(uuid4()),
            "email": email,
            "password_salt": salt,
            "password_hash": self._hash_password(payload.password, salt),
            "profile_id": profile.profile_id,
            "created_at": created_at,
            "last_login_at": created_at,
            "sessions": [self._session_record(token)],
        }
        users.append(user)
        self._write_all(users)
        return AuthSessionResponse(
            access_token=token,
            user=self._public_user(user),
            profile=profile,
        )

    def login(
        self,
        payload: UserLoginRequest,
        registration_service: RegistrationService,
    ) -> AuthSessionResponse:
        users = self._read_all()
        user_index = self._find_user_index_by_email(users, payload.email)
        if user_index is None:
            raise PermissionError("Invalid email or password")

        user = users[user_index]
        expected = user.get("password_hash", "")
        actual = self._hash_password(payload.password, user.get("password_salt", ""))
        if not hmac.compare_digest(expected, actual):
            raise PermissionError("Invalid email or password")

        token = secrets.token_urlsafe(32)
        user["last_login_at"] = _utc_now()
        user.setdefault("sessions", []).append(self._session_record(token))
        users[user_index] = user
        self._write_all(users)
        profile = registration_service.get_profile(user["profile_id"])
        return AuthSessionResponse(
            access_token=token,
            user=self._public_user(user),
            profile=profile,
        )

    def get_user_by_token(self, token: str) -> AuthenticatedUser:
        token_hash = self._hash_token(token)
        for user in self._read_all():
            sessions = user.get("sessions") or []
            if any(session.get("token_hash") == token_hash for session in sessions):
                return self._public_user(user)
        raise PermissionError("Invalid or expired session")

    def revoke_token(self, token: str) -> None:
        token_hash = self._hash_token(token)
        users = self._read_all()
        changed = False
        for user in users:
            sessions = user.get("sessions") or []
            filtered = [session for session in sessions if session.get("token_hash") != token_hash]
            if len(filtered) != len(sessions):
                user["sessions"] = filtered
                changed = True
        if changed:
            self._write_all(users)

    def _find_user_index_by_email(self, users: list[dict], email: str) -> int | None:
        normalized = _normalize_email(email)
        for index, user in enumerate(users):
            if user.get("email") == normalized:
                return index
        return None

    def _session_record(self, token: str) -> dict[str, str]:
        return {
            "token_hash": self._hash_token(token),
            "created_at": _utc_now(),
        }

    def _public_user(self, user: dict) -> AuthenticatedUser:
        return AuthenticatedUser(
            user_id=user["user_id"],
            email=user["email"],
            profile_id=user["profile_id"],
            created_at=user["created_at"],
            last_login_at=user.get("last_login_at"),
        )

    def _hash_password(self, password: str, salt: str) -> str:
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            HASH_ITERATIONS,
        )
        return digest.hex()

    def _hash_token(self, token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _read_all(self) -> list[dict]:
        if not self.storage_path.exists():
            return []
        raw = self.storage_path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        return json.loads(raw)

    def _write_all(self, users: list[dict]) -> None:
        self.storage_path.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
