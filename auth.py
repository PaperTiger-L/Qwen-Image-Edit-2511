import hashlib
import hmac
import os
import secrets
from typing import Any, Optional

import task_store

SCRYPT_N = int(os.getenv("QWEN_AUTH_SCRYPT_N", "16384"))
SCRYPT_R = int(os.getenv("QWEN_AUTH_SCRYPT_R", "8"))
SCRYPT_P = int(os.getenv("QWEN_AUTH_SCRYPT_P", "1"))


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    if not password:
        raise ValueError("密码不能为空")
    salt = salt or secrets.token_hex(16)
    password_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt.encode("utf-8"),
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=64,
    ).hex()
    return password_hash, salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    candidate_hash, _ = hash_password(password, salt=salt)
    return hmac.compare_digest(candidate_hash, password_hash)


def initialize_auth() -> None:
    task_store.init_db()
    admin_username = os.getenv("QWEN_ADMIN_USERNAME", "").strip()
    admin_password = os.getenv("QWEN_ADMIN_PASSWORD", "")
    if not admin_username or not admin_password:
        return
    password_hash, salt = hash_password(admin_password)
    task_store.upsert_user(
        username=admin_username,
        password_hash=password_hash,
        password_salt=salt,
        role=task_store.ROLE_ADMIN,
        is_active=True,
    )


def authenticate(username: str, password: str) -> bool:
    user = task_store.get_user_by_username(username)
    if not user or not user.get("is_active"):
        return False
    if not verify_password(password, user["password_hash"], user["password_salt"]):
        return False
    task_store.touch_user_login(username)
    return True


def get_user(username: Optional[str]) -> Optional[dict[str, Any]]:
    if not username:
        return None
    user = task_store.get_user_by_username(username)
    if not user or not user.get("is_active"):
        return None
    return user


def require_user(username: Optional[str]) -> dict[str, Any]:
    user = get_user(username)
    if not user:
        raise PermissionError("请先登录。")
    return user


def require_admin(username: Optional[str]) -> dict[str, Any]:
    user = require_user(username)
    if user.get("role") != task_store.ROLE_ADMIN:
        raise PermissionError("需要管理员权限。")
    return user


def create_user(username: str, password: str, role: str = task_store.ROLE_USER) -> dict[str, Any]:
    password_hash, salt = hash_password(password)
    return task_store.create_user(username, password_hash, salt, role=role, is_active=True)


def reset_password(user_id: str, password: str) -> None:
    password_hash, salt = hash_password(password)
    task_store.update_user_password(user_id, password_hash, salt)
