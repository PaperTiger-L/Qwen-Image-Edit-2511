import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

DB_PATH = Path(os.getenv("QWEN_APP_DB", Path(__file__).resolve().parent / "runtime" / "app.db"))
RETENTION_DAYS = int(os.getenv("OUTPUT_RETENTION_DAYS", "7"))

ROLE_ADMIN = "admin"
ROLE_USER = "user"
USER_ROLES = {ROLE_ADMIN, ROLE_USER}

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_RUNNING = "running"
JOB_STATUS_FINALIZING = "finalizing"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"
JOB_STATUS_INTERRUPTED = "interrupted"
JOB_STATUS_DELETING = "deleting"
JOB_STATUS_DELETED = "deleted"

JOB_ITEM_PENDING = "pending"
JOB_ITEM_RUNNING = "running"
JOB_ITEM_SUCCESS = "success"
JOB_ITEM_FAILED = "failed"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def parse_utc(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def new_id() -> str:
    return uuid.uuid4().hex


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def loads_json(value: Optional[str], default: Any = None) -> Any:
    if value in (None, ""):
        return default
    return json.loads(value)


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def row_to_dict(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    return dict(row) if row is not None else None


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                role TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT
            );

            CREATE TABLE IF NOT EXISTS user_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_token_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expire_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                is_revoked INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                display_name TEXT,
                session_dir TEXT NOT NULL,
                batch_mode TEXT,
                manifest_file TEXT,
                uploaded_package_file TEXT,
                extracted_package_dir TEXT,
                single_result_file TEXT,
                results_csv_file TEXT,
                results_json_file TEXT,
                results_zip_file TEXT,
                download_ready INTEGER NOT NULL DEFAULT 0,
                total_items INTEGER NOT NULL DEFAULT 0,
                completed_items INTEGER NOT NULL DEFAULT 0,
                success_items INTEGER NOT NULL DEFAULT 0,
                failed_items INTEGER NOT NULL DEFAULT 0,
                progress REAL NOT NULL DEFAULT 0,
                current_index INTEGER NOT NULL DEFAULT 0,
                current_row_id TEXT,
                current_phase TEXT,
                last_error TEXT,
                started_at TEXT,
                finished_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                heartbeat_at TEXT,
                retention_expire_at TEXT,
                cleanup_started_at TEXT,
                deleted_at TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_user_updated ON jobs(user_id, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_retention ON jobs(retention_expire_at);

            CREATE TABLE IF NOT EXISTS job_items (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                row_id TEXT NOT NULL,
                prompt TEXT NOT NULL,
                negative_prompt TEXT NOT NULL,
                image_refs_json TEXT NOT NULL,
                resolved_image_paths_json TEXT NOT NULL,
                seed INTEGER NOT NULL,
                num_inference_steps INTEGER NOT NULL,
                guidance_scale REAL NOT NULL,
                true_cfg_scale REAL NOT NULL,
                status TEXT NOT NULL,
                output_image TEXT,
                error TEXT,
                traceback TEXT,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(job_id) REFERENCES jobs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_job_items_job_order ON job_items(job_id, row_index);
            CREATE INDEX IF NOT EXISTS idx_job_items_status ON job_items(job_id, status);

            CREATE TABLE IF NOT EXISTS admin_audit_logs (
                id TEXT PRIMARY KEY,
                actor_user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                detail_json TEXT,
                created_at TEXT NOT NULL
            );
            """
        )


def create_user(username: str, password_hash: str, password_salt: str, role: str = ROLE_USER, is_active: bool = True) -> dict[str, Any]:
    username = username.strip()
    if not username:
        raise ValueError("用户名不能为空")
    if role not in USER_ROLES:
        raise ValueError("无效用户角色")
    user_id = new_id()
    now = isoformat_utc(now_utc())
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO users (id, username, password_hash, password_salt, role, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, username, password_hash, password_salt, role, 1 if is_active else 0, now, now),
        )
        return get_user_by_id(user_id, conn=conn)


def upsert_user(username: str, password_hash: str, password_salt: str, role: str = ROLE_USER, is_active: bool = True) -> dict[str, Any]:
    username = username.strip()
    if role not in USER_ROLES:
        raise ValueError("无效用户角色")
    now = isoformat_utc(now_utc())
    with connect() as conn:
        existing = get_user_by_username(username, conn=conn)
        if existing:
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, password_salt = ?, role = ?, is_active = ?, updated_at = ?
                WHERE username = ?
                """,
                (password_hash, password_salt, role, 1 if is_active else 0, now, username),
            )
            return get_user_by_username(username, conn=conn)
        user_id = new_id()
        conn.execute(
            """
            INSERT INTO users (id, username, password_hash, password_salt, role, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, username, password_hash, password_salt, role, 1 if is_active else 0, now, now),
        )
        return get_user_by_id(user_id, conn=conn)


def get_user_by_username(username: str, conn: Optional[sqlite3.Connection] = None) -> Optional[dict[str, Any]]:
    def query(c):
        return row_to_dict(c.execute("SELECT * FROM users WHERE username = ?", (username.strip(),)).fetchone())
    if conn is not None:
        return query(conn)
    with connect() as c:
        return query(c)


def get_user_by_id(user_id: str, conn: Optional[sqlite3.Connection] = None) -> Optional[dict[str, Any]]:
    def query(c):
        return row_to_dict(c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())
    if conn is not None:
        return query(conn)
    with connect() as c:
        return query(c)


def list_users() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, username, role, is_active, created_at, updated_at, last_login_at FROM users ORDER BY created_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]


def set_user_active(user_id: str, is_active: bool) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE users SET is_active = ?, updated_at = ? WHERE id = ?",
            (1 if is_active else 0, isoformat_utc(now_utc()), user_id),
        )


def update_user_password(user_id: str, password_hash: str, password_salt: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, password_salt = ?, updated_at = ? WHERE id = ?",
            (password_hash, password_salt, isoformat_utc(now_utc()), user_id),
        )


def touch_user_login(username: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE users SET last_login_at = ?, updated_at = ? WHERE username = ?",
            (isoformat_utc(now_utc()), isoformat_utc(now_utc()), username),
        )


def create_job(user_id: str, job_type: str, session_dir: Path, status: str = JOB_STATUS_QUEUED, batch_mode: Optional[str] = None, display_name: Optional[str] = None) -> dict[str, Any]:
    job_id = session_dir.name
    now = isoformat_utc(now_utc())
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, user_id, job_type, status, display_name, session_dir, batch_mode, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, user_id, job_type, status, display_name, str(session_dir), batch_mode, now, now),
        )
        return get_job(job_id, conn=conn)


def get_job(job_id: str, conn: Optional[sqlite3.Connection] = None) -> Optional[dict[str, Any]]:
    def query(c):
        return row_to_dict(c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())
    if conn is not None:
        return query(conn)
    with connect() as c:
        return query(c)


def get_job_for_user(job_id: str, user: dict[str, Any]) -> Optional[dict[str, Any]]:
    job = get_job(job_id)
    if not job:
        return None
    if user.get("role") == ROLE_ADMIN or job.get("user_id") == user.get("id"):
        return job
    return None


def update_job(job_id: str, **fields: Any) -> Optional[dict[str, Any]]:
    if not fields:
        return get_job(job_id)
    fields["updated_at"] = isoformat_utc(now_utc())
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [job_id]
    with connect() as conn:
        conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", values)
        return get_job(job_id, conn=conn)


def finish_job(job_id: str, status: str, **fields: Any) -> Optional[dict[str, Any]]:
    finished_at = isoformat_utc(now_utc())
    fields.update(
        {
            "status": status,
            "finished_at": finished_at,
            "retention_expire_at": isoformat_utc(now_utc() + timedelta(days=RETENTION_DAYS)),
        }
    )
    return update_job(job_id, **fields)


def list_jobs_for_user(user: dict[str, Any], limit: int = 100) -> list[dict[str, Any]]:
    with connect() as conn:
        if user.get("role") == ROLE_ADMIN:
            rows = conn.execute(
                """
                SELECT jobs.*, users.username
                FROM jobs JOIN users ON users.id = jobs.user_id
                WHERE jobs.status != ?
                ORDER BY jobs.created_at DESC LIMIT ?
                """,
                (JOB_STATUS_DELETED, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT jobs.*, users.username
                FROM jobs JOIN users ON users.id = jobs.user_id
                WHERE jobs.user_id = ? AND jobs.status != ?
                ORDER BY jobs.created_at DESC LIMIT ?
                """,
                (user["id"], JOB_STATUS_DELETED, limit),
            ).fetchall()
        return [dict(row) for row in rows]


def create_job_item(job_id: str, row_index: int, row_id: str, prompt: str, negative_prompt: str, image_refs: list[str], image_paths: list[Path], seed: int, num_inference_steps: int, guidance_scale: float, true_cfg_scale: float) -> dict[str, Any]:
    item_id = new_id()
    now = isoformat_utc(now_utc())
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO job_items (
                id, job_id, row_index, row_id, prompt, negative_prompt, image_refs_json,
                resolved_image_paths_json, seed, num_inference_steps, guidance_scale,
                true_cfg_scale, status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                job_id,
                row_index,
                row_id,
                prompt,
                negative_prompt,
                dumps_json(image_refs),
                dumps_json([str(path) for path in image_paths]),
                seed,
                num_inference_steps,
                guidance_scale,
                true_cfg_scale,
                JOB_ITEM_PENDING,
                now,
            ),
        )
        return dict(conn.execute("SELECT * FROM job_items WHERE id = ?", (item_id,)).fetchone())


def list_job_items(job_id: str) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM job_items WHERE job_id = ? ORDER BY row_index", (job_id,)).fetchall()
        return [dict(row) for row in rows]


def update_job_item(item_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = isoformat_utc(now_utc())
    assignments = ", ".join(f"{key} = ?" for key in fields)
    values = list(fields.values()) + [item_id]
    with connect() as conn:
        conn.execute(f"UPDATE job_items SET {assignments} WHERE id = ?", values)


def aggregate_job_items(job_id: str) -> dict[str, int]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM job_items WHERE job_id = ? GROUP BY status",
            (job_id,),
        ).fetchall()
    counts = {row["status"]: int(row["count"]) for row in rows}
    success = counts.get(JOB_ITEM_SUCCESS, 0)
    failed = counts.get(JOB_ITEM_FAILED, 0)
    completed = success + failed
    total = sum(counts.values())
    return {"total": total, "success": success, "failed": failed, "completed": completed}


def update_job_progress_from_items(job_id: str) -> Optional[dict[str, Any]]:
    counts = aggregate_job_items(job_id)
    total = counts["total"]
    progress = counts["completed"] / total if total else 0.0
    return update_job(
        job_id,
        total_items=total,
        completed_items=counts["completed"],
        success_items=counts["success"],
        failed_items=counts["failed"],
        progress=progress,
        heartbeat_at=isoformat_utc(now_utc()),
    )


def rows_for_batch_results(job_id: str) -> list[dict[str, Any]]:
    rows = []
    for item in list_job_items(job_id):
        rows.append(
            {
                "id": item["row_id"],
                "prompt": item["prompt"],
                "negative_prompt": item["negative_prompt"],
                "images": "|".join(loads_json(item["image_refs_json"], [])),
                "seed": item["seed"],
                "num_inference_steps": item["num_inference_steps"],
                "guidance_scale": item["guidance_scale"],
                "true_cfg_scale": item["true_cfg_scale"],
                "status": "success" if item["status"] == JOB_ITEM_SUCCESS else "failed" if item["status"] == JOB_ITEM_FAILED else item["status"],
                "output_image": item.get("output_image") or "",
                "error": item.get("error") or "",
                "traceback": item.get("traceback") or "",
            }
        )
    return rows


def expired_jobs(now: Optional[datetime] = None) -> list[dict[str, Any]]:
    now_value = isoformat_utc(now or now_utc())
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs
            WHERE retention_expire_at IS NOT NULL
              AND retention_expire_at <= ?
              AND status IN (?, ?, ?)
            ORDER BY retention_expire_at ASC
            """,
            (now_value, JOB_STATUS_COMPLETED, JOB_STATUS_FAILED, JOB_STATUS_INTERRUPTED),
        ).fetchall()
        return [dict(row) for row in rows]


def mark_stale_running_jobs_interrupted(stale_before: datetime) -> int:
    stale_value = isoformat_utc(stale_before)
    now = isoformat_utc(now_utc())
    with connect() as conn:
        cursor = conn.execute(
            """
            UPDATE jobs
            SET status = ?, current_phase = ?, last_error = COALESCE(last_error, ?), updated_at = ?
            WHERE status IN (?, ?) AND heartbeat_at IS NOT NULL AND heartbeat_at < ?
            """,
            (
                JOB_STATUS_INTERRUPTED,
                "任务中断",
                "服务重启或任务线程中断，批量任务未完成。",
                now,
                JOB_STATUS_RUNNING,
                JOB_STATUS_FINALIZING,
                stale_value,
            ),
        )
        return cursor.rowcount


def mark_job_deleted(job_id: str) -> None:
    update_job(job_id, status=JOB_STATUS_DELETED, deleted_at=isoformat_utc(now_utc()), download_ready=0)


def mark_job_deleting(job_id: str) -> None:
    update_job(job_id, status=JOB_STATUS_DELETING, cleanup_started_at=isoformat_utc(now_utc()), download_ready=0)


def audit(actor_user_id: str, action: str, target_type: str, target_id: str, detail: Optional[dict[str, Any]] = None) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO admin_audit_logs (id, actor_user_id, action, target_type, target_id, detail_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (new_id(), actor_user_id, action, target_type, target_id, dumps_json(detail or {}), isoformat_utc(now_utc())),
        )
