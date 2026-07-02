import logging
import shutil
from datetime import timedelta
from pathlib import Path

import task_store

LOGGER = logging.getLogger(__name__)
STALE_RUNNING_MINUTES = 10


def reconcile_interrupted_jobs() -> int:
    stale_before = task_store.now_utc() - timedelta(minutes=STALE_RUNNING_MINUTES)
    interrupted = task_store.mark_stale_running_jobs_interrupted(stale_before)
    if interrupted:
        LOGGER.info("Marked interrupted jobs: %s", interrupted)
    return interrupted


def cleanup_expired_jobs() -> dict[str, int]:
    deleted = 0
    failed = 0
    skipped = 0
    for job in task_store.expired_jobs():
        if job["status"] in {task_store.JOB_STATUS_RUNNING, task_store.JOB_STATUS_FINALIZING, task_store.JOB_STATUS_QUEUED}:
            skipped += 1
            continue
        try:
            task_store.mark_job_deleting(job["id"])
            session_dir = Path(job["session_dir"])
            paths_to_delete = [session_dir]
            for key in ("manifest_file", "uploaded_package_file", "extracted_package_dir"):
                value = job.get(key)
                if value:
                    path = Path(value)
                    if path.name in {"uploads", "input_package"}:
                        paths_to_delete.append(path)
                        paths_to_delete.append(path.parent)
                    else:
                        paths_to_delete.append(path.parent)
                        paths_to_delete.append(path.parent.parent)
            for path in sorted(set(paths_to_delete), key=lambda item: len(str(item)), reverse=True):
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
            task_store.mark_job_deleted(job["id"])
            deleted += 1
            LOGGER.info("Deleted expired job: id=%s session_dir=%s", job["id"], session_dir)
        except Exception as exc:
            failed += 1
            LOGGER.exception("Failed to delete expired job id=%s: %s", job["id"], exc)
            task_store.update_job(job["id"], status=job["status"], last_error=str(exc))
    LOGGER.info("Cleanup expired jobs completed: deleted=%s skipped=%s failed=%s", deleted, skipped, failed)
    return {"deleted": deleted, "skipped": skipped, "failed": failed}
