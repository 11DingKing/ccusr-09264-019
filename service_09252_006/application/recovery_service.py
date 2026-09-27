"""失败恢复作业：失败队列的登记、重试与恢复（运维操作）。

- 作业（失败步骤、重试次数、最后错误）持久化在 SQLite recovery_jobs 表，
  服务重启后失败历史不丢失；
- 重试在 BEGIN IMMEDIATE 事务内先复查状态：已被并发重试恢复的作业直接
  回放，处理器不会重复执行；
- 重试成功把作业条件推进到 succeeded（WHERE status='failed'），恰好一次；
  处理器的仓库写入与状态推进同事务提交，处理器抛错则整体回滚，再由独立
  事务把这次失败（次数+1、最后错误）落库。
"""
from __future__ import annotations

from typing import Callable

from ..domain.enums import RecoveryJobStatus, Role
from ..domain.errors import NotFoundError, ValidationError
from ..domain.models import RecoveryJob, User
from .base import Service, require_roles

StepHandler = Callable[[dict], None]
"""重试处理器：接收作业 payload；抛异常表示本次重试仍失败。"""


class _HandlerFailed(Exception):
    """处理器失败：回滚成功事务，改由独立事务记录这次失败。"""


class RecoveryService(Service):
    """失败恢复作业（运维操作）。"""

    def __init__(self, repo, clock, ids) -> None:
        super().__init__(repo, clock, ids)
        self._handlers: dict[str, StepHandler] = {}

    def register_step(self, step: str, handler: StepHandler) -> None:
        """登记失败步骤的重试处理器（进程内存，重启后需重新登记）。"""
        self._handlers[step] = handler

    # ------------------------------------------------------------ 登记失败
    def record_failure(
        self,
        actor: User,
        *,
        step: str,
        error: str,
        payload: dict | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        require_roles(actor, Role.QUALITY_AUTHORITY)
        if not step or not step.strip():
            raise ValidationError("失败步骤不能为空")
        if not error or not error.strip():
            raise ValidationError("错误消息不能为空")

        def work() -> dict:
            now = self.clock.now_iso()
            job = RecoveryJob(
                job_id=self.ids.new_id("job"),
                step=step.strip(),
                status=RecoveryJobStatus.FAILED.value,
                payload=dict(payload or {}),
                retry_count=0,
                last_error=error.strip(),
                created_at=now,
                updated_at=now,
                resolved_at=None,
            )
            self.repo.insert_recovery_job(job)
            self.audit(
                actor.user_id,
                "recovery.recorded",
                detail={"job_id": job.job_id, "step": job.step},
            )
            return self._job_dict(job)

        return self.idempotent(idempotency_key, work)

    # ---------------------------------------------------------------- 查询
    def get_job(self, actor: User, job_id: str) -> dict:
        require_roles(actor, Role.QUALITY_AUTHORITY, Role.AUDITOR)
        job = self.repo.get_recovery_job(job_id)
        if job is None:
            raise NotFoundError("恢复作业不存在")
        return self._job_dict(job)

    def list_jobs(self, actor: User, *, status: str | None = None) -> list[dict]:
        require_roles(actor, Role.QUALITY_AUTHORITY, Role.AUDITOR)
        valid = {s.value for s in RecoveryJobStatus}
        if status is not None and status not in valid:
            raise ValidationError("未知作业状态", details={"status": status})
        return [self._job_dict(j) for j in self.repo.list_recovery_jobs(status)]

    # ---------------------------------------------------------------- 重试
    def retry(self, actor: User, *, job_id: str) -> dict:
        """重试一个失败作业；无论成败，状态都只推进一次。

        不需要幂等键：成功推进用条件更新 + 事务内复查保证恰好一次，
        并发或重复调用只会回放当前状态。
        """
        require_roles(actor, Role.QUALITY_AUTHORITY)
        try:
            with self.repo.transaction():
                job = self.repo.get_recovery_job(job_id)
                if job is None:
                    raise NotFoundError("恢复作业不存在")
                if job.status != RecoveryJobStatus.FAILED.value:
                    # 已恢复：只回放，不再推进（并发/重复重试安全）
                    return self._job_dict(job, replayed=True)
                handler = self._handlers.get(job.step)
                if handler is None:
                    raise ValidationError(
                        "未登记该步骤的重试处理器", details={"step": job.step}
                    )
                try:
                    handler(dict(job.payload))
                except Exception as exc:  # noqa: BLE001 处理器失败也是可恢复信号
                    raise _HandlerFailed(str(exc)) from exc
                moved = self.repo.complete_recovery_job(
                    job.job_id, RecoveryJobStatus.FAILED.value, self.clock.now_iso()
                )
                fresh = self.repo.get_recovery_job(job.job_id)
                self.audit(
                    actor.user_id,
                    "recovery.succeeded",
                    detail={"job_id": job.job_id, "retry_count": fresh.retry_count},
                )
                return self._job_dict(fresh, replayed=not moved)
        except _HandlerFailed as failed:
            # 成功事务已回滚（处理器写入一并撤销）；独立事务记录这次失败
            with self.repo.transaction():
                moved = self.repo.note_recovery_retry_failure(
                    job_id,
                    RecoveryJobStatus.FAILED.value,
                    str(failed),
                    self.clock.now_iso(),
                )
                fresh = self.repo.get_recovery_job(job_id)
                if moved:
                    self.audit(
                        actor.user_id,
                        "recovery.retry_failed",
                        detail={
                            "job_id": job_id,
                            "error": str(failed),
                            "retry_count": fresh.retry_count,
                        },
                    )
                return self._job_dict(fresh, retried=moved, replayed=not moved)

    # ---------------------------------------------------------------- 视图
    @staticmethod
    def _job_dict(job: RecoveryJob, **flags) -> dict:
        result = {
            "job_id": job.job_id,
            "step": job.step,
            "status": job.status,
            "payload": job.payload,
            "retry_count": job.retry_count,
            "last_error": job.last_error,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "resolved_at": job.resolved_at,
        }
        result.update(flags)
        return result
