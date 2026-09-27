"""失败恢复作业：失败队列的登记、查看与重试（运维操作）。

- 作业与每次失败/重试事件都写入 SQLite，服务重启后队列与失败历史不丢；
- 重试在单个写事务内完成“业务状态推进 + 作业条件完成”，业务状态迁移
  本身也是条件更新，因此重试成功只推进一次，重复重试只是回放；
- 重试失败时业务事务整体回滚，失败步骤/重试次数/最后错误在独立事务
  中记录，不随业务回滚丢失。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..domain.enums import RecoveryJobStatus, Role
from ..domain.errors import (
    DomainError,
    NotFoundError,
    ValidationError,
)
from ..domain.models import RecoveryAttempt, RecoveryJob, User
from .base import Service, require_roles

if TYPE_CHECKING:
    from ..application.package_service import PackageService
    from ..application.review_service import ReviewService
    from ..application.ports import Clock, IdGenerator
    from ..application.repository import Repository

OP_PACKAGE_SEAL = "package.seal"
OP_DECISION_ISSUE = "decision.issue"

# 运维操作 -> 失败步骤标签（重试时实际执行的步骤）
OPERATION_STEPS = {
    OP_PACKAGE_SEAL: "seal_package",
    OP_DECISION_ISSUE: "issue_decision",
}


class RecoveryService(Service):
    def __init__(
        self,
        repo: "Repository",
        clock: "Clock",
        ids: "IdGenerator",
        *,
        packages: "PackageService",
        reviews: "ReviewService",
    ) -> None:
        super().__init__(repo, clock, ids)
        self._packages = packages
        self._reviews = reviews

    # ------------------------------------------------------------- 登记
    def record_failure(
        self,
        actor: User,
        *,
        operation: str,
        target_id: str,
        failed_step: str,
        error: str,
        payload: dict | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """把一次失败的运维操作登记进失败队列。

        同一操作同一对象已有待恢复作业时，不重复建单，只刷新失败步骤与
        最后错误（失败事件仍计入历史）。
        """
        require_roles(actor, Role.QUALITY_AUTHORITY)
        if operation not in OPERATION_STEPS:
            raise ValidationError("未知的运维操作", details={"operation": operation})
        if not target_id or not target_id.strip():
            raise ValidationError("恢复作业缺少作用对象")
        if not failed_step or not failed_step.strip():
            raise ValidationError("失败步骤不能为空")
        if not error or not error.strip():
            raise ValidationError("错误消息不能为空")
        if payload is not None and not isinstance(payload, dict):
            raise ValidationError("payload 必须是对象")
        step = failed_step.strip()
        message = error.strip()

        def work() -> dict:
            now = self.clock.now_iso()
            existing = self.repo.find_pending_recovery_job(
                operation, target_id.strip()
            )
            if existing is not None:
                self.repo.note_recovery_failure(
                    existing.job_id, step, message, now, count_retry=False
                )
                self._append_attempt(existing.job_id, step, False, message, now)
                return self._job_dict(
                    self.repo.get_recovery_job(existing.job_id), replayed=True
                )
            job = RecoveryJob(
                job_id=self.ids.new_id("job"),
                operation=operation,
                target_id=target_id.strip(),
                payload=dict(payload or {}),
                status=RecoveryJobStatus.PENDING.value,
                failed_step=step,
                retry_count=0,
                last_error=message,
                created_by=actor.user_id,
                created_at=now,
                updated_at=now,
                completed_at=None,
            )
            self.repo.insert_recovery_job(job)
            self._append_attempt(job.job_id, step, False, message, now)
            self.audit(
                actor.user_id, "recovery.job_recorded",
                detail={
                    "job_id": job.job_id,
                    "operation": operation,
                    "target_id": job.target_id,
                    "failed_step": step,
                },
            )
            return self._job_dict(job)

        return self.idempotent(idempotency_key, work)

    # ------------------------------------------------------------- 查询
    def list_jobs(self, actor: User, *, status: str | None = None) -> list[dict]:
        require_roles(actor, Role.QUALITY_AUTHORITY, Role.AUDITOR)
        if status is not None and status not in {s.value for s in RecoveryJobStatus}:
            raise ValidationError("未知的作业状态", details={"status": status})
        return [self._job_dict(j) for j in self.repo.list_recovery_jobs(status)]

    def get_job(self, actor: User, job_id: str) -> dict:
        require_roles(actor, Role.QUALITY_AUTHORITY, Role.AUDITOR)
        job = self.repo.get_recovery_job(job_id)
        if job is None:
            raise NotFoundError("恢复作业不存在")
        result = self._job_dict(job)
        result["attempts"] = [
            self._attempt_dict(a) for a in self.repo.list_recovery_attempts(job_id)
        ]
        return result

    # ------------------------------------------------------------- 重试
    def retry(
        self,
        actor: User,
        *,
        job_id: str,
        idempotency_key: str | None = None,
    ) -> dict:
        """重试失败作业。

        成功：业务状态与作业状态在同一事务内各推进一次；已完成的作业
        再重试只是回放，不会重复推进。失败：业务事务回滚，失败步骤、
        重试次数与最后错误在独立事务中记录后，把原错误继续抛给调用方。
        """
        require_roles(actor, Role.QUALITY_AUTHORITY)

        def work() -> dict:
            job = self.repo.get_recovery_job(job_id)
            if job is None:
                raise NotFoundError("恢复作业不存在")
            if job.status == RecoveryJobStatus.SUCCEEDED.value:
                return self._job_dict(job, replayed=True)
            outcome = self._execute(job, actor)
            now = self.clock.now_iso()
            moved = self.repo.complete_recovery_job(
                job_id, RecoveryJobStatus.PENDING.value, now
            )
            if not moved:
                # 并发重试：另一方已完成作业；业务状态迁移自身也是条件
                # 更新，不会重复推进
                return self._job_dict(
                    self.repo.get_recovery_job(job_id), replayed=True
                )
            self._append_attempt(
                job_id, OPERATION_STEPS.get(job.operation, job.operation),
                True, None, now,
            )
            self.audit(
                actor.user_id, "recovery.retry_succeeded",
                detail={
                    "job_id": job_id,
                    "operation": job.operation,
                    "target_id": job.target_id,
                },
            )
            result = self._job_dict(self.repo.get_recovery_job(job_id))
            result["outcome"] = outcome
            return result

        try:
            return self.idempotent(idempotency_key, work)
        except Exception as exc:
            self._record_retry_failure(actor, job_id, exc)
            raise

    # ------------------------------------------------------------- 内部
    def _execute(self, job: RecoveryJob, actor: User) -> dict:
        if job.operation == OP_PACKAGE_SEAL:
            return self._packages.seal_package(actor, package_id=job.target_id)
        if job.operation == OP_DECISION_ISSUE:
            decision = job.payload.get("decision")
            if not decision:
                raise ValidationError(
                    "恢复作业缺少 decision 参数", details={"job_id": job.job_id}
                )
            return self._reviews.issue_decision(
                actor,
                package_id=job.target_id,
                decision=decision,
                note=job.payload.get("note", ""),
            )
        raise ValidationError("未知的运维操作", details={"operation": job.operation})

    def _record_retry_failure(
        self, actor: User, job_id: str, exc: Exception
    ) -> None:
        """在独立事务中记录失败（业务事务已回滚，失败记录必须保留）。"""
        if isinstance(exc, DomainError):
            message = exc.message
        else:
            message = f"{type(exc).__name__}: {exc}"
        try:
            with self.repo.transaction():
                job = self.repo.get_recovery_job(job_id)
                if job is None or job.status != RecoveryJobStatus.PENDING.value:
                    return
                step = OPERATION_STEPS.get(job.operation, job.operation)
                now = self.clock.now_iso()
                if self.repo.note_recovery_failure(
                    job_id, step, message, now, count_retry=True
                ):
                    self._append_attempt(job_id, step, False, message, now)
                    self.audit(
                        actor.user_id, "recovery.retry_failed",
                        detail={
                            "job_id": job_id,
                            "operation": job.operation,
                            "error": message,
                        },
                    )
        except Exception:
            pass  # 失败记录不可掩盖原始错误

    def _append_attempt(
        self, job_id: str, step: str, ok: bool, error: str | None, at: str
    ) -> None:
        self.repo.insert_recovery_attempt(
            RecoveryAttempt(
                attempt_id=self.ids.new_id("att"),
                job_id=job_id,
                step=step,
                ok=ok,
                error=error,
                at=at,
            )
        )

    @staticmethod
    def _job_dict(job: RecoveryJob, *, replayed: bool = False) -> dict:
        return {
            "job_id": job.job_id,
            "operation": job.operation,
            "target_id": job.target_id,
            "payload": dict(job.payload),
            "status": job.status,
            "failed_step": job.failed_step,
            "retry_count": job.retry_count,
            "last_error": job.last_error,
            "created_by": job.created_by,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "completed_at": job.completed_at,
            "replayed": replayed,
        }

    @staticmethod
    def _attempt_dict(attempt: RecoveryAttempt) -> dict:
        return {
            "attempt_id": attempt.attempt_id,
            "job_id": attempt.job_id,
            "step": attempt.step,
            "ok": attempt.ok,
            "error": attempt.error,
            "at": attempt.at,
        }
