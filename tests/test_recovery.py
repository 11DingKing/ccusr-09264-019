"""失败恢复作业：失败队列登记、重试只推进一次、重启不丢作业。

- 作业记录失败步骤、重试次数与最后错误，全部写入 SQLite；
- 重试成功后业务状态与作业状态各推进一次，重复重试只是回放；
- 重试失败时业务回滚，但失败历史（步骤/次数/错误）保留；
- 关掉上下文（模拟服务重启）后，新连接仍能看到队列与失败历史。
"""
import unittest
from types import SimpleNamespace

from service_09252_006.application.container import ApplicationContext
from service_09252_006.application.ports import Uuid4IdGenerator
from service_09252_006.application.recovery_service import (
    OP_DECISION_ISSUE,
    OP_PACKAGE_SEAL,
)
from service_09252_006.domain.enums import Role
from service_09252_006.domain.errors import (
    ConflictError,
    PermissionDeniedError,
    ValidationError,
)
from tests.flow import complete_review, seal_new_package, upload_material
from tests.support import Harness


class RecoveryJobTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness()
        self.admin = self.h.user("admin-a", Role.INSTITUTION_ADMIN)
        self.authority = self.h.user(
            "auth", Role.QUALITY_AUTHORITY, institution_id=None
        )
        self.auditor = self.h.user("aud", Role.AUDITOR, institution_id=None)
        self.reviewer = self.h.user("rev-1", Role.REVIEWER, institution_id="inst-ext")

    def tearDown(self) -> None:
        self.h.close()

    def _draft_package(self) -> str:
        uploaded = upload_material(self.h, self.admin, data="大纲 v1".encode("utf-8"))
        pkg = self.h.ctx.packages.create_package(self.admin, title="待封存包")
        pid = pkg["package_id"]
        self.h.ctx.packages.add_entry(
            self.admin, package_id=pid, version_id=uploaded.version["version_id"]
        )
        return pid

    # ------------------------------------------------------------- 登记
    def test_record_failure_persists_job_fields(self) -> None:
        pid = self._draft_package()
        job = self.h.ctx.recovery.record_failure(
            self.authority,
            operation=OP_PACKAGE_SEAL,
            target_id=pid,
            failed_step="seal_package",
            error="sqlite3.OperationalError: database is locked",
        )
        self.assertEqual(job["status"], "pending")
        self.assertEqual(job["failed_step"], "seal_package")
        self.assertEqual(job["retry_count"], 0)
        self.assertEqual(
            job["last_error"], "sqlite3.OperationalError: database is locked"
        )
        self.assertFalse(job["replayed"])

        queue = self.h.ctx.recovery.list_jobs(self.authority, status="pending")
        self.assertEqual([j["job_id"] for j in queue], [job["job_id"]])

        detail = self.h.ctx.recovery.get_job(self.auditor, job["job_id"])
        self.assertEqual(len(detail["attempts"]), 1)
        self.assertFalse(detail["attempts"][0]["ok"])
        self.assertEqual(detail["attempts"][0]["step"], "seal_package")
        self.assertIn("database is locked", detail["attempts"][0]["error"])

    def test_record_failure_dedups_pending_job(self) -> None:
        pid = self._draft_package()
        first = self.h.ctx.recovery.record_failure(
            self.authority,
            operation=OP_PACKAGE_SEAL,
            target_id=pid,
            failed_step="seal_package",
            error="第一次失败",
        )
        second = self.h.ctx.recovery.record_failure(
            self.authority,
            operation=OP_PACKAGE_SEAL,
            target_id=pid,
            failed_step="seal_package",
            error="第二次失败",
        )
        self.assertEqual(first["job_id"], second["job_id"])
        self.assertTrue(second["replayed"])
        # 同一待恢复作业只保留一单，最后错误被刷新，重试次数不受影响
        self.assertEqual(len(self.h.ctx.recovery.list_jobs(self.authority)), 1)
        detail = self.h.ctx.recovery.get_job(self.authority, first["job_id"])
        self.assertEqual(detail["last_error"], "第二次失败")
        self.assertEqual(detail["retry_count"], 0)
        self.assertEqual(len(detail["attempts"]), 2)

    def test_record_failure_validates_input(self) -> None:
        with self.assertRaises(ValidationError):
            self.h.ctx.recovery.record_failure(
                self.authority,
                operation="unknown.op",
                target_id="pkg_x",
                failed_step="step",
                error="err",
            )
        with self.assertRaises(ValidationError):
            self.h.ctx.recovery.record_failure(
                self.authority,
                operation=OP_PACKAGE_SEAL,
                target_id="pkg_x",
                failed_step="",
                error="err",
            )

    # ------------------------------------------------------------- 重试
    def test_retry_success_advances_state_only_once(self) -> None:
        pid = self._draft_package()
        job = self.h.ctx.recovery.record_failure(
            self.authority,
            operation=OP_PACKAGE_SEAL,
            target_id=pid,
            failed_step="seal_package",
            error="database is locked",
        )

        result = self.h.ctx.recovery.retry(self.authority, job_id=job["job_id"])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(result["outcome"]["status"], "sealed")
        sealed = self.h.repo.get_package(pid)
        self.assertEqual(sealed.status, "sealed")
        self.assertIsNotNone(sealed.manifest_fingerprint)

        # 再次重试：回放，状态不重复推进，重试次数不变
        again = self.h.ctx.recovery.retry(self.authority, job_id=job["job_id"])
        self.assertTrue(again["replayed"])
        self.assertEqual(again["retry_count"], 1)
        still = self.h.repo.get_package(pid)
        self.assertEqual(still.status, "sealed")
        self.assertEqual(still.sealed_at, sealed.sealed_at)
        self.assertEqual(still.manifest_fingerprint, sealed.manifest_fingerprint)

        detail = self.h.ctx.recovery.get_job(self.authority, job["job_id"])
        successes = [a for a in detail["attempts"] if a["ok"]]
        self.assertEqual(len(successes), 1)

    def test_retry_failure_records_error_then_recovers(self) -> None:
        sealed = seal_new_package(self.h, self.admin)
        pid = sealed.package_id
        job = self.h.ctx.recovery.record_failure(
            self.authority,
            operation=OP_DECISION_ISSUE,
            target_id=pid,
            failed_step="issue_decision",
            error="sqlite3.OperationalError: database is locked",
            payload={"decision": "approved", "note": "补签发"},
        )

        # 条件未满足（尚无评审完成），重试失败：业务回滚，失败历史保留
        with self.assertRaises(ConflictError):
            self.h.ctx.recovery.retry(self.authority, job_id=job["job_id"])
        job_view = self.h.ctx.recovery.get_job(self.authority, job["job_id"])
        self.assertEqual(job_view["status"], "pending")
        self.assertEqual(job_view["retry_count"], 1)
        self.assertEqual(job_view["failed_step"], "issue_decision")
        self.assertIn("尚无评审人完成评审", job_view["last_error"])
        failed_attempts = [a for a in job_view["attempts"] if not a["ok"]]
        self.assertEqual(len(failed_attempts), 2)  # 初始失败 + 一次失败重试
        self.assertIn("尚无评审人完成评审", failed_attempts[-1]["error"])
        # 业务状态未被失败的重试改动
        self.assertEqual(self.h.repo.get_package(pid).status, "sealed")

        # 补上评审后重试成功，状态只推进一次
        complete_review(self.h, self.authority, self.reviewer, pid)
        result = self.h.ctx.recovery.retry(self.authority, job_id=job["job_id"])
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["retry_count"], 2)
        decided = self.h.repo.get_package(pid)
        self.assertEqual(decided.status, "decided")
        self.assertEqual(decided.decision, "approved")
        self.assertEqual(decided.decision_note, "补签发")

        again = self.h.ctx.recovery.retry(self.authority, job_id=job["job_id"])
        self.assertTrue(again["replayed"])
        self.assertEqual(again["retry_count"], 2)
        self.assertEqual(self.h.repo.get_package(pid).decided_at, decided.decided_at)

    def test_retry_unknown_job(self) -> None:
        from service_09252_006.domain.errors import NotFoundError

        with self.assertRaises(NotFoundError):
            self.h.ctx.recovery.retry(self.authority, job_id="job_missing")

    # --------------------------------------------------------- 重启恢复
    def test_restart_preserves_failure_history(self) -> None:
        sealed = seal_new_package(self.h, self.admin)
        pid = sealed.package_id
        job = self.h.ctx.recovery.record_failure(
            self.authority,
            operation=OP_DECISION_ISSUE,
            target_id=pid,
            failed_step="issue_decision",
            error="sqlite3.OperationalError: database is locked",
            payload={"decision": "approved"},
        )
        with self.assertRaises(ConflictError):
            self.h.ctx.recovery.retry(self.authority, job_id=job["job_id"])

        # 模拟服务重启：关闭上下文，用同一数据库文件开新连接
        self.h.ctx.close()
        with ApplicationContext(
            self.h.db_path, clock=self.h.clock, ids=Uuid4IdGenerator()
        ) as ctx2:
            queue = ctx2.recovery.list_jobs(self.authority, status="pending")
            self.assertEqual(len(queue), 1)
            restored = queue[0]
            self.assertEqual(restored["job_id"], job["job_id"])
            self.assertEqual(restored["failed_step"], "issue_decision")
            self.assertEqual(restored["retry_count"], 1)
            self.assertIn("尚无评审人完成评审", restored["last_error"])

            detail = ctx2.recovery.get_job(self.auditor, job["job_id"])
            self.assertEqual(len(detail["attempts"]), 2)
            self.assertTrue(all(not a["ok"] for a in detail["attempts"]))
            self.assertEqual(detail["attempts"][0]["step"], "issue_decision")
            self.assertIn("database is locked", detail["attempts"][0]["error"])
            self.assertIn("尚无评审人完成评审", detail["attempts"][1]["error"])

            # 重启后作业仍可重试：补齐评审即可恢复
            complete_review(SimpleNamespace(ctx=ctx2), self.authority, self.reviewer, pid)
            result = ctx2.recovery.retry(self.authority, job_id=job["job_id"])
            self.assertEqual(result["status"], "succeeded")
            self.assertEqual(ctx2.repo.get_package(pid).status, "decided")

    # ------------------------------------------------------------- 权限
    def test_permissions(self) -> None:
        pid = self._draft_package()
        with self.assertRaises(PermissionDeniedError):
            self.h.ctx.recovery.record_failure(
                self.reviewer,
                operation=OP_PACKAGE_SEAL,
                target_id=pid,
                failed_step="seal_package",
                error="err",
            )
        job = self.h.ctx.recovery.record_failure(
            self.authority,
            operation=OP_PACKAGE_SEAL,
            target_id=pid,
            failed_step="seal_package",
            error="err",
        )
        with self.assertRaises(PermissionDeniedError):
            self.h.ctx.recovery.retry(self.admin, job_id=job["job_id"])
        with self.assertRaises(PermissionDeniedError):
            self.h.ctx.recovery.retry(self.auditor, job_id=job["job_id"])
        with self.assertRaises(PermissionDeniedError):
            self.h.ctx.recovery.list_jobs(self.reviewer)
        # 审计只读
        self.assertEqual(len(self.h.ctx.recovery.list_jobs(self.auditor)), 1)


if __name__ == "__main__":
    unittest.main()
