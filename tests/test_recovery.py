"""失败恢复作业：失败步骤/重试次数/最后错误落 SQLite，重试只推进一次，
服务重启后失败历史仍可见。"""
import concurrent.futures
import threading
import time
import unittest

from service_09252_006.application.container import ApplicationContext
from service_09252_006.application.ports import (
    FixedClock,
    SystemClock,
    Uuid4IdGenerator,
)
from service_09252_006.domain.enums import Role
from service_09252_006.domain.errors import NotFoundError, ValidationError
from tests.support import Harness, START


class RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h = Harness()
        self.ops = self.h.user("ops", Role.QUALITY_AUTHORITY, institution_id=None)

    def tearDown(self) -> None:
        self.h.close()

    def _record(
        self,
        step: str = "package.seal",
        error: str = "boom",
        payload: dict | None = None,
    ) -> dict:
        return self.h.ctx.recovery.record_failure(
            self.ops, step=step, error=error, payload=payload
        )

    def test_record_failure_persists_step_error_and_zero_retries(self) -> None:
        job = self._record(error="数据库锁定")
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["retry_count"], 0)
        self.assertEqual(job["last_error"], "数据库锁定")
        self.assertIsNone(job["resolved_at"])
        fetched = self.h.ctx.recovery.get_job(self.ops, job["job_id"])
        self.assertEqual(fetched["step"], "package.seal")
        self.assertEqual(fetched["last_error"], "数据库锁定")
        self.assertEqual(fetched["payload"], {})

    def test_record_failure_validates_input(self) -> None:
        with self.assertRaises(ValidationError):
            self.h.ctx.recovery.record_failure(self.ops, step="  ", error="x")
        with self.assertRaises(ValidationError):
            self.h.ctx.recovery.record_failure(self.ops, step="s", error=" ")

    def test_failed_retry_increments_count_and_records_last_error(self) -> None:
        job = self._record(error="首次失败")

        def handler(payload: dict) -> None:
            raise RuntimeError("仍然失败")

        self.h.ctx.recovery.register_step("package.seal", handler)
        first = self.h.ctx.recovery.retry(self.ops, job_id=job["job_id"])
        self.assertEqual(first["status"], "failed")
        self.assertEqual(first["retry_count"], 1)
        self.assertEqual(first["last_error"], "仍然失败")
        second = self.h.ctx.recovery.retry(self.ops, job_id=job["job_id"])
        self.assertEqual(second["status"], "failed")
        self.assertEqual(second["retry_count"], 2)
        self.assertEqual(second["last_error"], "仍然失败")
        self.assertIsNone(second["resolved_at"])

    def test_successful_retry_advances_exactly_once(self) -> None:
        job = self._record(payload={"package_id": "pkg-1"})
        attempts: list[dict] = []
        self.h.ctx.recovery.register_step(
            "package.seal", lambda payload: attempts.append(payload)
        )
        done = self.h.ctx.recovery.retry(self.ops, job_id=job["job_id"])
        self.assertEqual(done["status"], "succeeded")
        self.assertEqual(done["retry_count"], 1)
        self.assertIsNotNone(done["resolved_at"])
        resolved_at = done["resolved_at"]
        # 再次重试：只回放，状态不再推进，处理器不再执行
        again = self.h.ctx.recovery.retry(self.ops, job_id=job["job_id"])
        self.assertTrue(again["replayed"])
        self.assertEqual(again["status"], "succeeded")
        self.assertEqual(again["retry_count"], 1)
        self.assertEqual(again["resolved_at"], resolved_at)
        self.assertEqual(attempts, [{"package_id": "pkg-1"}])

    def test_flaky_step_recovers_after_failures(self) -> None:
        job = self._record(error="首次失败")
        outcomes: list[Exception | None] = [
            RuntimeError("网络抖动"),
            RuntimeError("超时"),
            None,
        ]

        def handler(payload: dict) -> None:
            outcome = outcomes.pop(0)
            if outcome is not None:
                raise outcome

        self.h.ctx.recovery.register_step("package.seal", handler)
        r1 = self.h.ctx.recovery.retry(self.ops, job_id=job["job_id"])
        self.assertEqual((r1["status"], r1["retry_count"]), ("failed", 1))
        self.assertEqual(r1["last_error"], "网络抖动")
        r2 = self.h.ctx.recovery.retry(self.ops, job_id=job["job_id"])
        self.assertEqual((r2["status"], r2["retry_count"]), ("failed", 2))
        self.assertEqual(r2["last_error"], "超时")
        r3 = self.h.ctx.recovery.retry(self.ops, job_id=job["job_id"])
        self.assertEqual((r3["status"], r3["retry_count"]), ("succeeded", 3))
        # 最后一次错误保留为历史
        self.assertEqual(r3["last_error"], "超时")

    def test_retry_unknown_job_and_unregistered_step(self) -> None:
        with self.assertRaises(NotFoundError):
            self.h.ctx.recovery.retry(self.ops, job_id="job_404")
        job = self._record(step="no.handler")
        with self.assertRaises(ValidationError):
            self.h.ctx.recovery.retry(self.ops, job_id=job["job_id"])

    def test_restart_preserves_failure_history(self) -> None:
        failed = self._record(step="package.seal", error="封存失败")
        recovered = self._record(step="decision.issue", error="签发失败")
        self.h.ctx.recovery.register_step("decision.issue", lambda payload: None)
        self.h.ctx.recovery.retry(self.ops, job_id=recovered["job_id"])

        # 模拟服务重启：关闭上下文，用同一数据库文件重新打开
        self.h.ctx.close()
        reopened = ApplicationContext(
            self.h.db_path, clock=FixedClock(START), ids=Uuid4IdGenerator()
        )
        try:
            ops = reopened.repo.get_user("ops")
            jobs = {j["job_id"]: j for j in reopened.recovery.list_jobs(ops)}
            self.assertEqual(set(jobs), {failed["job_id"], recovered["job_id"]})

            # 失败历史：步骤、错误、重试次数都在
            still_failed = jobs[failed["job_id"]]
            self.assertEqual(still_failed["step"], "package.seal")
            self.assertEqual(still_failed["status"], "failed")
            self.assertEqual(still_failed["last_error"], "封存失败")
            self.assertEqual(still_failed["retry_count"], 0)

            done = jobs[recovered["job_id"]]
            self.assertEqual(done["status"], "succeeded")
            self.assertEqual(done["retry_count"], 1)
            self.assertEqual(done["last_error"], "签发失败")
            self.assertIsNotNone(done["resolved_at"])

            # 重启后已恢复的作业依然只回放，不再推进
            replay = reopened.recovery.retry(ops, job_id=recovered["job_id"])
            self.assertTrue(replay["replayed"])
            self.assertEqual(replay["retry_count"], 1)
        finally:
            reopened.close()

    def test_concurrent_retries_advance_once(self) -> None:
        job = self._record()
        attempts: list[int] = []
        lock = threading.Lock()

        def handler(payload: dict) -> None:
            time.sleep(0.05)  # 放大并发窗口
            with lock:
                attempts.append(1)

        def worker() -> dict:
            # 每个 worker 独立连接，等价于不同进程
            ctx = ApplicationContext(
                self.h.db_path, clock=SystemClock(), ids=Uuid4IdGenerator()
            )
            try:
                ctx.recovery.register_step("package.seal", handler)
                ops = ctx.repo.get_user("ops")
                return ctx.recovery.retry(ops, job_id=job["job_id"])
            finally:
                ctx.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: worker(), range(2)))

        # 状态只推进一次：重试次数为 1，处理器只执行一次
        final = self.h.ctx.recovery.get_job(self.ops, job["job_id"])
        self.assertEqual(final["status"], "succeeded")
        self.assertEqual(final["retry_count"], 1)
        self.assertEqual(len(attempts), 1)
        # 两个调用都看到成功终态；恰好一个真实推进，一个回放
        self.assertEqual({r["status"] for r in results}, {"succeeded"})
        self.assertEqual([r["replayed"] for r in results].count(False), 1)
        self.assertEqual([r["replayed"] for r in results].count(True), 1)


if __name__ == "__main__":
    unittest.main()
