"""No-GPU regression suite for the RX 6650 XT v0.5 safety/VRAM protocol."""
from dataclasses import replace
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from memory_backend import (
    MAX_BUDGET_MIB, MemoryBudgetExceeded, MemoryConfig, MemoryUnavailable,
    RetryExhausted, check_allocation_request, estimate_tile_peak_mib,
    evaluate_vram_budget, execute_with_retries, is_oom_error, plan_memory,
    shrink_plan, tile_tasks, valid_config,
)
from verification import evaluate_vram_result


def valid_device():
    return {"hip_version": "7.0", "cuda_version": None, "device_count": 1,
            "available": True, "gpu_name": "Sapphire Radeon RX 6650 XT", "gpu_arch": "gfx1032:sramecc-"}


def valid_payload():
    policy = MemoryConfig()
    plan = plan_memory(policy, free_mib=7000, total_mib=8192)
    return {**valid_device(), "stage": "vram-budget", "ok": True,
            "gpu_compute_confirmed": True, "config": vars(policy),
            "plan": plan.to_dict(), "attempts": [{"status": "PASS"}],
            "oom_recoveries": 0, "safe_oversize_rejected": True,
            "iterations_completed": 20, "gpu_tensor_verified": True,
            "mismatch_count": 0, "max_abs_difference": 1e-5,
            "elapsed_seconds": 2.0, "peak_torch_allocated_mib": 44.0}


class ConfigTests(unittest.TestCase):
    def test_valid_defaults(self):
        self.assertEqual(valid_config(MemoryConfig()), [])

    def test_config_requires_dataclass(self):
        self.assertTrue(valid_config(None))

    def test_budget_upper_limit_exact(self):
        self.assertEqual(valid_config(replace(MemoryConfig(), budget_mib=MAX_BUDGET_MIB)), [])
        self.assertTrue(valid_config(replace(MemoryConfig(), budget_mib=MAX_BUDGET_MIB + 1)))

    def test_small_budget_rejected(self):
        self.assertTrue(valid_config(replace(MemoryConfig(), budget_mib=32)))

    def test_boolean_values_rejected(self):
        self.assertTrue(valid_config(replace(MemoryConfig(), iterations=True)))
        self.assertTrue(valid_config(replace(MemoryConfig(), tile_size=False)))

    def test_nonpower_tile_rejected(self):
        self.assertTrue(valid_config(replace(MemoryConfig(), tile_size=48)))

    def test_microbatch_cannot_exceed_batch(self):
        self.assertTrue(valid_config(replace(MemoryConfig(), microbatch_size=5)))

    def test_tile_cannot_exceed_matrix(self):
        self.assertTrue(valid_config(replace(MemoryConfig(), tile_size=256)))

    def test_iteration_count_below_20_rejected(self):
        self.assertTrue(valid_config(replace(MemoryConfig(), iterations=19)))

    def test_negative_reserve_rejected(self):
        self.assertTrue(valid_config(replace(MemoryConfig(), reserve_mib=-1)))


class PlannerTests(unittest.TestCase):
    def test_normal_target_plan(self):
        config = MemoryConfig()
        plan = plan_memory(config, free_mib=7000, total_mib=8192)
        self.assertEqual(plan.effective_budget_mib, 512)
        self.assertEqual(plan.tile_size, 64)
        self.assertEqual(plan.microbatch_size, 2)
        self.assertLessEqual(plan.estimated_tile_peak_mib, 512)

    def test_auto_microbatch_shrink(self):
        config = MemoryConfig(budget_mib=64, matrix_size=1024, tile_size=512,
                              batch_size=16, microbatch_size=16)
        plan = plan_memory(config, free_mib=7000, total_mib=8192)
        self.assertEqual(plan.microbatch_size, 4)
        self.assertEqual(plan.tile_size, 512)
        self.assertEqual(plan.adjustments, 2)
        self.assertLessEqual(plan.estimated_tile_peak_mib, 64)

    def test_reported_free_memory_clamps_budget(self):
        plan = plan_memory(MemoryConfig(), free_mib=1280, total_mib=8192)
        self.assertEqual(plan.effective_budget_mib, 256)

    def test_low_headroom_returns_blocked(self):
        with self.assertRaises(MemoryUnavailable):
            plan_memory(MemoryConfig(), free_mib=1050, total_mib=8192)

    def test_reject_nonfinite_memory_reading(self):
        for value in [math.inf, math.nan, -1, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                plan_memory(MemoryConfig(), free_mib=value, total_mib=8192)

    def test_reject_free_greater_than_total(self):
        with self.assertRaises(ValueError):
            plan_memory(MemoryConfig(), free_mib=8193, total_mib=8192)

    def test_reject_wrong_gpu_total(self):
        with self.assertRaises(MemoryUnavailable):
            plan_memory(MemoryConfig(), free_mib=2000, total_mib=2048)

    def test_guard_rejects_oversize_without_allocating(self):
        self.assertFalse(check_allocation_request(513, 512))
        self.assertTrue(check_allocation_request(512, 512))
        self.assertFalse(check_allocation_request(math.inf, 512))
        self.assertFalse(check_allocation_request(True, 512))
        self.assertFalse(check_allocation_request(-1, 512))

    def test_shrink_prefers_microbatch_then_tile(self):
        plan = plan_memory(MemoryConfig(), free_mib=7000, total_mib=8192)
        first = shrink_plan(plan)
        self.assertEqual((first.tile_size, first.microbatch_size), (64, 1))
        second = shrink_plan(first)
        self.assertEqual((second.tile_size, second.microbatch_size), (32, 1))

    def test_minimum_plan_cannot_shrink(self):
        plan = plan_memory(MemoryConfig(tile_size=16, microbatch_size=1),
                           free_mib=7000, total_mib=8192)
        self.assertIsNone(shrink_plan(plan))

    def test_estimate_increases_with_working_set(self):
        self.assertLess(estimate_tile_peak_mib(64, 2), estimate_tile_peak_mib(128, 2))
        self.assertLess(estimate_tile_peak_mib(64, 2), estimate_tile_peak_mib(64, 4))


class SchedulerTests(unittest.TestCase):
    def test_output_tiles_cover_all_cells_exactly_once(self):
        batch, dim = 3, 5
        counts = [[[0] * dim for _ in range(dim)] for _ in range(batch)]
        tasks = list(tile_tasks(batch, dim, tile=2, microbatch=2))
        self.assertEqual(len(tasks), 18)
        for s, e, i, i2, j, j2 in tasks:
            for b in range(s, e):
                for row in range(i, i2):
                    for col in range(j, j2):
                        counts[b][row][col] += 1
        self.assertTrue(all(x == 1 for mat in counts for row in mat for x in row))

    def test_edge_tile_cpu_numerical_oracle(self):
        # Uses the exact tile_tasks iterator as the real GPU program.
        dim, batch = 5, 3
        aa = [[[float((b+1)*3+i+j) for j in range(dim)] for i in range(dim)] for b in range(batch)]
        bb = [[[float((b+1)+i*2-j) for j in range(dim)] for i in range(dim)] for b in range(batch)]
        reference = [[[sum(aa[b][i][k] * bb[b][k][j] for k in range(dim))
                       for j in range(dim)] for i in range(dim)] for b in range(batch)]
        for _ in range(20):
            actual = [[[0.] * dim for _ in range(dim)] for _ in range(batch)]
            for s, e, i, i2, j, j2 in tile_tasks(batch, dim, tile=2, microbatch=2):
                for k in range(0, dim, 2):
                    k2 = min(k+2, dim)
                    for b in range(s, e):
                        for row in range(i, i2):
                            for col in range(j, j2):
                                actual[b][row][col] += sum(aa[b][row][x] * bb[b][x][col]
                                                           for x in range(k, k2))
            self.assertEqual(reference, actual)

    def test_oom_recovery_shrinks_microbatch(self):
        start = plan_memory(MemoryConfig(), free_mib=7000, total_mib=8192)
        calls = []
        cleans = []
        def work(plan):
            calls.append(plan.microbatch_size)
            if plan.microbatch_size == 2:
                raise RuntimeError('HIP out of memory')
            return 42
        outcome, plan, attempts = execute_with_retries(start, work, lambda: cleans.append(1))
        self.assertEqual(outcome, 42)
        self.assertEqual(calls, [2, 1])
        self.assertEqual(plan.microbatch_size, 1)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(cleans), 1)

    def test_budget_exceeded_recovers(self):
        start = plan_memory(MemoryConfig(), free_mib=7000, total_mib=8192)
        outcome, plan, attempts = execute_with_retries(
            start, lambda p: 'done' if p.microbatch_size == 1 else (_ for _ in ()).throw(MemoryBudgetExceeded('cap')),
            lambda: None)
        self.assertEqual(outcome, 'done')
        self.assertEqual(plan.microbatch_size, 1)
        self.assertEqual(len(attempts), 2)

    def test_unrelated_errors_never_swallowed(self):
        plan = plan_memory(MemoryConfig(), free_mib=7000, total_mib=8192)
        with self.assertRaisesRegex(RuntimeError, 'kernel missing'):
            execute_with_retries(plan, lambda p: (_ for _ in ()).throw(RuntimeError('kernel missing')),
                                 lambda: self.fail('Should not try cleanup'))

    def test_exhaustion_fails_closed(self):
        plan = plan_memory(MemoryConfig(tile_size=16, microbatch_size=1), free_mib=7000, total_mib=8192)
        with self.assertRaises(RetryExhausted) as caught:
            execute_with_retries(plan, lambda p: (_ for _ in ()).throw(RuntimeError('out of memory')),
                                 lambda: None)
        self.assertEqual(len(caught.exception.attempts), 1)

    def test_non_runtime_oom_message_not_caught(self):
        self.assertFalse(is_oom_error(ValueError('out of memory')))
        self.assertFalse(is_oom_error(RuntimeError('some other failure')))
        self.assertTrue(is_oom_error(RuntimeError('HIP_ERROR_OUT_OF_MEMORY')))


class EvidenceTests(unittest.TestCase):
    def test_valid_evidence(self):
        self.assertEqual(evaluate_vram_budget(valid_payload())[0], 'PASS')
        self.assertEqual(evaluate_vram_result(0, valid_payload())[0], 'PASS')

    def test_cpu_only_rejected(self):
        payload = valid_payload(); payload['hip_version'] = None
        self.assertEqual(evaluate_vram_budget(payload)[0], 'BLOCKED')

    def test_wrong_gpu_rejected(self):
        payload = valid_payload(); payload['gpu_name'] = 'RX 6800 XT'
        self.assertEqual(evaluate_vram_budget(payload)[0], 'BLOCKED')

    def test_mismatched_result_rejected(self):
        payload = valid_payload(); payload['mismatch_count'] = 10
        self.assertEqual(evaluate_vram_budget(payload)[0], 'FAIL')

    def test_partial_iterations_rejected(self):
        payload = valid_payload(); payload['iterations_completed'] = 19
        self.assertEqual(evaluate_vram_budget(payload)[0], 'FAIL')

    def test_false_gpu_compute_rejected(self):
        payload = valid_payload(); payload['gpu_tensor_verified'] = False
        self.assertEqual(evaluate_vram_budget(payload)[0], 'FAIL')

    def test_nan_peak_rejected(self):
        payload = valid_payload(); payload['peak_torch_allocated_mib'] = float('nan')
        self.assertEqual(evaluate_vram_budget(payload)[0], 'FAIL')

    def test_peak_exceeds_budget_rejected(self):
        payload = valid_payload(); payload['peak_torch_allocated_mib'] = 600
        self.assertEqual(evaluate_vram_budget(payload)[0], 'FAIL')

    def test_plan_exceeds_headroom_rejected(self):
        payload = valid_payload(); payload['plan']['effective_budget_mib'] = 6200
        self.assertEqual(evaluate_vram_budget(payload)[0], 'FAIL')

    def test_oversize_guard_missing_rejected(self):
        payload = valid_payload(); payload['safe_oversize_rejected'] = False
        self.assertEqual(evaluate_vram_budget(payload)[0], 'FAIL')

    def test_retry_count_inconsistency_rejected(self):
        payload = valid_payload(); payload['oom_recoveries'] = 7
        self.assertEqual(evaluate_vram_budget(payload)[0], 'FAIL')

    def test_oom_exhaustion_result_fails(self):
        self.assertEqual(evaluate_vram_result(6, {'stage': 'oom-recovery-exhausted'})[0], 'FAIL')

    def test_missing_memory_returns_blocked(self):
        self.assertEqual(evaluate_vram_result(3, {'stage': 'memory-unavailable', 'error': 'low'})[0], 'BLOCKED')

    def test_nonzero_return_code_overrides_valid_evidence(self):
        self.assertEqual(evaluate_vram_result(6, valid_payload())[0], 'FAIL')

    def test_invalid_evidence(self):
        self.assertEqual(evaluate_vram_result(0, None)[0], 'FAIL')


if __name__ == '__main__':
    unittest.main()
