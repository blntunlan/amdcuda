import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from native_backend import check_native_device, evaluate_native_math, numeric_accuracy
from verification import evaluate_smoke_result


def valid_info():
    return {
        'hip_version': '10.0', 'cuda_version': None, 'device_count': 1,
        'available': True, 'gpu_name': 'AMD Radeon RX 6650 XT', 'gpu_arch': 'gfx1032:sramecc-'
    }


def valid_math():
    return {
        **valid_info(), 'stage': 'matmul', 'ok': True, 'gpu_compute_confirmed': True,
        'iterations': 20, 'max_abs_difference': 0.00001, 'gpu_kernel_seconds': 0.055
    }


class RuntimeIdentityTests(unittest.TestCase):
    def test_accepts_gfx1032_sapphire(self):
        self.assertEqual(check_native_device(valid_info())[0], 'PASS')

    def test_rejects_cpu_build(self):
        obj = valid_info(); obj['hip_version'] = None
        self.assertEqual(check_native_device(obj)[0], 'BLOCKED')

    def test_rejects_cuda_only_build(self):
        obj = valid_info(); obj.update(hip_version='', cuda_version='12.8')
        self.assertEqual(check_native_device(obj)[0], 'BLOCKED')

    def test_rejects_same_arch_wrong_card(self):
        obj = valid_info(); obj['gpu_name'] = 'AMD Radeon RX 6600 XT'
        self.assertEqual(check_native_device(obj)[0], 'BLOCKED')

    def test_rejects_wrong_arch(self):
        obj = valid_info(); obj['gpu_arch'] = 'gfx1100'
        self.assertEqual(check_native_device(obj)[0], 'BLOCKED')

    def test_rejects_multiple_gpus(self):
        obj = valid_info(); obj['device_count'] = 2
        self.assertEqual(check_native_device(obj)[0], 'BLOCKED')

    def test_rejects_missing_gpu(self):
        obj = valid_info(); obj['device_count'] = 0
        self.assertEqual(check_native_device(obj)[0], 'BLOCKED')

    def test_rejects_unavailable_gpu(self):
        obj = valid_info(); obj['available'] = False
        self.assertEqual(check_native_device(obj)[0], 'BLOCKED')

    def test_rejects_nonnumeric_count(self):
        obj = valid_info(); obj['device_count'] = True
        self.assertEqual(check_native_device(obj)[0], 'BLOCKED')

    def test_rejects_wrong_object(self):
        self.assertEqual(check_native_device(None)[0], 'FAIL')


class MathEvidenceTests(unittest.TestCase):
    def test_accepts_reproducible_math(self):
        self.assertEqual(evaluate_native_math(valid_math())[0], 'PASS')

    def test_rejects_legacy_smoke_payload(self):
        self.assertEqual(evaluate_smoke_result(0, {'stage': 'matmul', 'ok': True,
                                                   'max_abs_difference': 0.0001})[0], 'BLOCKED')

    def test_rejects_unsupported_cpu_fallback(self):
        obj = valid_math(); obj['gpu_compute_confirmed'] = False
        self.assertEqual(evaluate_native_math(obj)[0], 'FAIL')

    def test_rejects_incorrect_math(self):
        obj = valid_math(); obj['ok'] = False
        self.assertEqual(evaluate_native_math(obj)[0], 'FAIL')

    def test_rejects_insufficient_iterations(self):
        obj = valid_math(); obj['iterations'] = 2
        self.assertEqual(evaluate_native_math(obj)[0], 'FAIL')

    def test_rejects_nan_error(self):
        obj = valid_math(); obj['max_abs_difference'] = float('nan')
        self.assertEqual(evaluate_native_math(obj)[0], 'FAIL')

    def test_rejects_missing_elapsed(self):
        obj = valid_math(); del obj['gpu_kernel_seconds']
        self.assertEqual(evaluate_native_math(obj)[0], 'FAIL')

    def test_failure_exit_overrides_valid_payload(self):
        self.assertEqual(evaluate_smoke_result(6, valid_math())[0], 'FAIL')

    def test_import_failure_blocked(self):
        self.assertEqual(evaluate_smoke_result(3, {'stage': 'import', 'error': 'No module torch'})[0], 'BLOCKED')

    def test_missing_gpu_payload_fail(self):
        self.assertEqual(evaluate_smoke_result(0, None)[0], 'FAIL')

    def test_valid_matrices_oracle(self):
        self.assertTrue(numeric_accuracy([1.001, -1.0], reference=[1.0, -1.0]))

    def test_nonnumeric_matrix_oracle(self):
        self.assertFalse(numeric_accuracy([float('inf')], reference=[1.0]))

    def test_wrong_shape_oracle(self):
        self.assertFalse(numeric_accuracy([1.0], reference=[1.0, 2.0]))


if __name__ == '__main__':
    unittest.main()
