import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from primitives_backend import (REQUIRED_OPERATIONS, check_parameters,
                                evaluate_primitives, validate_operation)
from verification import evaluate_primitive_result


def info():
    return {"hip_version": "7.13", "cuda_version": None, "device_count": 1,
            "available": True, "gpu_name": "Sapphire Radeon RX 6650 XT", "gpu_arch": "gfx1032:sramecc-"}


def valid_operation(name):
    op = {"name": name, "status": "PASS", "correct": True,
          "gpu_tensor_verified": True, "iterations": 20 if name in ("vector_add", "matmul") else 1,
          "elapsed_seconds": 0.05}
    if name in ("vector_add", "matmul"):
        op.update(max_abs_difference=0.0, mismatch_count=0)
    return op


def payload():
    return {**info(), "stage": "primitives", "ok": True, "gpu_compute_confirmed": True,
            "operations": [valid_operation(name) for name in REQUIRED_OPERATIONS]}


class ParametersTests(unittest.TestCase):
    def test_defaults_allowed(self):
        self.assertEqual(check_parameters(20, 262144, 256, 4), [])

    def test_too_few_iterations_rejected(self):
        self.assertIn("iterations", ' '.join(check_parameters(19, 262144, 256, 4)))

    def test_too_large_allocations_rejected(self):
        self.assertEqual(len(check_parameters(20, 2_097_153, 1025, 33)), 3)

    def test_boolean_iterations_not_accepted(self):
        self.assertNotEqual(check_parameters(True, 262144, 256, 4), [])

    def test_negative_sizes_rejected(self):
        self.assertEqual(len(check_parameters(20, -1, -1, -1)), 3)


class OperationsTests(unittest.TestCase):
    def test_valid_suite(self):
        self.assertEqual(evaluate_primitives(payload())[0], "PASS")

    def test_missing_suite(self):
        self.assertEqual(evaluate_primitives(None)[0], "FAIL")

    def test_missing_operation(self):
        p = payload(); p["operations"].pop()
        self.assertEqual(evaluate_primitives(p)[0], "FAIL")

    def test_duplicate_operation(self):
        p = payload(); p["operations"][1] = valid_operation("allocation")
        self.assertEqual(evaluate_primitives(p)[0], "FAIL")

    def test_incorrect_order(self):
        p = payload(); p["operations"].reverse()
        self.assertEqual(evaluate_primitives(p)[0], "FAIL")

    def test_cpu_fallback_rejected(self):
        p = payload(); p["operations"][3]["gpu_tensor_verified"] = False
        self.assertEqual(evaluate_primitives(p)[0], "FAIL")

    def test_matmul_mismatch_rejected(self):
        p = payload(); p["operations"][4]["mismatch_count"] = 1
        self.assertEqual(evaluate_primitives(p)[0], "FAIL")

    def test_nan_error_rejected(self):
        p = payload(); p["operations"][4]["max_abs_difference"] = float('nan')
        self.assertEqual(evaluate_primitives(p)[0], "FAIL")

    def test_infinity_runtime_rejected(self):
        p = payload(); p["operations"][3]["elapsed_seconds"] = float('inf')
        self.assertEqual(evaluate_primitives(p)[0], "FAIL")

    def test_incomplete_iterations_rejected(self):
        p = payload(); p["operations"][4]["iterations"] = 5
        self.assertEqual(evaluate_primitives(p)[0], "FAIL")

    def test_wrong_target_rejected(self):
        p = payload(); p["gpu_name"] = "AMD Radeon RX 6600"
        self.assertEqual(evaluate_primitives(p)[0], "BLOCKED")

    def test_cpu_pytorch_rejected(self):
        p = payload(); p["hip_version"] = None
        self.assertEqual(evaluate_primitives(p)[0], "BLOCKED")

    def test_wrong_arch_rejected(self):
        p = payload(); p["gpu_arch"] = "gfx1100"
        self.assertEqual(evaluate_primitives(p)[0], "BLOCKED")

    def test_unsuccessful_suite_rejected(self):
        p = payload(); p["ok"] = False
        self.assertEqual(evaluate_primitives(p)[0], "FAIL")

    def test_missing_compute_confirmation_rejected(self):
        p = payload(); p["gpu_compute_confirmed"] = False
        self.assertEqual(evaluate_primitives(p)[0], "FAIL")

    def test_failed_transfer_rejected(self):
        p = payload(); p["operations"][2]["correct"] = False
        self.assertEqual(evaluate_primitives(p)[0], "FAIL")

    def test_operation_non_dict_fails(self):
        self.assertNotEqual(validate_operation("gpu", "matmul"), [])

    def test_negative_elapsed_rejected(self):
        p = payload(); p["operations"][0]["elapsed_seconds"] = -1
        self.assertEqual(evaluate_primitives(p)[0], "FAIL")


class ResultClassificationTests(unittest.TestCase):
    def test_valid_suite_and_exit_code(self):
        self.assertEqual(evaluate_primitive_result(0, payload())[0], 'PASS')

    def test_nonzero_exit_overrides_valid_payload(self):
        self.assertEqual(evaluate_primitive_result(5, payload())[0], 'FAIL')

    def test_import_failure_blocked(self):
        self.assertEqual(evaluate_primitive_result(3, {"stage": "import", "error": "no torch"})[0], 'BLOCKED')

    def test_invalid_json_rejected(self):
        self.assertEqual(evaluate_primitive_result(0, None)[0], 'FAIL')

    def test_parameters_error_rejected(self):
        self.assertEqual(evaluate_primitive_result(1, {"stage": "parameters", "error": "bad"})[0], 'FAIL')

    def test_execution_error_rejected(self):
        self.assertEqual(evaluate_primitive_result(6, {"stage": "execution", "error": "OOM"})[0], 'FAIL')


if __name__ == '__main__':
    unittest.main()
