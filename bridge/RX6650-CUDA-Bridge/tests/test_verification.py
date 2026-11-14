"""Offline and platform-independent unit tests. Run on each change."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verification import evaluate_diagnostic, evaluate_smoke_result, overall_status, validate_profile
from test_native_backend import valid_math


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "config" / "gfx1032.json").read_text(encoding="utf-8"))

    def test_real_profile_strict_and_safe(self):
        self.assertEqual(validate_profile(self.data), [])

    def test_rejects_wrong_arch(self):
        self.data["llvm_target"] = "gfx1030"
        self.assertIn("Unexpected LLVM target", validate_profile(self.data))

    def test_rejects_wrong_card(self):
        self.data["device_allowlist"] = ["RX 6650 XT", "RX 6800"]
        self.assertTrue(validate_profile(self.data))

    def test_rejects_unsafe_binary_changes(self):
        self.data["safety"]["no_global_cuda_dll_overrides"] = False
        self.assertTrue(validate_profile(self.data))


class DiagnosticTests(unittest.TestCase):
    def test_accept_exact_target(self):
        status, _ = evaluate_diagnostic({"target_detected": True,
                                         "gpus": [{"name": "AMD Radeon RX 6650 XT"}]})
        self.assertEqual(status, "PASS")

    def test_accept_vendor_specific_gpu_name(self):
        status, _ = evaluate_diagnostic({"target_detected": True,
                                         "gpus": [{"name": "Sapphire Radeon RX6650XT"}]})
        self.assertEqual(status, "PASS")

    def test_reject_other_gpu(self):
        status, _ = evaluate_diagnostic({"target_detected": False,
                                         "gpus": [{"name": "AMD Radeon RX 6600 XT"}]})
        self.assertEqual(status, "BLOCKED")

    def test_blocks_secondary_nvidia(self):
        status, _ = evaluate_diagnostic({"target_detected": True,
                                         "gpus": [{"name": "AMD Radeon RX 6650 XT"},
                                                  {"name": "NVIDIA GeForce RTX 4090"}]})
        self.assertEqual(status, "BLOCKED")

    def test_rejects_bad_flag(self):
        status, _ = evaluate_diagnostic({"target_detected": False,
                                         "gpus": [{"name": "AMD Radeon RX 6650 XT"}]})
        self.assertEqual(status, "FAIL")

    def test_multiple_targets_require_explicit_selection(self):
        status, _ = evaluate_diagnostic({"target_detected": True,
                                         "gpus": [{"name": "AMD Radeon RX 6650 XT"},
                                                  {"name": "AMD Radeon RX 6650 XT"}]})
        self.assertEqual(status, "FAIL")


class HardwareClassificationTests(unittest.TestCase):
    def test_no_runtime_is_blocked(self):
        self.assertEqual(evaluate_smoke_result(3, {"stage": "device", "ok": False})[0], "BLOCKED")

    def test_false_numeric_result_is_failed(self):
        self.assertEqual(evaluate_smoke_result(5, {**valid_math(), "ok": False})[0], "FAIL")

    def test_true_numeric_result_passes(self):
        payload = {**valid_math(), "max_abs_difference": 0.00014}
        self.assertEqual(evaluate_smoke_result(0, payload)[0], "PASS")

    def test_missing_error_value_does_not_pass(self):
        payload = {**valid_math(), "max_abs_difference": None}
        self.assertEqual(evaluate_smoke_result(0, payload)[0], "FAIL")

    def test_bad_output_is_failed(self):
        self.assertEqual(evaluate_smoke_result(0, None)[0], "FAIL")

    def test_overall_blocked_not_pass(self):
        self.assertEqual(overall_status([{"status": "PASS"}, {"status": "BLOCKED"}]), "BLOCKED")

    def test_overall_fail_higher_priority(self):
        self.assertEqual(overall_status([{"status": "BLOCKED"}, {"status": "FAIL"}]), "FAIL")

    def test_overall_pass(self):
        self.assertEqual(overall_status([{"status": "PASS"}, {"status": "PASS"}]), "PASS")


if __name__ == "__main__":
    unittest.main()
