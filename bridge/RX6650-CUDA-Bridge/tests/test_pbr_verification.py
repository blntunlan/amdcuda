import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from verification import evaluate_pbr_cpu_result, evaluate_pbr_gpu_result
from pbr_gpu_backend import compare_cpu_gpu


CPU_REPORT = {'stage':'pbr-cpu', 'ok':True,'roundtrip_verified':True,'orm_verified':True,
              'gpu_compute_confirmed':False, 'map_count':7,'width':32,'height':32,
              'maps':['BaseColor','Normal','Roughness','Metallic','AO','ORM','Height'],
              'normal_max_length_error':.004}
GPU_REPORT = {'stage':'gpu-pbr','ok':True,'gpu_compute_confirmed':True,
              'hip_version':'6.5','cuda_version':None,'device_count':1,'available':True,
              'gpu_name':'AMD Radeon RX 6650 XT', 'gpu_arch':'gfx1032',
              'iterations_completed':3,'gpu_tensor_map_count':7,'roundtrip_verified':True,
              'orm_verified':True,'map_count':7,'mismatch_count':0,'max_channel_delta':1,
              'normal_max_length_error':.005,'peak_torch_allocated_mib':14}


class PBRVerificationTests(unittest.TestCase):
    def test_cpu_sample_pass(self):
        self.assertEqual(evaluate_pbr_cpu_result(0, dict(CPU_REPORT))[0], 'PASS')

    def test_cpu_claims_gpu_rejected(self):
        r = dict(CPU_REPORT, gpu_compute_confirmed=True)
        self.assertEqual(evaluate_pbr_cpu_result(0,r)[0], 'FAIL')

    def test_cpu_incomplete_maps_fail(self):
        r = dict(CPU_REPORT, maps=['BaseColor'])
        self.assertEqual(evaluate_pbr_cpu_result(0,r)[0], 'FAIL')

    def test_cpu_corrupt_normal_fail(self):
        r = dict(CPU_REPORT, normal_max_length_error=float('nan'))
        self.assertEqual(evaluate_pbr_cpu_result(0,r)[0], 'FAIL')

    def test_gpu_good_fixture_passes_shape_validation_only(self):
        # Simulated fixture does NOT imply actual hardware or GPU verification.
        self.assertEqual(evaluate_pbr_gpu_result(0,dict(GPU_REPORT))[0], 'PASS')

    def test_gpu_missing_payload_fails(self):
        self.assertEqual(evaluate_pbr_gpu_result(3,None)[0], 'FAIL')

    def test_gpu_missing_runtime_blocks(self):
        self.assertEqual(evaluate_pbr_gpu_result(3,{'stage':'identity','error':'missing'})[0], 'BLOCKED')

    def test_gpu_wrong_device_blocks(self):
        r = dict(GPU_REPORT, gpu_name='RTX 4090')
        self.assertEqual(evaluate_pbr_gpu_result(0,r)[0], 'BLOCKED')

    def test_gpu_no_hip_blocks(self):
        r = dict(GPU_REPORT, hip_version=None)
        self.assertEqual(evaluate_pbr_gpu_result(0,r)[0], 'BLOCKED')

    def test_gpu_fake_cuda_only_blocks(self):
        r = dict(GPU_REPORT, hip_version=None,cuda_version='12.9')
        self.assertEqual(evaluate_pbr_gpu_result(0,r)[0], 'BLOCKED')

    def test_gpu_cpu_fallback_rejected(self):
        r = dict(GPU_REPORT, gpu_tensor_map_count=0)
        self.assertEqual(evaluate_pbr_gpu_result(0,r)[0], 'FAIL')

    def test_gpu_one_iteration_rejected(self):
        r = dict(GPU_REPORT, iterations_completed=1)
        self.assertEqual(evaluate_pbr_gpu_result(0,r)[0], 'FAIL')

    def test_gpu_numeric_mismatch_rejected(self):
        r = dict(GPU_REPORT, mismatch_count=5)
        self.assertEqual(evaluate_pbr_gpu_result(0,r)[0], 'FAIL')

    def test_gpu_large_error_rejected(self):
        r = dict(GPU_REPORT, max_channel_delta=9)
        self.assertEqual(evaluate_pbr_gpu_result(0,r)[0], 'FAIL')

    def test_gpu_peak_over_budget_rejected(self):
        r = dict(GPU_REPORT, peak_torch_allocated_mib=513)
        self.assertEqual(evaluate_pbr_gpu_result(0,r)[0], 'FAIL')

    def test_gpu_missing_roundtrip_rejected(self):
        r = dict(GPU_REPORT, roundtrip_verified=False)
        self.assertEqual(evaluate_pbr_gpu_result(0,r)[0], 'FAIL')

    def test_gpu_failure_exit_code_rejected(self):
        self.assertEqual(evaluate_pbr_gpu_result(6,dict(GPU_REPORT))[0], 'FAIL')

    def test_compare_cpu_gpu_identical(self):
        self.assertEqual(compare_cpu_gpu({'x':bytes([0,2,5])}, {'x':bytes([0,2,5])})['mismatch_count'],0)

    def test_compare_cpu_gpu_tolerates_small_rounding(self):
        self.assertEqual(compare_cpu_gpu({'x':bytes([1,2,5])}, {'x':bytes([4,3,7])})['mismatch_count'],0)

    def test_compare_cpu_gpu_detects_corruption(self):
        self.assertEqual(compare_cpu_gpu({'x':bytes([1,2,5])}, {'x':bytes([1,3,42])})['mismatch_count'],1)

    def test_compare_cpu_gpu_rejects_shape(self):
        with self.assertRaises(ValueError):
            compare_cpu_gpu({'x':b'a'}, {'x':b'ab'})

    def test_compare_cpu_gpu_rejects_missing(self):
        with self.assertRaises(ValueError):
            compare_cpu_gpu({'x':b'a'}, {'y':b'a'})


if __name__ == '__main__':
    unittest.main()
