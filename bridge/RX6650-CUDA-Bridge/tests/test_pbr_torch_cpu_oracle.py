"""CPU-only torch tensor math is a unit oracle, NEVER a GPU acceptance test."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pbr_backend import generate_cpu
from pbr_gpu_backend import compare_cpu_gpu, generate_tensor_maps


class TorchMathOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('Optional torch CPU oracle not installed')
        cls.torch = torch

    def test_cpu_tensor_path_matches_plain_python_basalt(self):
        torch = self.torch
        groups = generate_tensor_maps(torch, 24, 24, seed=6650, preset='basalt', device='cpu')
        converted = {k: bytes(torch.clamp(torch.round(t*255),0,255).to(torch.uint8).flatten().tolist())
                     for k,t in groups.items()}
        metrics = compare_cpu_gpu(generate_cpu(24, 24, preset='basalt', seed=6650), converted)
        self.assertEqual(metrics['mismatch_count'], 0)

    def test_cpu_tensor_path_matches_plain_python_metal(self):
        torch = self.torch
        groups = generate_tensor_maps(torch, 20, 20, seed=1234, preset='worn_metal', device='cpu', normal_format='opengl')
        converted = {k: bytes(torch.clamp(torch.round(t*255),0,255).to(torch.uint8).flatten().tolist())
                     for k,t in groups.items()}
        metrics = compare_cpu_gpu(generate_cpu(20,20,preset='worn_metal',seed=1234,normal_format='opengl'),converted)
        self.assertEqual(metrics['mismatch_count'], 0)

    def test_cpu_tensor_path_is_not_gpu_evidence(self):
        groups = generate_tensor_maps(self.torch, 8, 8, device='cpu')
        self.assertTrue(all(x.device.type == 'cpu' for x in groups.values()))


if __name__ == '__main__':
    unittest.main()
