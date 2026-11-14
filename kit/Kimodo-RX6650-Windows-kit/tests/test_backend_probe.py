import importlib.util
import pathlib
import unittest

path = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'backend_probe.py'
spec = importlib.util.spec_from_file_location('backend_probe', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class BackendChecks(unittest.TestCase):
    def test_correct_gpu(self):
        self.assertEqual(m.classify_backend('AMD Radeon RX 6650 XT', '7.0', None, True)[0], 'READY')

    def test_variants(self):
        self.assertEqual(m.classify_backend('AMD Radeon RX 6650XT', '7.0', None, True)[0], 'READY')

    def test_wrong_model(self):
        self.assertEqual(m.classify_backend('AMD Radeon RX 6600 XT', '7.0', None, True)[0], 'BLOCKED')

    def test_wrong_vendor(self):
        self.assertEqual(m.classify_backend('NVIDIA RTX 4090', None, '12.8', True)[0], 'BLOCKED')

    def test_cpu_only(self):
        self.assertEqual(m.classify_backend('', None, None, False)[0], 'BLOCKED')

    def test_non_rocm_6650(self):
        self.assertEqual(m.classify_backend('AMD Radeon RX 6650 XT', None, None, True)[0], 'BLOCKED')

    def test_case_insensitive(self):
        self.assertEqual(m.classify_backend('amd radeon rx 6650 xt', '1', None, True)[0], 'READY')

    def test_ram_warning(self):
        self.assertIn('may exceed', m.assess_free_memory(16))

    def test_high_ram(self):
        self.assertIn('does not guarantee', m.assess_free_memory(32))

    def test_mock_no_torch(self):
        # No import-time torch dependency (doctor works before torch installation).
        self.assertTrue(callable(m.probe))


if __name__ == '__main__':
    unittest.main()
