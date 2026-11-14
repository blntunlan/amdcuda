import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def test_runner_help(self):
        proc = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'run-tests.py'), '--help'],
                              capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('--tier', proc.stdout)

    def test_primitives_cli_help(self):
        proc = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'gpu-primitives.py'), '--help'],
                              capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('--vector-size', proc.stdout)

    def test_primitives_invalid_parameters_do_not_import_torch(self):
        import json
        proc = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'gpu-primitives.py'),
                               '--iterations', '2'], capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(json.loads(proc.stdout)['stage'], 'parameters')

    def test_primitives_source_compiles(self):
        import ast
        ast.parse((ROOT / 'scripts' / 'gpu-primitives.py').read_text(encoding='utf-8'))

    def test_python_smoke_source_compiles(self):
        import ast
        path = ROOT / 'scripts' / 'smoke-test-torch.py'
        ast.parse(path.read_text(encoding='utf-8'))

    def test_vram_cli_help(self):
        proc = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'gpu-vram-budget.py'), '--help'],
                              capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, 0)
        self.assertIn('--budget-mib', proc.stdout)
        self.assertIn('--microbatch-size', proc.stdout)

    def test_vram_rejects_oversize_before_importing_torch(self):
        import json
        proc = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'gpu-vram-budget.py'),
                               '--budget-mib', '6145'], capture_output=True, text=True, timeout=15)
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(json.loads(proc.stdout)['stage'], 'parameters')

    def test_vram_source_compiles(self):
        import ast
        for filename in ('gpu-vram-budget.py', 'memory_backend.py', 'gpu_primitives_shim.py'):
            ast.parse((ROOT / 'scripts' / filename).read_text(encoding='utf-8'))

    def test_roadmap_has_explicit_blocked_status(self):
        content = (ROOT / 'docs' / 'ROADMAP.md').read_text(encoding='utf-8')
        self.assertIn('BLOCKED', content)
        self.assertIn('v1.0', content)


if __name__ == '__main__':
    unittest.main()
