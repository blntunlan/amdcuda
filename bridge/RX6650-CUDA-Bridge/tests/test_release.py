import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_release', ROOT / 'scripts' / 'verify-release.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseTests(unittest.TestCase):
    def test_manifest_verifies_actual_code(self):
        self.assertEqual(module.verify(ROOT), [])

    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'src.txt').write_text('changed', encoding='utf-8')
            (p / 'MANIFEST.json').write_text(json.dumps({'sha256': {'src.txt': '0' * 64}}), encoding='utf-8')
            self.assertIn('Checksum mismatch', ' '.join(module.verify(p)))

    def test_manifest_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'MANIFEST.json').write_text(json.dumps({'sha256': {'../passwords': '0' * 64}}), encoding='utf-8')
            self.assertIn('unsafe path', ' '.join(module.verify(p)))


if __name__ == '__main__':
    unittest.main()
