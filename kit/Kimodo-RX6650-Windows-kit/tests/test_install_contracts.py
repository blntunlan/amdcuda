from pathlib import Path
import unittest
root=Path(__file__).resolve().parents[1]
class ContractTests(unittest.TestCase):
    def test_bootstrap_targets_exact_gfx(self):
        x=(root/'install.ps1').read_text()
        self.assertIn('torch[device-gfx1032]', x)
        self.assertIn('SKIP_MOTION_CORRECTION_IN_SETUP', x)
    def test_gpu_does_not_silently_fallback(self):
        x=(root/'generate.ps1').read_text()
        self.assertIn('No silent CPU fallback', x)
        self.assertIn("$LASTEXITCODE -ne 0", x)
    def test_postprocess_explicitly_disabled(self):
        self.assertIn('--no-postprocess', (root/'generate.ps1').read_text())
    def test_cpu_encoder_offload(self):
        self.assertIn("$env:TEXT_ENCODER_DEVICE='cpu'", (root/'generate.ps1').read_text())
    def test_license_not_embedded(self):
        self.assertNotIn('hf_', (root/'README.md').read_text().replace('huggingface',''))
    def test_docs_have_clear_blockers(self):
        x=(root/'ROADMAP_TR.md').read_text()
        self.assertIn('BLOCKED', x)
        self.assertIn('16 GB', x)
if __name__=='__main__': unittest.main()
