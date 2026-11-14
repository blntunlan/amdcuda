"""Compare streaming NumPy against scalar v0.6 reference at small dimensions."""
from __future__ import annotations
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import numpy
    from PIL import Image
    DEPS = True
except ImportError:
    DEPS = False
from pbr_backend import MAPS, generate_cpu
from pbr_4k_backend import make_tile, render_material, validate_4k_params


@unittest.skipUnless(DEPS, 'numpy/Pillow required for 4K tests')
class Stream4KTests(unittest.TestCase):
    def test_exact_cpu_reference_small_directx(self):
        for preset in ('basalt','worn_metal'):
            w,h = 64,48
            ref = generate_cpu(w,h,preset=preset, normal_format='directx')
            tiles = [make_tile(w,h,y,min(13,h-y),preset=preset,seed=6650,
                               normal_format='directx',strength=4)[0] for y in range(0,h,13)]
            for name in MAPS:
                stitched = b''.join(t[name].tobytes() for t in tiles)
                self.assertEqual(ref[name],stitched, f'{name} {preset} differs from CPU reference')

    def test_exact_cpu_reference_opengl(self):
        w,h=24,29
        ref=generate_cpu(w,h,preset='basalt',normal_format='opengl',strength=2.25,seed=144)
        tile = make_tile(w,h,0,h,preset='basalt',seed=144,normal_format='opengl',strength=2.25)[0]
        for name in MAPS:
            self.assertEqual(ref[name],tile[name].tobytes(),name)

    def test_png_roundtrip_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            meta=render_material(Path(temp),65,37,tile_rows=8,preset='worn_metal',normal_format='opengl')
            self.assertEqual(meta['tile_count'],5)
            self.assertTrue(meta['pixel_readback_verified'])
            self.assertEqual(len(list(Path(temp).glob('*.png'))),7)
            for name in MAPS:
                with Image.open(Path(temp)/f'{name}.png') as im:
                    self.assertEqual(im.size,(65,37))
            self.assertFalse(meta['gpu_compute_confirmed'])


    def test_normal_amplitude_does_not_collapse_at_512(self):
        import numpy as np
        n256=make_tile(256,256,0,256,preset='basalt',seed=6650,
                       normal_format='directx',strength=4)[0]['Normal']
        n512=make_tile(512,512,0,512,preset='basalt',seed=6650,
                       normal_format='directx',strength=4)[0]['Normal']
        std256=float(np.std(n256[...,0])); std512=float(np.std(n512[...,0]))
        self.assertGreater(std512, 0.70*std256)
        self.assertLess(std512, 1.30*std256)

    def test_png_writer_rejects_incomplete_output(self):
        from pbr_4k_backend import StreamingPNG
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            writer=StreamingPNG(Path(d)/'incomplete.png',8,8,1)
            try:
                writer.write_rows(np.zeros((2,8),dtype=np.uint8))
                with self.assertRaises(ValueError):
                    writer.finish()
            finally:
                writer.abort()

    def test_reject_bad_params(self):
        cases = [(4097,4096,256,'basalt','directx',4.,6650),
                 (4096,4096,0,'basalt','directx',4.,6650),
                 (4096,4096,256,'bad','directx',4.,6650)]
        for values in cases:
            with self.assertRaises(ValueError):
                validate_4k_params(*values)

    def test_row_partition_independence(self):
        with tempfile.TemporaryDirectory() as d:
            a=render_material(Path(d)/'a',72,61,tile_rows=7)
            b=render_material(Path(d)/'b',72,61,tile_rows=31)
            for name in MAPS:
                self.assertEqual(a['maps'][name]['sha256_raw'],b['maps'][name]['sha256_raw'])
