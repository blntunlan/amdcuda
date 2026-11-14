import importlib.util
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pbr_backend import (MAPS, CHANNELS, check_params, decode_own_png, encode_png,
                         export_maps, generate_cpu, height_at, load_height_png, validate_maps)


class PBRMaterialTests(unittest.TestCase):
    def setUp(self):
        self.maps = generate_cpu(32, 24, preset='basalt', seed=6650, normal_format='directx')

    def test_all_seven_maps_in_correct_channel_shapes(self):
        self.assertEqual(set(self.maps), set(MAPS))
        for name, data in self.maps.items():
            self.assertEqual(len(data), 32 * 24 * CHANNELS[name])

    def test_orm_channel_mapping_exact(self):
        ao, rough, metal = (self.maps[k] for k in ('AO', 'Roughness', 'Metallic'))
        self.assertEqual(self.maps['ORM'], bytes(c for i in range(32*24)
                                                 for c in (ao[i], rough[i], metal[i])))

    def test_normal_vectors_close_to_unit(self):
        m = validate_maps(self.maps, 32, 24, normal_format='directx')
        self.assertLess(m['normal_max_length_error'], .015)

    def test_metal_is_zero_for_basalt(self):
        self.assertEqual(set(self.maps['Metallic']), {0})

    def test_metallic_for_worn_metal(self):
        m = generate_cpu(16, 16, preset='worn_metal')
        self.assertGreater(min(m['Metallic']), 200)
        self.assertLessEqual(max(m['Metallic']), 255)

    def test_presets_have_different_base_color(self):
        m = generate_cpu(32, 24, preset='worn_metal')
        self.assertNotEqual(m['BaseColor'], self.maps['BaseColor'])

    def test_seed_determinism(self):
        self.assertEqual(self.maps, generate_cpu(32, 24, seed=6650))

    def test_different_seeds_generate_different_material(self):
        self.assertNotEqual(self.maps['Height'], generate_cpu(32, 24, seed=6651)['Height'])

    def test_different_sizes_supported(self):
        m = generate_cpu(8, 11)
        self.assertEqual(len(m['BaseColor']), 8*11*3)

    def test_zero_normal_strength_is_flat(self):
        m = generate_cpu(16, 16, strength=0)
        self.assertEqual(len(set(tuple(m['Normal'][i:i+3]) for i in range(0, len(m['Normal']), 3))), 1)

    def test_directx_and_opengl_only_flip_green(self):
        gl = generate_cpu(32, 24, normal_format='opengl')['Normal']
        dx = self.maps['Normal']
        for i in range(0, len(dx), 3):
            self.assertEqual((dx[i], dx[i+2]), (gl[i], gl[i+2]))
            self.assertLessEqual(abs((dx[i+1] + gl[i+1]) - 255), 1)

    def test_periodic_height_equation(self):
        for x, y in ((0.2, 0.32), (0, 1), (.97, -.2)):
            h = height_at(x, y, 'basalt', 6650, math.sin, math.cos)
            self.assertAlmostEqual(h, height_at(x+1, y, 'basalt', 6650, math.sin, math.cos), places=12)
            self.assertAlmostEqual(h, height_at(x, y+1, 'basalt', 6650, math.sin, math.cos), places=12)

    def test_png_roundtrip_all_maps(self):
        for name in MAPS:
            raw = encode_png(32, 24, CHANNELS[name], self.maps[name])
            self.assertEqual(decode_own_png(raw), (32, 24, CHANNELS[name], self.maps[name]))

    def test_png_crc_corruption_rejected(self):
        raw = bytearray(encode_png(32, 24, 1, self.maps['AO']))
        raw[-16] ^= 0x20
        with self.assertRaises(ValueError):
            decode_own_png(bytes(raw))

    def test_png_truncated_rejected(self):
        with self.assertRaises(ValueError):
            decode_own_png(encode_png(32, 24, 1, self.maps['AO'])[:-6])

    def test_png_invalid_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            encode_png(0, 1, 1, b'')

    def test_png_decompression_bomb_rejected(self):
        with self.assertRaises(ValueError):
            decode_own_png(encode_png(32, 24, 1, self.maps['AO']), max_pixels=20)

    def test_export_all_maps_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = export_maps(Path(tmp), self.maps, 32, 24, preset='basalt', seed=6650,
                               normal_format='directx', strength=4)
            self.assertTrue(meta['roundtrip_verified'])
            self.assertEqual(meta['orm_channels'], {'R':'AO','G':'Roughness','B':'Metallic'})
            self.assertEqual(len(list(Path(tmp).glob('*.png'))), 7)
            self.assertEqual(json.loads((Path(tmp)/'material.json').read_text())['backend'], 'cpu-reference')

    def test_import_grayscale_height_own_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / 'input.png'
            image.write_bytes(encode_png(32, 24, 1, self.maps['Height']))
            w,h,pixels = load_height_png(image)
            new = generate_cpu(w,h,height_override=pixels)
            self.assertEqual(new['Height'], pixels)

    def test_import_reject_wrong_byte_count(self):
        with self.assertRaises(ValueError):
            generate_cpu(16, 16, height_override=b'bad')

    def test_validation_detects_invalid_normal(self):
        corrupted = dict(self.maps)
        corrupted['Normal'] = bytes(len(self.maps['Normal']))
        with self.assertRaises(ValueError):
            validate_maps(corrupted,32,24,normal_format='directx')

    def test_validation_detects_invalid_orm(self):
        corrupted = dict(self.maps)
        corrupted['ORM'] = bytes(len(self.maps['ORM']))
        with self.assertRaises(ValueError):
            validate_maps(corrupted,32,24,normal_format='directx')

    def test_validation_rejects_missing_map(self):
        m = dict(self.maps)
        m.pop('AO')
        with self.assertRaises(ValueError):
            validate_maps(m,32,24,normal_format='directx')

    def test_validation_rejects_wrong_dimension(self):
        with self.assertRaises(ValueError):
            validate_maps(self.maps,24,32+1,normal_format='directx')

    def test_nan_normal_strength_rejected(self):
        with self.assertRaises(ValueError):
            check_params(32,32,'basalt','directx',float('nan'),6650)

    def test_huge_resolution_rejected_before_allocation(self):
        with self.assertRaises(ValueError):
            check_params(2048,2048,'basalt','directx',4,6650)

    def test_wrong_preset_rejected(self):
        with self.assertRaises(ValueError):
            check_params(32,32,'unknown','directx',4,6650)

    def test_wrong_format_rejected(self):
        with self.assertRaises(ValueError):
            check_params(32,32,'basalt','tangent',4,6650)

    def test_bad_seed_rejected(self):
        with self.assertRaises(ValueError):
            check_params(32,32,'basalt','directx',4,-1)

    def test_Pillow_can_read_all_generated_images(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow optional')
        with tempfile.TemporaryDirectory() as tmp:
            export_maps(Path(tmp), self.maps, 32,24,preset='basalt',seed=6650,
                        normal_format='directx',strength=4)
            for name in MAPS:
                with Image.open(Path(tmp)/f'{name}.png') as im:
                    self.assertEqual(im.size, (32,24))
                    self.assertEqual(im.mode, 'L' if CHANNELS[name] == 1 else 'RGB')


if __name__ == '__main__':
    unittest.main()
