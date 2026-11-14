"""Memory-bounded 4K PBR CPU renderer. NumPy compute, streaming PNG output.

No AMD GPU acceleration is claimed. Preserves v0.6's PBR equations and UV
normalization; vertical halo rows avoid mismatched normals at strip boundaries.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import time
import zlib
from pathlib import Path

from pbr_backend import CHANNELS, FORMATS, MAPS, PRESETS, PNG_SIGNATURE, TAU, height_at, surface_channels


def _np():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('4K renderer requires numpy: python -m pip install numpy Pillow') from exc
    return np


def validate_4k_params(width: int, height: int, tile_rows: int, preset: str, normal_format: str, strength: float, seed: int):
    if type(width) is not int or type(height) is not int or not 8 <= width <= 4096 or not 8 <= height <= 4096:
        raise ValueError('4K renderer supports integer width and height in [8, 4096]')
    if type(tile_rows) is not int or not 1 <= tile_rows <= 1024:
        raise ValueError('tile_rows must be an integer in [1, 1024]')
    if preset not in PRESETS or normal_format not in FORMATS:
        raise ValueError('Invalid PBR preset or normal convention')
    if not math.isfinite(strength) or not 0 <= strength <= 10:
        raise ValueError('normal strength must be finite in [0, 10]')
    if type(seed) is not int or not 0 <= seed <= 1_000_000:
        raise ValueError('seed must be integer in [0, 1000000]')


def _chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack('>I', len(payload)) + tag + payload +
            struct.pack('>I', zlib.crc32(tag + payload) & 0xffffffff))


class StreamingPNG:
    """Writes standards-compliant filter-0 PNG with bounded zlib/row memory."""
    def __init__(self, path: Path, width: int, height: int, channels: int):
        self.path = path
        self.width, self.height, self.channels = width, height, channels
        self.rows = 0
        self.stream = path.open('wb')
        self.stream.write(PNG_SIGNATURE)
        ihdr = struct.pack('>IIBBBBB', width, height, 8, 0 if channels == 1 else 2, 0, 0, 0)
        self.stream.write(_chunk(b'IHDR', ihdr))
        self.compressor = zlib.compressobj(level=6)
        self.digest = hashlib.sha256()
        self.closed = False

    def _emit(self, data: bytes):
        for offset in range(0, len(data), 1_048_576):
            self.stream.write(_chunk(b'IDAT', data[offset:offset+1_048_576]))

    def write_rows(self, rows):
        if self.closed:
            raise ValueError('PNG is already finished')
        np = _np()
        expected_shape = (rows.shape[0], self.width) if self.channels == 1 else (rows.shape[0], self.width, self.channels)
        if rows.shape != expected_shape or rows.dtype != np.uint8 or self.rows + len(rows) > self.height:
            raise ValueError('PNG tile shape/dtype/row count mismatch')
        # Compress in row order, keeping memory below a single encoded scanline.
        for row in rows:
            raw = row.tobytes(order='C')
            self.digest.update(raw)
            self._emit(self.compressor.compress(b'\x00' + raw))
            self.rows += 1

    def finish(self) -> dict:
        if self.rows != self.height:
            raise ValueError(f'PNG incomplete ({self.rows}/{self.height} rows)')
        self._emit(self.compressor.flush())
        self.stream.write(_chunk(b'IEND', b''))
        self.stream.close()
        self.closed = True
        return {'file': self.path.name, 'bytes': self.path.stat().st_size,
                'sha256_raw': self.digest.hexdigest(),
                'sha256_png': hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'channels': self.channels}

    def abort(self):
        if not self.closed:
            self.stream.close()
            self.closed = True


def make_tile(width: int, height: int, y_start: int, nrows: int, *, preset: str, seed: int,
              normal_format: str, strength: float):
    """Compute tile with 1px halo top/bottom for correct normal gradients."""
    np = _np()
    xx = (np.arange(width, dtype=np.float64) / width)[None, :]
    yy = ((np.arange(y_start-1, y_start+nrows+1, dtype=np.int64) % height).astype(np.float64) / height)[:, None]
    heights = height_at(xx, yy, preset, seed, np.sin, np.cos)
    h = heights[1:-1]
    left = np.roll(h, 1, axis=1)
    right = np.roll(h, -1, axis=1)
    top, bottom = heights[:-2], heights[2:]
    avg = (left + right + top + bottom) * .25
    # The v0.6 formula works in pixel step units; without this normalization
    # a 4096px normal becomes 16x flatter than the 256px reference material.
    # Preserve exact 8..256 behavior, then normalize larger maps per axis.
    scale_x, scale_y = max(1., width / 256.), max(1., height / 256.)
    dx = (right - left) * .5 * strength * scale_x
    dy = (bottom - top) * .5 * strength * scale_y
    nx, ny, nz = -dx, -dy * (-1.0 if normal_format == 'directx' else 1.0), np.ones_like(h)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    clamp = lambda a: np.clip(a, 0., 1.)
    base, rough, metal, ao = surface_channels(h, avg, preset, clamp)
    u8 = lambda a: np.rint(np.clip(a, 0, 1) * 255).astype(np.uint8)
    base_rgb = np.stack([u8(b) for b in base], axis=-1)
    normal_rgb = np.stack([u8(((v / norm) + 1) * .5) for v in (nx, ny, nz)], axis=-1)
    rr, mm, aa = u8(rough), u8(metal), u8(ao)
    orm = np.stack((aa, rr, mm), axis=-1)
    results = {'BaseColor': base_rgb, 'Normal': normal_rgb, 'Roughness': rr, 'Metallic': mm,
               'AO': aa, 'ORM': orm, 'Height': u8(h)}
    # Validate before writing: avoid silent bad data from computation.
    for k, v in results.items():
        if v.shape[0] != nrows or v.shape[1] != width:
            raise AssertionError(f'Bad tile shape: {k}')
    norm_decoded = (normal_rgb.astype(np.float32) * (2 / 255)) - 1
    err = np.max(np.abs(np.linalg.norm(norm_decoded, axis=2) - 1.0))
    if err > .015:
        raise ValueError(f'Invalid normal vectors: {err}')
    if not np.array_equal(orm[..., 0], aa) or not np.array_equal(orm[..., 1], rr) or not np.array_equal(orm[..., 2], mm):
        raise ValueError('ORM mismatch')
    return results, float(err)


def verify_png_readback(directory: Path, width: int, height: int, specs: dict, *, rows_per_check: int = 256) -> dict:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError('4K PNG verification requires Pillow: python -m pip install Pillow') from exc
    readbacks = {}
    for name in MAPS:
        path = directory / f'{name}.png'
        with Image.open(path) as img:
            expected_mode = 'RGB' if CHANNELS[name] == 3 else 'L'
            if img.format != 'PNG' or img.size != (width, height) or img.mode != expected_mode:
                raise ValueError(f'{name}: invalid PNG format, mode, dimensions')
            h = hashlib.sha256()
            for y in range(0, height, rows_per_check):
                tile = img.crop((0, y, width, min(height, y+rows_per_check)))
                h.update(tile.tobytes())
            if h.hexdigest() != specs[name]['sha256_raw']:
                raise ValueError(f'{name}: PNG pixel readback SHA256 mismatch')
        readbacks[name] = True
    return readbacks


def render_material(out_dir: Path, width: int = 4096, height: int = 4096, *, tile_rows: int = 256,
                    preset: str = 'basalt', seed: int = 6650, normal_format: str = 'directx',
                    strength: float = 4., verify: bool = True) -> dict:
    validate_4k_params(width, height, tile_rows, preset, normal_format, strength, seed)
    # Fail fast if Pillow verification is required but unavailable.
    if verify:
        try:
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('Pillow is required for verified 4K export') from exc
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    writers = {name: StreamingPNG(out_dir / f'{name}.png', width, height, CHANNELS[name]) for name in MAPS}
    normal_max_error = 0.
    try:
        for y in range(0, height, tile_rows):
            rows, err = make_tile(width, height, y, min(tile_rows, height-y),
                                  preset=preset, seed=seed, normal_format=normal_format, strength=strength)
            normal_max_error = max(normal_max_error, err)
            for name in MAPS:
                writers[name].write_rows(rows[name])
        specs = {name: writers[name].finish() for name in MAPS}
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    readbacks = verify_png_readback(out_dir, width, height, specs) if verify else {}
    details = {'format_version': 2, 'preset': preset, 'seed': seed, 'normal_format': normal_format,
               'normal_strength': strength, 'width': width, 'height': height, 'backend': 'numpy-cpu-streaming',
               'tile_rows': tile_rows, 'tile_count': math.ceil(height / tile_rows), 'maps': specs,
               'normal_scale_x': max(1., width / 256.), 'normal_scale_y': max(1., height / 256.),
               'pixel_readback_verified': verify and all(readbacks.values()),
               'orm_verified': True, 'normal_max_length_error': round(normal_max_error, 7),
               'color_space': {k: ('sRGB' if k == 'BaseColor' else 'linear-data') for k in MAPS},
               'orm_channels': {'R': 'AO', 'G': 'Roughness', 'B': 'Metallic'},
               'gpu_compute_confirmed': False, 'elapsed_seconds': round(time.perf_counter()-started, 4)}
    (out_dir / 'material.json').write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')
    return details
