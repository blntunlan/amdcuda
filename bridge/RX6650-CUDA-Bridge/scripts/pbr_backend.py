"""Dependency-free tiled PBR generator and strict, minimal PNG IO.

Reference pipeline for v0.6. Values are explicitly generated in linear/scalar space;
BaseColor PNG is treated as display/sRGB encoded, and packed ORM uses raw data.
No assertion that this CPU reference is GPU accelerated.
"""
from __future__ import annotations

import json
import math
import struct
import zlib
from pathlib import Path

MAPS = ('BaseColor', 'Normal', 'Roughness', 'Metallic', 'AO', 'ORM', 'Height')
CHANNELS = {'BaseColor': 3, 'Normal': 3, 'Roughness': 1,
            'Metallic': 1, 'AO': 1, 'ORM': 3, 'Height': 1}
PRESETS = ('basalt', 'worn_metal')
FORMATS = ('directx', 'opengl')
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'
TAU = math.tau


def check_params(width: int, height: int, preset: str, normal_format: str,
                 strength: float, seed: int) -> None:
    if type(width) is not int or type(height) is not int or not 8 <= width <= 2048 or not 8 <= height <= 2048:
        raise ValueError('width/height must be integers in [8, 2048]')
    if width * height > 1_048_576:
        raise ValueError('pixel limit is 1,048,576 for bounded CPU/VRAM use')
    if preset not in PRESETS:
        raise ValueError('preset must be basalt or worn_metal')
    if normal_format not in FORMATS:
        raise ValueError('normal format must be directx or opengl')
    if not math.isfinite(strength) or not 0.0 <= strength <= 10.0:
        raise ValueError('normal strength must be finite and in [0, 10]')
    if type(seed) is not int or not 0 <= seed <= 1_000_000:
        raise ValueError('seed must be integer in [0, 1000000]')


def clamp(v: float) -> float:
    return min(1.0, max(0.0, v))


def quant(v: float) -> int:
    if not math.isfinite(v):
        raise ValueError('Nonfinite material channel')
    return int(round(255 * clamp(v)))


def height_at(u, v, preset: str, seed: int, sin, cos):
    """Periodic analytical detail. Scalar and torch tensor arguments both work."""
    p = (seed % 991) * 0.014
    if preset == 'basalt':
        n0 = sin(TAU * 3 * u + p) * cos(TAU * 3 * v - p)
        n1 = cos(TAU * (7 * u + 5 * v) + p) * sin(TAU * (5 * u - 7 * v))
        n2 = sin(TAU * 19 * u - p) * cos(TAU * 13 * v + p)
        return 0.50 + .21 * n0 + .065 * n1 + .023 * n2
    n0 = sin(TAU * 17 * v + p) * cos(TAU * 2 * u - p)
    n1 = sin(TAU * (9 * u + 4 * v) + p) * sin(TAU * 3 * u - p)
    n2 = cos(TAU * 31 * v + p) * sin(TAU * 11 * u + p)
    return 0.50 + .105 * n0 + .08 * n1 + .018 * n2


def surface_channels(h, avg, preset: str, clamp_fn):
    """Shared CPU/GPU scalar or tensor PBR channel equations."""
    if preset == 'basalt':
        base = (clamp_fn(.12 + .26 * h), clamp_fn(.135 + .26 * h), clamp_fn(.16 + .25 * h))
        rough = clamp_fn(.77 + .20 * (.5 - h))
        metal = h * 0
        ao = clamp_fn(.77 + .65 * (h - avg) - .18 * (.5 - h))
    else:
        base = (clamp_fn(.24 + .32 * h), clamp_fn(.225 + .30 * h), clamp_fn(.205 + .27 * h))
        rough = clamp_fn(.46 + .40 * (.5 - h))
        metal = clamp_fn(.78 + .16 * h)
        ao = clamp_fn(.84 + .50 * (h - avg) - .10 * (.5 - h))
    return base, rough, metal, ao


def validate_maps(maps: dict[str, bytes], width: int, height: int, *, normal_format: str) -> dict:
    if set(maps) != set(MAPS):
        raise ValueError(f'Missing/extra maps: {set(MAPS) ^ set(maps)}')
    for name, pixels in maps.items():
        if not isinstance(pixels, bytes) or len(pixels) != width * height * CHANNELS[name]:
            raise ValueError(f'{name}: unexpected channel count or pixel length')
    normal = maps['Normal']
    max_length_error = 0.0
    for j in range(0, len(normal), 3):
        x, y, z = ((c / 255.0) * 2.0 - 1.0 for c in normal[j:j+3])
        err = abs(math.sqrt(x*x + y*y + z*z) - 1.0)
        max_length_error = max(max_length_error, err)
        if z < 0:
            raise ValueError('Normal blue channel must face outward')
    if max_length_error > .015:
        raise ValueError(f'Normals are not normalized: maximum error {max_length_error}')
    ao, rough, metal = maps['AO'], maps['Roughness'], maps['Metallic']
    if maps['ORM'] != bytes(channel for i in range(width * height)
                            for channel in (ao[i], rough[i], metal[i])):
        raise ValueError('ORM channel packing is incorrect (expected AO/Roughness/Metallic)')
    return {'normal_max_length_error': round(max_length_error, 7), 'orm_verified': True,
            'normal_format': normal_format, 'map_count': len(maps), 'width': width, 'height': height}


def generate_cpu(width: int, height: int, *, preset: str = 'basalt', seed: int = 6650,
                 normal_format: str = 'directx', strength: float = 4.0,
                 height_override: bytes | None = None) -> dict[str, bytes]:
    check_params(width, height, preset, normal_format, strength, seed)
    if height_override is not None and len(height_override) != width * height:
        raise ValueError('Imported height must be exactly one 8-bit channel per pixel')
    if height_override is None:
        heights = [height_at(x / width, y / height, preset, seed, math.sin, math.cos)
                   for y in range(height) for x in range(width)]
    else:
        heights = [v / 255 for v in height_override]
    out = {k: bytearray() for k in MAPS}
    green_sign = -1.0 if normal_format == 'directx' else 1.0
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            left = heights[y * width + (x-1) % width]
            right = heights[y * width + (x+1) % width]
            top = heights[((y-1) % height) * width + x]
            bottom = heights[((y+1) % height) * width + x]
            h = heights[idx]
            avg = (left + right + top + bottom) * .25
            dx = (right - left) * .5 * strength
            dy = (bottom - top) * .5 * strength
            nx, ny, nz = -dx, -dy * green_sign, 1.0
            length = math.sqrt(nx*nx + ny*ny + nz*nz)
            base, rough, metal, ao = surface_channels(h, avg, preset, clamp)
            out['BaseColor'].extend(quant(v) for v in base)
            out['Normal'].extend(quant((v / length + 1) * .5) for v in (nx, ny, nz))
            out['Roughness'].append(quant(rough))
            out['Metallic'].append(quant(metal))
            out['AO'].append(quant(ao))
            out['ORM'].extend((quant(ao), quant(rough), quant(metal)))
            out['Height'].append(quant(h))
    maps = {k: bytes(v) for k, v in out.items()}
    validate_maps(maps, width, height, normal_format=normal_format)
    return maps


def _chunk(name: bytes, data: bytes) -> bytes:
    return struct.pack('>I', len(data)) + name + data + struct.pack('>I', zlib.crc32(name + data) & 0xffffffff)


def encode_png(width: int, height: int, channels: int, pixels: bytes) -> bytes:
    if width <= 0 or height <= 0 or channels not in (1, 3) or len(pixels) != width * height * channels:
        raise ValueError('Invalid PNG shape/channels')
    stride = width * channels
    filtered = b''.join(b'\0' + pixels[y*stride:(y+1)*stride] for y in range(height))
    header = struct.pack('>IIBBBBB', width, height, 8, 0 if channels == 1 else 2, 0, 0, 0)
    return PNG_SIGNATURE + _chunk(b'IHDR', header) + _chunk(b'IDAT', zlib.compress(filtered, 6)) + _chunk(b'IEND', b'')


def decode_own_png(raw: bytes, *, max_pixels: int = 1_048_576) -> tuple[int, int, int, bytes]:
    """Intentionally narrow reader: only 8bit grayscale/RGB, filter 0, no interlace.

    Other external PNGs can be imported with Pillow through load_height_png().
    """
    if not raw.startswith(PNG_SIGNATURE):
        raise ValueError('Not PNG')
    off = len(PNG_SIGNATURE)
    chunks = []
    while off < len(raw):
        if off + 12 > len(raw):
            raise ValueError('Truncated PNG chunk')
        length = struct.unpack('>I', raw[off:off+4])[0]
        name = raw[off+4:off+8]
        end = off+12+length
        if length > 16_000_000 or end > len(raw):
            raise ValueError('Invalid or oversized PNG chunk')
        payload = raw[off+8:off+8+length]
        crc = struct.unpack('>I', raw[off+8+length:end])[0]
        if zlib.crc32(name+payload) & 0xffffffff != crc:
            raise ValueError('PNG CRC mismatch')
        chunks.append((name, payload))
        off = end
        if name == b'IEND':
            break
    if off != len(raw) or len(chunks) < 3 or chunks[0][0] != b'IHDR' or chunks[-1][0] != b'IEND' or len(chunks[-1][1]) != 0:
        raise ValueError('Invalid PNG structure')
    if any(n not in (b'IHDR', b'IDAT', b'IEND') for n, _ in chunks):
        raise ValueError('Unsupported PNG ancillary/critical chunk')
    if sum(n == b'IHDR' for n, _ in chunks) != 1 or not any(n == b'IDAT' for n, _ in chunks):
        raise ValueError('PNG missing IHDR/IDAT')
    ihdr = chunks[0][1]
    if len(ihdr) != 13:
        raise ValueError('Invalid IHDR')
    w, h, depth, ctype, compression, filt, interlace = struct.unpack('>IIBBBBB', ihdr)
    channels = {0: 1, 2: 3}.get(ctype)
    if channels is None or depth != 8 or compression or filt or interlace or w <= 0 or h <= 0 or w*h > max_pixels:
        raise ValueError('Unsupported PNG dimensions or format')
    expected = h * (1 + w * channels)
    compressed = b''.join(data for n, data in chunks if n == b'IDAT')
    stream = zlib.decompressobj()
    decompressed = stream.decompress(compressed, expected+1)
    if len(decompressed) != expected or not stream.eof or stream.unconsumed_tail or stream.unused_data:
        raise ValueError('PNG decompression invalid/oversized')
    stride = w * channels
    pixels = bytearray()
    for y in range(h):
        row = decompressed[y*(stride+1):(y+1)*(stride+1)]
        if row[0] != 0:
            raise ValueError('Unsupported PNG row filter')
        pixels.extend(row[1:])
    return w, h, channels, bytes(pixels)


def load_height_png(path: Path) -> tuple[int, int, bytes]:
    try:
        from PIL import Image
    except ImportError:
        w, h, channels, pixels = decode_own_png(path.read_bytes())
        if channels != 1:
            raise ValueError('Height map must be grayscale; install Pillow for RGB/other PNG import')
        return w, h, pixels
    with Image.open(path) as image:
        if image.format != 'PNG' or image.width * image.height > 1_048_576:
            raise ValueError('Height input must be PNG of at most 1,048,576 pixels')
        pixels = image.convert('L').tobytes()
        return image.width, image.height, pixels


def export_maps(out_dir: Path, maps: dict[str, bytes], width: int, height: int,
                *, preset: str, seed: int, normal_format: str, strength: float,
                backend: str = 'cpu-reference') -> dict:
    # Callers choose output destination explicitly. Existing files of the same names are overwritten.
    checks = validate_maps(maps, width, height, normal_format=normal_format)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in MAPS:
        path = out_dir / f'{name}.png'
        path.write_bytes(encode_png(width, height, CHANNELS[name], maps[name]))
        w2, h2, c2, readback = decode_own_png(path.read_bytes())
        if (w2, h2, c2, readback) != (width, height, CHANNELS[name], maps[name]):
            raise ValueError(f'PNG roundtrip failed for {name}')
        files[name] = path.name
    details = {'format_version': 1, 'preset': preset, 'seed': seed, 'normal_format': normal_format,
               'normal_strength': strength, 'width': width, 'height': height, 'maps': files,
               'backend': backend, 'roundtrip_verified': True, **checks,
               'color_space': {'BaseColor': 'sRGB', 'Normal': 'linear-data', 'Roughness': 'linear-data',
                               'Metallic': 'linear-data', 'AO': 'linear-data', 'ORM': 'linear-data', 'Height': 'linear-data'},
               'orm_channels': {'R': 'AO', 'G': 'Roughness', 'B': 'Metallic'},
               'license': 'Generated procedural content; no third-party assets'}
    (out_dir / 'material.json').write_text(json.dumps(details, indent=2) + '\n', encoding='utf-8')
    return details
