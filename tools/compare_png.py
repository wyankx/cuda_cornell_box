#!/usr/bin/env python3
"""Self-check for test task 02: mean brightness of PNG images and their difference.

Standard library only (no Pillow needed).

    python3 tools/compare_png.py images/cornell_accum.png
    python3 tools/compare_png.py images/cornell_accum.png tools/expected_direct_128.png

For each image the script prints its size and mean brightness L (0-255, the same
value as Pillow's Image.convert('L')). With two images it also prints the
difference: RMSE, max |delta| and the share of pixels that differ by more than 16
levels. If the sizes differ by an integer factor (512 vs 128), the larger image is
first reduced by averaging blocks of pixels.

Reference for the base part (direct light only, 512x512): mean brightness of the
accumulated image (1024 spp) is about 37, of a single 1 spp frame about 35.6.
"""
import math
import struct
import sys
import zlib


def read_png(path):
    """Return (width, height, rows) where rows[y] is a list of (r, g, b) tuples."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG file")
    pos, idat, ihdr = 8, [], None
    while pos < len(data):
        length, tag = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
        pos += 12 + length
    if ihdr is None:
        raise ValueError(f"{path}: no IHDR chunk")
    width, height, depth, color, _, _, interlace = ihdr
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color)
    if depth != 8 or channels is None or interlace != 0:
        raise ValueError(f"{path}: only 8-bit gray/RGB/RGBA non-interlaced PNG is supported")

    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    prev = bytearray(stride)
    rows = []
    for y in range(height):
        start = y * (stride + 1)
        ftype = raw[start]
        line = bytearray(raw[start + 1:start + 1 + stride])
        for i in range(stride):
            a = line[i - channels] if i >= channels else 0
            b = prev[i]
            c = prev[i - channels] if i >= channels else 0
            if ftype == 1:
                line[i] = (line[i] + a) & 0xFF
            elif ftype == 2:
                line[i] = (line[i] + b) & 0xFF
            elif ftype == 3:
                line[i] = (line[i] + (a + b) // 2) & 0xFF
            elif ftype == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 0xFF
        prev = line
        if channels >= 3:
            rows.append([tuple(line[x * channels:x * channels + 3]) for x in range(width)])
        else:
            rows.append([(line[x * channels],) * 3 for x in range(width)])
    return width, height, rows


def luma(rgb):
    # ITU-R 601-2, integer rounding as in Pillow's convert('L').
    r, g, b = rgb
    return (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16


def mean_brightness(rows):
    total = sum(luma(p) for row in rows for p in row)
    return total / (len(rows) * len(rows[0]))


def downscale(width, height, rows, factor):
    out = []
    n = factor * factor
    for y in range(0, height, factor):
        line = []
        for x in range(0, width, factor):
            acc = [0, 0, 0]
            for yy in range(y, y + factor):
                for xx in range(x, x + factor):
                    p = rows[yy][xx]
                    acc[0] += p[0]; acc[1] += p[1]; acc[2] += p[2]
            line.append(tuple((v + n // 2) // n for v in acc))
        out.append(line)
    return width // factor, height // factor, out


def match_sizes(a, b):
    (wa, ha, ra), (wb, hb, rb) = a, b
    if (wa, ha) == (wb, hb):
        return a, b
    if wa > wb and wa % wb == 0 and ha == hb * (wa // wb):
        return downscale(wa, ha, ra, wa // wb), b
    if wb > wa and wb % wa == 0 and hb == ha * (wb // wa):
        return a, downscale(wb, hb, rb, wb // wa)
    raise ValueError(f"sizes {wa}x{ha} and {wb}x{hb} differ by a non-integer factor")


def compare(a, b):
    (w, h, ra), (_, _, rb) = match_sizes(a, b)
    sq, worst, big = 0, 0, 0
    for y in range(h):
        for x in range(w):
            pa, pb = ra[y][x], rb[y][x]
            d = [abs(pa[c] - pb[c]) for c in range(3)]
            sq += sum(v * v for v in d)
            worst = max(worst, max(d))
            big += max(d) > 16
    return math.sqrt(sq / (w * h * 3)), worst, big / (w * h), (w, h)


def main(argv):
    if len(argv) not in (2, 3):
        print("usage:\n" + __doc__.strip().split("\n\n")[2])
        return 2
    images = []
    for path in argv[1:]:
        img = read_png(path)
        images.append(img)
        print(f"{path}: {img[0]}x{img[1]}, mean brightness L = {mean_brightness(img[2]):.2f}")
    if len(images) == 2:
        rmse, worst, big, (w, h) = compare(*images)
        print(f"difference at {w}x{h}: RMSE = {rmse:.2f}, max |delta| = {worst}, "
              f"pixels with |delta| > 16: {100 * big:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
