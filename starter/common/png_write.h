// png_write.h — dependency-free PNG writer for test task 02 (any OS).
//
// Helper code, not part of the task. Writes 8-bit RGB PNG files using
// uncompressed ("stored") deflate blocks: files are larger than usual
// (~770 KB for 512x512) but valid everywhere and need no zlib.
//
// WriteLinearPng() converts a linear-RGB float image to sRGB 8-bit:
//   c = clamp(rgb * scale * exposure, 0, 1), then the sRGB OETF.
// Pass scale = 1 / sampleCount if your buffer holds a SUM of samples, or
// scale = 1 if it already holds the average.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace bhs {

namespace detail {
inline uint32_t Crc32(const uint8_t* data, size_t n, uint32_t crc = 0) {
    static uint32_t table[256];
    static bool init = false;
    if (!init) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        init = true;
    }
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

inline void PutU32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(uint8_t(x >> 24)); v.push_back(uint8_t(x >> 16));
    v.push_back(uint8_t(x >> 8)); v.push_back(uint8_t(x));
}

inline void Chunk(std::vector<uint8_t>& out, const char* tag, const std::vector<uint8_t>& data) {
    PutU32(out, (uint32_t)data.size());
    std::vector<uint8_t> body(tag, tag + 4);
    body.insert(body.end(), data.begin(), data.end());
    out.insert(out.end(), body.begin(), body.end());
    PutU32(out, Crc32(body.data(), body.size()));
}
}  // namespace detail

// rgb: width*height*3 bytes, rows top to bottom.
inline bool WritePngRGB8(const std::string& path, const uint8_t* rgb, int width, int height) {
    std::vector<uint8_t> raw;
    raw.reserve((size_t)height * (width * 3 + 1));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);  // filter: none
        raw.insert(raw.end(), rgb + (size_t)y * width * 3, rgb + (size_t)(y + 1) * width * 3);
    }
    // zlib stream with stored blocks.
    std::vector<uint8_t> z = {0x78, 0x01};
    size_t pos = 0;
    do {
        size_t len = std::min<size_t>(65535, raw.size() - pos);
        bool last = pos + len == raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(uint8_t(len)); z.push_back(uint8_t(len >> 8));
        z.push_back(uint8_t(~len)); z.push_back(uint8_t(~len >> 8));
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + len);
        pos += len;
    } while (pos < raw.size());
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    detail::PutU32(z, (b << 16) | a);

    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    std::vector<uint8_t> ihdr;
    detail::PutU32(ihdr, (uint32_t)width);
    detail::PutU32(ihdr, (uint32_t)height);
    ihdr.insert(ihdr.end(), {8, 2, 0, 0, 0});  // 8 bit, RGB, deflate, no filter, no interlace
    detail::Chunk(png, "IHDR", ihdr);
    detail::Chunk(png, "IDAT", z);
    detail::Chunk(png, "IEND", {});

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "WritePngRGB8: cannot open %s\n", path.c_str()); return false; }
    bool ok = std::fwrite(png.data(), 1, png.size(), f) == png.size();
    ok = (std::fclose(f) == 0) && ok;
    return ok;
}

inline float LinearToSrgb(float c) {
    c = std::min(std::max(c, 0.0f), 1.0f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}

// rgba: width*height*4 floats (linear RGB + unused 4th channel), rows top to bottom.
inline bool WriteLinearPng(const std::string& path, const float* rgba, int width, int height, float scale,
                           float exposure = 1.0f) {
    std::vector<uint8_t> rgb((size_t)width * height * 3);
    size_t nonFinite = 0;
    for (size_t i = 0; i < (size_t)width * height; ++i)
        for (int c = 0; c < 3; ++c) {
            float v = rgba[i * 4 + c] * scale * exposure;
            if (!std::isfinite(v)) { v = 0.0f; ++nonFinite; }  // written as black, but reported
            rgb[i * 3 + c] = (uint8_t)std::lround(LinearToSrgb(v) * 255.0f);
        }
    if (nonFinite) std::fprintf(stderr, "WriteLinearPng: %zu NaN/Inf values in %s\n", nonFinite, path.c_str());
    return WritePngRGB8(path, rgb.data(), width, height);
}

}  // namespace bhs
