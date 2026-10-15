// main.cu — headless Cornell box renderer skeleton for test task 02 (CUDA track:
// any NVIDIA GPU, including the free T4 in Google Colab).
//
// Already done: scene loading (common/cornell_scene.h), GPU buffers for
// triangles / materials / accumulation, camera basis, the frame loop (one
// launch = one sample per pixel) and PNG output (common/png_write.h):
// images/cornell_1spp.png after the first frame and images/cornell_accum.png
// after the last one.
//
// Your part: the kernel in trace.cuh.
//
// Build and run from the package root:
//   nvcc -O2 -std=c++17 -o cornell starter/cuda/main.cu
//   ./cornell                    # 256 frames
//   ./cornell --frames 1024      # more samples
//   ./cornell --out images       # output directory (created if missing)
//   ./cornell --scene scene      # directory with cornell_box.obj / .mtl

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "../common/cornell_scene.h"
#include "../common/png_write.h"
#include "trace.cuh"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,    \
                         #call, cudaGetErrorString(err_));                        \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

namespace {

float3 ToFloat3(const float v[3]) { return make_float3(v[0], v[1], v[2]); }

FrameParams MakeParams(const bhs::CornellScene& s) {
    FrameParams p{};
    const bhs::CornellCamera& c = s.camera;
    p.width = (unsigned int)c.width;
    p.height = (unsigned int)c.height;
    p.triangleCount = (unsigned int)s.triangles.size();
    p.camPos = ToFloat3(c.position);
    p.camForward = normalize(ToFloat3(c.lookAt) - p.camPos);
    p.camRight = normalize(cross(p.camForward, ToFloat3(c.up)));  // right-handed: forward x up = image right
    p.camUp = cross(p.camRight, p.camForward);
    p.tanHalfFovY = std::tan(c.verticalFovDeg * 0.5f * BHS_PI / 180.0f);
    p.aspect = (float)c.width / (float)c.height;
    const bhs::CornellLight& l = s.light;
    p.lightCorner = ToFloat3(l.corner);
    p.lightEdge1 = ToFloat3(l.edge1);
    p.lightEdge2 = ToFloat3(l.edge2);
    p.lightNormal = ToFloat3(l.normal);
    p.lightRadiance = ToFloat3(l.radiance);
    p.lightArea = l.area;
    return p;
}

}  // namespace

int main(int argc, char** argv) {
    int frames = 256;
    std::string outDir = "images", sceneDir = "scene";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--frames" && i + 1 < argc) frames = std::max(1, std::atoi(argv[++i]));
        else if (a == "--out" && i + 1 < argc) outDir = argv[++i];
        else if (a == "--scene" && i + 1 < argc) sceneDir = argv[++i];
        else { std::fprintf(stderr, "usage: %s [--frames N] [--out DIR] [--scene DIR]\n", argv[0]); return 2; }
    }
    std::filesystem::create_directories(outDir);

    bhs::CornellScene scene;
    if (!bhs::LoadCornellScene(sceneDir, scene)) return 1;
    FrameParams params = MakeParams(scene);
    std::printf("scene: %zu triangles, %zu materials, %ux%u\n", scene.triangles.size(),
                scene.materials.size(), params.width, params.height);

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s\n", prop.name);

    static_assert(sizeof(Triangle) == sizeof(bhs::Triangle), "Triangle layout");
    static_assert(sizeof(Material) == sizeof(bhs::Material), "Material layout");
    const size_t pixelCount = (size_t)params.width * params.height;
    Triangle* dTriangles = nullptr;
    Material* dMaterials = nullptr;
    float4* dAccum = nullptr;
    CUDA_CHECK(cudaMalloc(&dTriangles, scene.triangles.size() * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&dMaterials, scene.materials.size() * sizeof(Material)));
    CUDA_CHECK(cudaMalloc(&dAccum, pixelCount * sizeof(float4)));
    CUDA_CHECK(cudaMemcpy(dTriangles, scene.triangles.data(), scene.triangles.size() * sizeof(Triangle),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dMaterials, scene.materials.data(), scene.materials.size() * sizeof(Material),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dAccum, 0, pixelCount * sizeof(float4)));

    std::vector<float> host(pixelCount * 4);
    const dim3 block(8, 8);
    const dim3 grid((params.width + block.x - 1) / block.x, (params.height + block.y - 1) / block.y);
    const std::string out1 = outDir + "/cornell_1spp.png";
    const std::string outN = outDir + "/cornell_accum.png";
    auto t0 = std::chrono::steady_clock::now();
    for (int f = 0; f < frames; ++f) {
        params.frameIndex = (unsigned int)f;
        trace<<<grid, block>>>(params, dTriangles, dMaterials, dAccum);  // params copied: safe to change next frame
        CUDA_CHECK(cudaGetLastError());
        if (f == 0) {  // only to save the 1 spp image
            CUDA_CHECK(cudaMemcpy(host.data(), dAccum, pixelCount * sizeof(float4), cudaMemcpyDeviceToHost));
            bhs::WriteLinearPng(out1, host.data(), (int)params.width, (int)params.height, 1.0f);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    CUDA_CHECK(cudaMemcpy(host.data(), dAccum, pixelCount * sizeof(float4), cudaMemcpyDeviceToHost));
    bhs::WriteLinearPng(outN, host.data(), (int)params.width, (int)params.height, 1.0f / (float)frames);
    std::printf("%d spp in %.2f s (%.2f ms/frame) -> %s, %s\n", frames, secs, 1000.0 * secs / frames,
                out1.c_str(), outN.c_str());

    CUDA_CHECK(cudaFree(dTriangles));
    CUDA_CHECK(cudaFree(dMaterials));
    CUDA_CHECK(cudaFree(dAccum));
    return 0;
}
