// cornell_scene.h — Cornell box loader for test task 02 (any OS, any GPU API).
//
// Helper code, not part of the task. Reads scene/cornell_box.obj + .mtl into
// flat arrays that can be uploaded to the GPU as is, and exposes the camera and
// light parameters from scene/scene.json as constants (no JSON parser needed).
//
// Layout notes (matching structs: starter/metal/src/trace.metal, starter/cuda/trace.cuh):
//   Triangle — 13 x 4 bytes = 52 bytes, no padding (use packed_float3 in MSL,
//              float[3] / scalar layout in GLSL, float3 in CUDA);
//   Material — 6 x 4 bytes = 24 bytes;
//   positions — 9 floats per triangle (v0, v1, v2), non-indexed: convenient as
//              the vertex buffer of an acceleration structure (stride 12 bytes);
//              primitive i of the acceleration structure == triangles[i].
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace bhs {

struct Triangle {
    float v0[3], v1[3], v2[3];
    float normal[3];      // geometric normal from the OBJ (walls point inside, blocks outside)
    uint32_t material;    // index into CornellScene::materials
};
static_assert(sizeof(Triangle) == 52, "Triangle must be tightly packed");

struct Material {
    float albedo[3];      // diffuse albedo, linear RGB
    float emission[3];    // emitted radiance, linear RGB (non-zero only for "light")
};
static_assert(sizeof(Material) == 24, "Material must be tightly packed");

// Values duplicated from scene/scene.json.
struct CornellCamera {
    float position[3] = {278.0f, 273.0f, -800.0f};
    float lookAt[3] = {278.0f, 273.0f, 0.0f};
    float up[3] = {0.0f, 1.0f, 0.0f};
    float verticalFovDeg = 39.3077f;
    int width = 512, height = 512;
};

// Rectangular area light: point = corner + u * edge1 + v * edge2, u, v in [0, 1].
// One-sided: emits only to the side of `normal` (down).
struct CornellLight {
    float corner[3] = {343.0f, 548.3f, 227.0f};
    float edge1[3] = {0.0f, 0.0f, 105.0f};
    float edge2[3] = {-130.0f, 0.0f, 0.0f};
    float normal[3] = {0.0f, -1.0f, 0.0f};
    float radiance[3] = {17.0f, 12.0f, 4.0f};
    float area = 13650.0f;  // mm^2, |edge1 x edge2|
};

struct CornellScene {
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    std::vector<std::string> materialNames;
    std::vector<std::string> objectNames;   // per triangle: "floor", "light", "tall_block", ...
    std::vector<float> positions;           // 9 floats per triangle
    CornellCamera camera;
    CornellLight light;
};

namespace detail {
inline bool ReadMtl(const std::string& path, CornellScene& s) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "LoadCornellScene: cannot open %s\n", path.c_str()); return false; }
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "newmtl") {
            std::string name; ss >> name;
            s.materialNames.push_back(name);
            s.materials.push_back(Material{});
        } else if ((tag == "Kd" || tag == "Ke") && !s.materials.empty()) {
            float* dst = tag == "Kd" ? s.materials.back().albedo : s.materials.back().emission;
            ss >> dst[0] >> dst[1] >> dst[2];
        }
    }
    return !s.materials.empty();
}
}  // namespace detail

// sceneDir — directory with cornell_box.obj and cornell_box.mtl.
inline bool LoadCornellScene(const std::string& sceneDir, CornellScene& s) {
    s = CornellScene{};
    if (!detail::ReadMtl(sceneDir + "/cornell_box.mtl", s)) return false;
    std::map<std::string, uint32_t> matIndex;
    for (uint32_t i = 0; i < s.materialNames.size(); ++i) matIndex[s.materialNames[i]] = i;

    const std::string objPath = sceneDir + "/cornell_box.obj";
    std::ifstream in(objPath);
    if (!in) { std::fprintf(stderr, "LoadCornellScene: cannot open %s\n", objPath.c_str()); return false; }
    std::vector<std::array<float, 3>> pos, nrm;
    std::string line, object = "?";
    uint32_t material = 0;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") { std::array<float, 3> p{}; ss >> p[0] >> p[1] >> p[2]; pos.push_back(p); }
        else if (tag == "vn") { std::array<float, 3> n{}; ss >> n[0] >> n[1] >> n[2]; nrm.push_back(n); }
        else if (tag == "o") { ss >> object; }
        else if (tag == "usemtl") {
            std::string name; ss >> name;
            auto it = matIndex.find(name);
            if (it == matIndex.end()) { std::fprintf(stderr, "LoadCornellScene: unknown material %s\n", name.c_str()); return false; }
            material = it->second;
        } else if (tag == "f") {
            // Faces are triangles in the form "v//vn v//vn v//vn".
            int vi[3], ni[3];
            std::string tok[3];
            ss >> tok[0] >> tok[1] >> tok[2];
            for (int k = 0; k < 3; ++k)
                if (std::sscanf(tok[k].c_str(), "%d//%d", &vi[k], &ni[k]) != 2 || vi[k] < 1 || vi[k] > (int)pos.size() ||
                    ni[k] < 1 || ni[k] > (int)nrm.size()) {
                    std::fprintf(stderr, "LoadCornellScene: unsupported face '%s'\n", line.c_str());
                    return false;
                }
            Triangle t{};
            for (int c = 0; c < 3; ++c) {
                t.v0[c] = pos[vi[0] - 1][c];
                t.v1[c] = pos[vi[1] - 1][c];
                t.v2[c] = pos[vi[2] - 1][c];
                t.normal[c] = nrm[ni[0] - 1][c];
            }
            t.material = material;
            s.triangles.push_back(t);
            s.objectNames.push_back(object);
            for (const float* v : {t.v0, t.v1, t.v2}) s.positions.insert(s.positions.end(), v, v + 3);
        }
    }
    if (s.triangles.empty()) { std::fprintf(stderr, "LoadCornellScene: no triangles\n"); return false; }
    return true;
}

}  // namespace bhs
