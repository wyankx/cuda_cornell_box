// trace.cuh — Cornell box tracing kernel for test task 02 (CUDA track).
// Included by main.cu. Compute path: your own ray-triangle intersection and a
// loop over all 32 triangles (no BVH needed for a scene this small).
#pragma once

#include "vec3.cuh"
#include <cuda/std/limits>
#include <vector_functions.h>

const float epsilon = cuda::std::numeric_limits<float>::epsilon() * 10;
const float off = 10 * epsilon;
const float float_max = cuda::std::numeric_limits<float>::max();

// Must match bhs::Triangle / bhs::Material (common/cornell_scene.h): float3 in
// CUDA is 12 bytes with 4-byte alignment, so the layouts are identical.
struct Triangle {
    float3 v0, v1, v2;
    float3 normal;          // geometric normal (walls point inside, blocks outside)
    unsigned int material;  // index into materials[]
};
static_assert(sizeof(Triangle) == 52, "Triangle layout");

struct Material {
    float3 albedo;
    float3 emission;
};
static_assert(sizeof(Material) == 24, "Material layout");

// Filled by main.cu (MakeParams): camera basis and light, passed by value.
struct FrameParams {
    unsigned int width, height, frameIndex, triangleCount;
    float3 camPos;
    float3 camForward;
    float3 camRight;       // image "right" = world -X here, see scene.json
    float3 camUp;
    float tanHalfFovY, aspect;
    float3 lightCorner;    // light point = corner + u * edge1 + v * edge2, u, v in [0, 1]
    float3 lightEdge1, lightEdge2;
    float3 lightNormal;    // one-sided light, emits along the normal (down)
    float3 lightRadiance;
    float lightArea;
};

struct Hit {
    bool hit;
    float t;
    unsigned int prim;     // index into triangles[]
};

// ---------------------------------------------------------------------------
// Random numbers (provided). A different seed per pixel AND per frame is what
// makes accumulation converge: think about what happens otherwise.
// ---------------------------------------------------------------------------
__device__ inline unsigned int pcg_hash(unsigned int v) {
    unsigned int state = v * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ inline float rand01(unsigned int& state) {
    state = pcg_hash(state);
    return float(state >> 8) * (1.0f / 16777216.0f);  // [0, 1)
}

// ---------------------------------------------------------------------------
// Intersection
// ---------------------------------------------------------------------------
// (intersection): ray-triangle intersection (e.g. Moller-Trumbore).
// Return true and the distance t if the ray hits the triangle with tMin < t < tMax.
__device__ inline bool intersect_triangle(float3 origin, float3 dir, float tMin, float tMax,
                                          const Triangle& tri, float& t)
{
    float3 e1 = tri.v1 - tri.v0;
    float3 e2 = tri.v2 - tri.v0;
    float3 pvec = cross(dir, e2);
    float det = dot(e1, pvec);

    if (abs(det) < epsilon) {
        t = 0.0f;
        return false;
    }

    float inv_det = 1 / det;
    float3 tvec = origin - tri.v0;
    float u = dot(tvec, pvec) * inv_det;
    if (u < -epsilon || u > 1 + epsilon) {
        t = 0.0f;
        return false;
    }

    float3 qvec = cross(tvec, e1);
    float v = dot(dir, qvec) * inv_det;
    if (v < -epsilon || u + v > 1 + epsilon) {
        return 0;
    }
    t = dot(e2, qvec) * inv_det;
    if (t < tMin || t > tMax) {
        t = 0.0f;
        return false;
    }
    return true;
}

// (intersection): closest hit along the ray — loop over all
// params.triangleCount triangles and keep the nearest one.
__device__ inline Hit intersect_scene(float3 origin, float3 dir, float tMin, float tMax,
                                      const FrameParams& params, const Triangle* triangles, bool stop_on_first = false)
{
    Hit h;
    h.hit = false; h.t = tMax; h.prim = 0;
    (void)origin; (void)dir; (void)tMin; (void)params; (void)triangles;
    for (int i = 0; i < params.triangleCount; ++i) {
        float t = 0;
        if (intersect_triangle(origin, dir, tMin, tMax, triangles[i], t) == false) {
            continue;
        }
        if (t < h.t) {
            h.hit = true;
            h.t = t;
            h.prim = i;
            if (stop_on_first) {
                return h;
            }
        }
    }
    return h;
}

// (direct light): visibility test for shadow rays. Any hit is enough — the
// nearest one is not needed, so the loop can stop at the first hit.
__device__ inline bool occluded(float3 origin, float3 dir, float tMin, float tMax,
                                const FrameParams& params, const Triangle* triangles)
{
    return intersect_scene(origin, dir, tMin, tMax, params, triangles, true).hit;
}

// ---------------------------------------------------------------------------
// One sample per pixel per launch.
// ---------------------------------------------------------------------------
__global__ void trace(FrameParams params, const Triangle* triangles,
                      const Material* materials, float4* accum)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= params.width || y >= params.height) return;
    const unsigned int pixel = y * params.width + x;
    unsigned int rng = pcg_hash(pixel ^ pcg_hash(params.frameIndex));

    // (primary rays): primary ray through pixel (x, y) with a random offset inside
    //           the pixel (jitter). Row 0 is the TOP of the image.
    float3 origin = params.camPos;
    float rx = (float)x + rand01(rng);
    float ry = (float)y + rand01(rng);
    float ndcx = rx * 2 / (float)params.width - 1;
    float ndcy = 1 - ry * 2 / (float)params.height;
    float3 dir = params.camForward + ndcx * params.tanHalfFovY * params.aspect * params.camRight + ndcy * params.tanHalfFovY * params.camUp;

    // (intersection): find the closest hit.
    // (direct light): radiance = emission (if the light is hit directly)
    //           + direct light from the area light via a shadow ray:
    //             random point on the light, visibility, both cosines, 1/d^2,
    //             pdf = 1 / area; Lambertian BRDF = albedo / pi (BHS_PI).
    //           Avoid self-intersection (tMin / offset along the normal).
    //           The shadow ray's tMax must stop slightly short of the light point,
    //           measured from the (offset) origin of the shadow ray.
    float3 radiance = make_float3(0.0f, 0.0f, 0.0f);
    Hit hit = intersect_scene(origin, dir, off, float_max, params, triangles);
    if (hit.hit) {
        radiance = materials[triangles[hit.prim].material].emission;
    }
    if (hit.hit && is_zero(radiance)) {
        float3 sorigin = origin + dir * hit.t;
        float3 target_point = params.lightCorner + rand01(rng) * params.lightEdge1 + rand01(rng) * params.lightEdge2;
        float3 sdir = target_point - sorigin;
        if (occluded(sorigin, sdir, off, 1 - off, params, triangles) == false) {
            const float3& hit_albedo = materials[triangles[hit.prim].material].albedo;
            const float3& hit_norm = triangles[hit.prim].normal;
            float3 sdir_n = normalize(sdir);
            float cos_x = max(0.f, dot(sdir_n, hit_norm));
            float cos_y = max(0.f, dot(-sdir_n, params.lightNormal));
            float sdir_lensq = dot(sdir, sdir);
            radiance = hit_albedo / BHS_PI * params.lightRadiance * cos_x * cos_y / sdir_lensq * params.lightArea;
        }
    }

    // (accumulation): accumulate. main.cu assumes accum holds the SUM of samples and
    //           divides by the number of frames when saving (change if you
    //           prefer a running average).
    float4 a = accum[pixel];
    accum[pixel] = make_float4(a.x + radiance.x, a.y + radiance.y, a.z + radiance.z, a.w + 1.0f);
}
