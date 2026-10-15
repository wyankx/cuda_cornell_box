// vec3.cuh — small float3 / float4 helpers for the CUDA starter of test task 02.
//
// Helper code, not part of the task. CUDA has float3 but no arithmetic on it;
// these operators work both on the host and on the device.
#pragma once

#include <cuda_runtime.h>
#include <cmath>

#define BHS_HD __host__ __device__ inline
#define BHS_PI 3.14159265358979f

BHS_HD float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
BHS_HD float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
BHS_HD float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
BHS_HD float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
BHS_HD float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
BHS_HD float3 operator*(float s, float3 a) { return a * s; }
BHS_HD float3 operator/(float3 a, float s) { return a * (1.0f / s); }
BHS_HD float3& operator+=(float3& a, float3 b) { a = a + b; return a; }
BHS_HD float3& operator*=(float3& a, float3 b) { a = a * b; return a; }

BHS_HD float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
BHS_HD float3 cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
BHS_HD float length(float3 a) { return sqrtf(dot(a, a)); }
BHS_HD float3 normalize(float3 a) { return a / length(a); }
BHS_HD bool is_zero(float3 a) { return a.x == 0 && a.y == 0 && a.z == 0; }
