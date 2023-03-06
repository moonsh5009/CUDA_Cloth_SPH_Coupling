#include "MeshObject.h"
#include "../include/CUDA_Custom/DeviceManager.cuh"

__global__ void compNormals_kernel(uint* fs, REAL* ns, REAL* fNorms, REAL* nNorms, uint numFaces) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numFaces)
		return;

	id *= 3u;
	uint iv0 = fs[id + 0u]; iv0 *= 3u;
	uint iv1 = fs[id + 1u]; iv1 *= 3u;
	uint iv2 = fs[id + 2u]; iv2 *= 3u;

	REAL3 v0, v1, v2;
	v0.x = ns[iv0 + 0u]; v0.y = ns[iv0 + 1u]; v0.z = ns[iv0 + 2u];
	v1.x = ns[iv1 + 0u]; v1.y = ns[iv1 + 1u]; v1.z = ns[iv1 + 2u];
	v2.x = ns[iv2 + 0u]; v2.y = ns[iv2 + 1u]; v2.z = ns[iv2 + 2u];

	REAL3 norm = Cross(v1 - v0, v2 - v0);
	Normalize(norm);

	fNorms[id + 0u] = norm.x;
	fNorms[id + 1u] = norm.y;
	fNorms[id + 2u] = norm.z;

	REAL radian = AngleBetweenVectors(v1 - v0, v2 - v0);
	//radian = 1.0;
	atomicAdd_REAL(nNorms + iv0 + 0u, norm.x * radian);
	atomicAdd_REAL(nNorms + iv0 + 1u, norm.y * radian);
	atomicAdd_REAL(nNorms + iv0 + 2u, norm.z * radian);

	radian = AngleBetweenVectors(v2 - v1, v0 - v1);
	//radian = 1.0;
	atomicAdd_REAL(nNorms + iv1 + 0u, norm.x * radian);
	atomicAdd_REAL(nNorms + iv1 + 1u, norm.y * radian);
	atomicAdd_REAL(nNorms + iv1 + 2u, norm.z * radian);

	radian = AngleBetweenVectors(v0 - v2, v1 - v2);
	//radian = 1.0;
	atomicAdd_REAL(nNorms + iv2 + 0u, norm.x * radian);
	atomicAdd_REAL(nNorms + iv2 + 1u, norm.y * radian);
	atomicAdd_REAL(nNorms + iv2 + 2u, norm.z * radian);
}
__global__ void nodeNormNormalize_kernel(REAL* nNorms, uint numNodes) {
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	id *= 3u;
	REAL3 norm;
	norm.x = nNorms[id + 0u];
	norm.y = nNorms[id + 1u];
	norm.z = nNorms[id + 2u];
	
	Normalize(norm);

	nNorms[id + 0u] = norm.x;
	nNorms[id + 1u] = norm.y;
	nNorms[id + 2u] = norm.z;
}