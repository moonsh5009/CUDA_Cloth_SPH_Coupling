#ifndef __PARTICLE_SAMPLING_CUH__
#define __PARTICLE_SAMPLING_CUH__

#pragma once
#include "ParticleSampling.h"
#include "../include/CUDA_Custom/DeviceManager.cuh"

inline __device__ uint getLineSamplingNum(float l, float d) {
	if (l - d * 1.01f > 0.f)
		return (uint)ceilf((l - d * 1.01f) / d);
	return 0.f;
}
inline __device__ float generateParticle(
	float3 x, uint ino, BoundaryParticleParam& param, uint& icurr)
{
	param._xs[icurr * 3u + 0u] = x.x;
	param._xs[icurr * 3u + 1u] = x.y;
	param._xs[icurr * 3u + 2u] = x.z;
	param._ws[icurr << 1u] = 10.0;
	param._inos[icurr++] = ino;
}
inline __device__ float generateLineParticle(
	float3 a, float3 b, float d, uint ino,
	BoundaryParticleParam& param, uint& icurr)
{
	b -= a;
	float l = Length(b);
	uint num = getLineSamplingNum(l, d);
	if (num > 0u) {
		d = l / (float)(num + 1u);
		b *= d / l;
		for (uint i = 0u; i < num; i++) {
			a += b;
			generateParticle(a, ino, param, icurr);
		}
	}
}
__global__ void compSamplingNum_kernel(
	ObjParam obj, RTriParam RTri,
	BoundaryParticleParam boundaryParticles,
	uint* shortEs, uint* sampNums, uint* prevInds, uint* currInds,
	bool* isGenerateds, bool* isApplied)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= obj._numFaces)
		return;

	bool isGenerated = false;
	uint ino = id * 3u;
	uint ino0 = obj._fs[ino + 0u];
	uint ino1 = obj._fs[ino + 1u];
	uint ino2 = obj._fs[ino + 2u];
	uint phase = obj._nodePhases[ino0];

	float3 ns[3];
	ino0 *= 3u;
	ns[0].x = obj._ns[ino0 + 0u];
	ns[0].y = obj._ns[ino0 + 1u];
	ns[0].z = obj._ns[ino0 + 2u];
	ino1 *= 3u;
	ns[1].x = obj._ns[ino1 + 0u];
	ns[1].y = obj._ns[ino1 + 1u];
	ns[1].z = obj._ns[ino1 + 2u];
	ino2 *= 3u;
	ns[2].x = obj._ns[ino2 + 0u];
	ns[2].y = obj._ns[ino2 + 1u];
	ns[2].z = obj._ns[ino2 + 2u];

	float3 es[3];
	float ls[3];
	uint ishort;
	ishort = 0u;
	es[0] = ns[1] - ns[0];
	es[1] = ns[2] - ns[1];
	es[2] = ns[0] - ns[2];
	ls[0] = Length(es[0]);
	ls[1] = Length(es[1]);
	ls[2] = Length(es[2]);
	if (ls[ishort] > ls[1])
		ishort = 1u;
	if (ls[ishort] > ls[2])
		ishort = 2u;

	float d = boundaryParticles._radii[phase];
	d += d;

	// Compute Vertex, Edge
	uint rtri = RTri._info[id];
	uint num = 0u;
	for (uint i = 0u; i < 3u; i++) {
		if (RTriVertex(rtri, i))
			num++;
		if (RTriEdge(rtri, i))
			num += getLineSamplingNum(ls[i], d);
	}

	// Compute Scanline
	ino0 = (ishort + 1u) % 3u;
	ino1 = (ishort + 2u) % 3u;
	ns[ino1] = ns[ishort];
	float3 s = Cross(es[ishort], Cross(es[ishort], es[ino1]));
	Normalize(s);
	es[ino1].x = -es[ino1].x;
	es[ino1].y = -es[ino1].y;
	es[ino1].z = -es[ino1].z;

	float l = Dot(s, es[ino0]);
	uint nt = getLineSamplingNum(l, d);
	if (nt > 0u) {
		float stride = (l / (float)(nt + 1u)) / l;
		for (uint i = 0u; i < nt; i++) {
			ns[ino0] += stride * es[ino0];
			ns[ino1] += stride * es[ino1];
			num += getLineSamplingNum(Length(ns[ino0] - ns[ino1]), d);
		}
	}

	if (id == 0u) {
		prevInds[0] = 0u;
		currInds[0] = 0u;
	}
	prevInds[id + 1u] = sampNums[id];
	currInds[id + 1u] = num;

	if (ishort != shortEs[id]) {
		shortEs[id] = ishort;
		isGenerated = true;
	}
	if (num != sampNums[id]) {
		sampNums[id] = num;
		isGenerated = true;
	}

	isGenerateds[id] = isGenerated;
	if (isGenerated)
		*isApplied = true;
}
__global__ void generateBoundaryParticle_kernel(
	ObjParam obj, RTriParam RTri,
	BoundaryParticleParam boundaryParticles,
	REAL* prevXs, REAL* prevWs, uint* prevInos, 
	uint* shortEs, uint* prevInds, uint* currInds,
	bool* isGenerateds)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= obj._numFaces)
		return;

	uint icurr = currInds[id];
	uint iend = currInds[id + 1u];
	if (isGenerateds[id]) {
		uint ino = id * 3u;
		uint ino0 = obj._fs[ino + 0u];
		uint ino1 = obj._fs[ino + 1u];
		uint ino2 = obj._fs[ino + 2u];
		uint phase = obj._nodePhases[ino0];

		float3 ns[3];
		ino0 *= 3u;
		ns[0].x = obj._ns[ino0 + 0u];
		ns[0].y = obj._ns[ino0 + 1u];
		ns[0].z = obj._ns[ino0 + 2u];
		ino1 *= 3u;
		ns[1].x = obj._ns[ino1 + 0u];
		ns[1].y = obj._ns[ino1 + 1u];
		ns[1].z = obj._ns[ino1 + 2u];
		ino2 *= 3u;
		ns[2].x = obj._ns[ino2 + 0u];
		ns[2].y = obj._ns[ino2 + 1u];
		ns[2].z = obj._ns[ino2 + 2u];

		float3 es[3];
		uint ishort = shortEs[id];
		es[0] = ns[1] - ns[0];
		es[1] = ns[2] - ns[1];
		es[2] = ns[0] - ns[2];
		float lshort = Length(es[ishort]);

		float d = boundaryParticles._radii[phase];
		d += d;

		// Compute Vertex, Edge
		uint rtri = RTri._info[id];
		uint num = 0u;
		for (uint i = 0u; i < 3u; i++) {
			if (RTriVertex(rtri, i))
				generateParticle(ns[i], id, boundaryParticles, icurr);
			if (RTriEdge(rtri, i))
				generateLineParticle(ns[i], ns[(i + 1u) % 3u], d, id, boundaryParticles, icurr);
		}

		// Compute Scanline
		ino0 = (ishort + 1u) % 3u;
		ino1 = (ishort + 2u) % 3u;
		ns[ino1] = ns[ishort];
		float3 s = Cross(es[ishort], Cross(es[ishort], es[ino1]));
		Normalize(s);
		es[ino1].x = -es[ino1].x;
		es[ino1].y = -es[ino1].y;
		es[ino1].z = -es[ino1].z;

		float l = Dot(s, es[ino0]);
		uint nt = getLineSamplingNum(l, d);
		if (nt > 0u) {
			float stride = (l / (float)(nt + 1u)) / l;
			for (uint i = 0u; i < nt; i++) {
				ns[ino0] += stride * es[ino0];
				ns[ino1] += stride * es[ino1];
				generateLineParticle(ns[ino0], ns[ino1], d, id, boundaryParticles, icurr);
			}
		}
	}
	else {
		uint iprev = prevInds[id];
		while (icurr < iend) {
			boundaryParticles._xs[icurr * 3u + 0u] = prevXs[iprev * 3u + 0u];
			boundaryParticles._xs[icurr * 3u + 1u] = prevXs[iprev * 3u + 1u];
			boundaryParticles._xs[icurr * 3u + 2u] = prevXs[iprev * 3u + 2u];
			boundaryParticles._ws[(icurr << 1u) + 0u] = prevWs[(iprev << 1u) + 0u];
			boundaryParticles._ws[(icurr << 1u) + 1u] = prevWs[(iprev << 1u) + 1u];
			boundaryParticles._inos[icurr++] = prevInos[iprev++];
		}
	}
}

__global__ void setBarycentric_kernel(
	ObjParam obj, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= boundaryParticles._numParticles)
		return;

	uint ino = boundaryParticles._inos[id];
	ino *= 3u;
	uint ino0 = obj._fs[ino + 0u];
	uint ino1 = obj._fs[ino + 1u];
	uint ino2 = obj._fs[ino + 2u];
	boundaryParticles._phases[id] = obj._nodePhases[ino0];

	ino = id << 1u;
	if (boundaryParticles._ws[ino] == 10.0) {
		float3 ns[3], x;
		ino0 *= 3u;
		ns[0].x = obj._ns[ino0 + 0u];
		ns[0].y = obj._ns[ino0 + 1u];
		ns[0].z = obj._ns[ino0 + 2u];
		ino1 *= 3u;
		ns[1].x = obj._ns[ino1 + 0u];
		ns[1].y = obj._ns[ino1 + 1u];
		ns[1].z = obj._ns[ino1 + 2u];
		ino2 *= 3u;
		ns[2].x = obj._ns[ino2 + 0u];
		ns[2].y = obj._ns[ino2 + 1u];
		ns[2].z = obj._ns[ino2 + 2u];

		ino = id * 3u;
		x.x = boundaryParticles._xs[ino + 0u];
		x.y = boundaryParticles._xs[ino + 1u];
		x.z = boundaryParticles._xs[ino + 2u];

		float w0 = 0.f;
		float w1 = 0.f;
		float3 n20 = ns[0] - ns[2];
		float3 n21 = ns[1] - ns[2];
		float t0 = Dot(n20, n20);
		float t1 = Dot(n21, n21);
		float t2 = Dot(n20, n21);
		float t3 = Dot(n20, x - ns[2]);
		float t4 = Dot(n21, x - ns[2]);
		float det = t0 * t1 - t2 * t2;
		if (fabs(det) > 1.0e-20f) {
			float invdet = 1.f / det;
			w0 = (+t1 * t3 - t2 * t4) * invdet;
			w1 = (-t2 * t3 + t0 * t4) * invdet;
		}

		if (w0 < 0.0)		w0 = 0.0;
		else if (w0 > 1.0)	w0 = 1.0;
		if (w1 < 0.0)		w1 = 0.0;
		else if (w1 > 1.0)	w1 = 1.0;
		ino = id << 1u;
		boundaryParticles._ws[ino + 0u] = w0;
		boundaryParticles._ws[ino + 1u] = w1;
	}
}

__global__ void compNodeWeights_kernel(
	ClothParam cloth, PoreParticleParam particles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= particles._numParticles)
		return;

	uint ino = id << 1u;
	REAL w0 = particles._ws[ino + 0u];
	REAL w1 = particles._ws[ino + 1u];
	REAL w2 = 1.0 - w0 - w1;

	ino = particles._inos[id];
	ino *= 3u;
	uint ino0 = cloth._fs[ino + 0u];
	uint ino1 = cloth._fs[ino + 1u];
	uint ino2 = cloth._fs[ino + 2u];
	
	atomicAdd_REAL(particles._nodeWeights + ino0, w0);
	atomicAdd_REAL(particles._nodeWeights + ino1, w1);
	atomicAdd_REAL(particles._nodeWeights + ino2, w2);
}
__global__ void lerpPosition_kernel(
	ObjParam obj, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= boundaryParticles._numParticles)
		return;

	uint ino = id << 1u;
	REAL w0 = boundaryParticles._ws[ino + 0u];
	REAL w1 = boundaryParticles._ws[ino + 1u];

	ino = boundaryParticles._inos[id];
	ino *= 3u;
	uint ino0 = obj._fs[ino + 0u];
	uint ino1 = obj._fs[ino + 1u];
	uint ino2 = obj._fs[ino + 2u];

	REAL3 xs[3];
	ino0 *= 3u; ino1 *= 3u; ino2 *= 3u;
	xs[0].x = obj._ns[ino0 + 0u]; xs[0].y = obj._ns[ino0 + 1u]; xs[0].z = obj._ns[ino0 + 2u];
	xs[1].x = obj._ns[ino1 + 0u]; xs[1].y = obj._ns[ino1 + 1u]; xs[1].z = obj._ns[ino1 + 2u];
	xs[2].x = obj._ns[ino2 + 0u]; xs[2].y = obj._ns[ino2 + 1u]; xs[2].z = obj._ns[ino2 + 2u];
	
	REAL3 x = xs[0] * w0 + xs[1] * w1 + (1.0 - w0 - w1) * xs[2];
	ino = id * 3u;
	boundaryParticles._xs[ino + 0u] = x.x;
	boundaryParticles._xs[ino + 1u] = x.y;
	boundaryParticles._xs[ino + 2u] = x.z;
}
__global__ void lerpVelocity_kernel(
	ObjParam obj, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= boundaryParticles._numParticles)
		return;

	uint ino = id << 1u;
	REAL w0 = boundaryParticles._ws[ino + 0u];
	REAL w1 = boundaryParticles._ws[ino + 1u];

	ino = boundaryParticles._inos[id];
	ino *= 3u;
	uint ino0 = obj._fs[ino + 0u];
	uint ino1 = obj._fs[ino + 1u];
	uint ino2 = obj._fs[ino + 2u];

	REAL3 vs[3];
	ino0 *= 3u; ino1 *= 3u; ino2 *= 3u;
	vs[0].x = obj._vs[ino0 + 0u]; vs[0].y = obj._vs[ino0 + 1u]; vs[0].z = obj._vs[ino0 + 2u];
	vs[1].x = obj._vs[ino1 + 0u]; vs[1].y = obj._vs[ino1 + 1u]; vs[1].z = obj._vs[ino1 + 2u];
	vs[2].x = obj._vs[ino2 + 0u]; vs[2].y = obj._vs[ino2 + 1u]; vs[2].z = obj._vs[ino2 + 2u];

	REAL3 v = vs[0] * w0 + vs[1] * w1 + (1.0 - w0 - w1) * vs[2];
	ino = id * 3u;
	boundaryParticles._vs[ino + 0u] = v.x;
	boundaryParticles._vs[ino + 1u] = v.y;
	boundaryParticles._vs[ino + 2u] = v.z;
}
__global__ void lerpForce_kernel(
	ObjParam cloth, PoreParticleParam poreParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= poreParticles._numParticles)
		return;

	uint ino = id << 1u;
	REAL w0 = poreParticles._ws[ino + 0u];
	REAL w1 = poreParticles._ws[ino + 1u];
	REAL w2 = 1.0 - w0 - w1;

	ino = poreParticles._inos[id];

	ino *= 3u;
	uint ino0 = cloth._fs[ino + 0u];
	uint ino1 = cloth._fs[ino + 1u];
	uint ino2 = cloth._fs[ino + 2u];
	REAL nodeWeight0 = poreParticles._nodeWeights[ino0];
	REAL nodeWeight1 = poreParticles._nodeWeights[ino1];
	REAL nodeWeight2 = poreParticles._nodeWeights[ino2];

	ino = id * 3u;
	REAL3 force;
	force.x = poreParticles._forces[ino + 0u];
	force.y = poreParticles._forces[ino + 1u];
	force.z = poreParticles._forces[ino + 2u];

	w0 *= w0 / nodeWeight0;
	w1 *= w1 / nodeWeight1;
	w2 *= w2 / nodeWeight2;
	ino0 *= 3u;
	ino1 *= 3u;
	ino2 *= 3u;
	atomicAdd_REAL(cloth._forces + ino0 + 0u, w0 * force.x);
	atomicAdd_REAL(cloth._forces + ino0 + 1u, w0 * force.y);
	atomicAdd_REAL(cloth._forces + ino0 + 2u, w0 * force.z);
	atomicAdd_REAL(cloth._forces + ino1 + 0u, w1 * force.x);
	atomicAdd_REAL(cloth._forces + ino1 + 1u, w1 * force.y);
	atomicAdd_REAL(cloth._forces + ino1 + 2u, w1 * force.z);
	atomicAdd_REAL(cloth._forces + ino2 + 0u, w2 * force.x);
	atomicAdd_REAL(cloth._forces + ino2 + 1u, w2 * force.y);
	atomicAdd_REAL(cloth._forces + ino2 + 2u, w2 * force.z);
}

#endif