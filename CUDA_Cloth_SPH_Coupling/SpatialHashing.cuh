#ifndef __SPATIAL_HASHING_CUH__
#define __SPATIAL_HASHING_CUH__

#include "HashFunc.cuh"
#include "SpatialHashing.h"

inline __global__ void initHash_kernel(
	ParticleParam sphParticles, ParticleParam poreParticles, ParticleParam boundaryParticles,
	SpatialHashParam hash)
{
	uint id = threadIdx.x + blockIdx.x * blockDim.x;
	ParticleParam* particles = nullptr;
	uint ino = id;
	if (id < sphParticles._numParticles) {
		particles = &sphParticles;
	}
	else if (id < sphParticles._numParticles + poreParticles._numParticles) {
		particles = &poreParticles;
		ino -= sphParticles._numParticles;
	}
	else if (id < sphParticles._numParticles + poreParticles._numParticles + boundaryParticles._numParticles) {
		particles = &boundaryParticles;
		ino -= sphParticles._numParticles + poreParticles._numParticles;
	}
	else return;

	REAL3 x;
	getVector(particles->_xs, ino, x);

	int3 p = getGridPos(x, hash._radius);
	uint key = getGridIndex(p, hash._size);
	hash._keys[id] = key;
	hash._ids[id] = make_uint2(particles->_type, ino);
}
inline __global__ void initHashZindex_kernel(
	REAL3* xs, SpatialHashParam hash)
{
	uint id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id >= hash._numParticles)
		return;

	REAL3 x = xs[id];
	int3 p = getGridPos(x, hash._radius);
	uint key = getZindex(p, hash._size);
	hash._keys[id] = key;
	hash._ids[id] = make_uint2(TYPE_SPH_PARTICLE, id);
}

inline __global__ void reorderHash_kernel(
	SpatialHashParam hash)
{
	extern __shared__ uint s_hash[];
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	uint hashId;
	if (id < hash._numParticles) {
		hashId = hash._keys[id];
		s_hash[threadIdx.x + 1] = hashId;
		if (id > 0 && threadIdx.x == 0)
			s_hash[0] = hash._keys[id - 1];
	}
	__syncthreads();

	if (id < hash._numParticles) {
		uint prev_hashId = s_hash[threadIdx.x];
		if (id == 0 || prev_hashId != hashId) {
			hash._istarts[hashId] = id;
			if (id > 0)
				hash._iends[prev_hashId] = id;
		}
		if (id == hash._numParticles - 1)
			hash._iends[hashId] = hash._numParticles;
	}
}

inline __global__ void ParticleZSort_kernel(
	SPHParticleParam particles, REAL* oldXs, REAL* oldVs, REAL* oldSs, REAL* oldRelaxTs, uint* oldPhases, uint2* ids)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= particles._numParticles)
		return;

	uint ino = ids[id].y;
	uint phase = oldPhases[ino];
	REAL s = oldSs[ino];
	REAL relaxT = oldRelaxTs[ino];

	REAL3 x, v;
	getVector(oldXs, ino, x);
	getVector(oldVs, ino, v);

	particles._phases[id] = phase;
	particles._ss[id] = s;
	particles._relaxTs[id] = relaxT;
	setVector(particles._xs, id, x);
	setVector(particles._vs, id, v);
}

inline __global__ void compNumNeighbors_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles, 
	SpatialHashParam hash)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	ParticleParam* particlesi = nullptr;
	ParticleParam* particlesj = nullptr;
	if (id < sphParticles._numParticles) {
		particlesi = &sphParticles;
	}
	else if (id < sphParticles._numParticles + poreParticles._numParticles) {
		particlesi = &poreParticles;
		id -= sphParticles._numParticles;
	}
	else if (id < sphParticles._numParticles + poreParticles._numParticles + boundaryParticles._numParticles) {
		particlesi = &boundaryParticles;
		id -= sphParticles._numParticles + poreParticles._numParticles;
	}
	else return;

	if (id == 0u)
		particlesi->_ineis[0] = 0u;

	uint phase = particlesi->_phases[id];

	REAL3 xi, xj;
	getVector(particlesi->_xs, id, xi);
	
	REAL hi = particlesi->_radii[phase], hj;
	if (particlesi->_type == TYPE_SPH_PARTICLE) {
		REAL s = ((SPHParticleParam*)particlesi)->_ss[id];
		hi *= S3TO1(s);
	}
	REAL dist;

	uint numNeis = 0u;

	int3 gridPos = getGridPos(xi, hash._radius), nhpos;
	uint2 inei;
	uint ihash, istart, iend, i;
	for (nhpos.z = gridPos.z - 1; nhpos.z <= gridPos.z + 1; nhpos.z++) {
		for (nhpos.y = gridPos.y - 1; nhpos.y <= gridPos.y + 1; nhpos.y++) {
			for (nhpos.x = gridPos.x - 1; nhpos.x <= gridPos.x + 1; nhpos.x++) {
				ihash = getGridIndex(nhpos, hash._size);
				istart = hash._istarts[ihash];
				if (istart != 0xffffffff) {
					iend = hash._iends[ihash];
					for (i = istart; i < iend; i++) {
						inei = hash._ids[i];
						if (particlesi->_type != inei.x || inei.y != id) {
							if (inei.x == TYPE_SPH_PARTICLE)
								particlesj = &sphParticles;
							else if (inei.x == TYPE_PORE_PARTICLE)
								particlesj = &poreParticles;
							else
								particlesj = &boundaryParticles;

							phase = particlesj->_phases[inei.y];
							hj = particlesj->_radii[phase];
							if (inei.x == TYPE_SPH_PARTICLE) {
								REAL s = ((SPHParticleParam*)particlesj)->_ss[inei.y];
								hj *= S3TO1(s);
							}

							getVector(particlesj->_xs, inei.y, xj);

							if (hj < hi) hj = hi;
							hj *= SPH_RADIUS_RATIO;

							dist = LengthSquared(xi - xj);
							if (dist < hj * hj)
								numNeis++;
						}
					}
				}
			}
		}
	}

	particlesi->_ineis[id + 1u] = numNeis;
}
inline __global__ void getNeighbors_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles,
	SpatialHashParam hash)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	ParticleParam* particlesi = nullptr;
	ParticleParam* particlesj = nullptr;
	if (id < sphParticles._numParticles) {
		particlesi = &sphParticles;
	}
	else if (id < sphParticles._numParticles + poreParticles._numParticles) {
		particlesi = &poreParticles;
		id -= sphParticles._numParticles;
	}
	else if (id < sphParticles._numParticles + poreParticles._numParticles + boundaryParticles._numParticles) {
		particlesi = &boundaryParticles;
		id -= sphParticles._numParticles + poreParticles._numParticles;
	}
	else return;

	uint phase = particlesi->_phases[id];

	REAL3 xi, xj;
	getVector(particlesi->_xs, id, xi);

	REAL hi = particlesi->_radii[phase], hj;
	if (particlesi->_type == TYPE_SPH_PARTICLE) {
		REAL s = ((SPHParticleParam*)particlesi)->_ss[id];
		hi *= S3TO1(s);
	}
	REAL dist;

	uint icurr = particlesi->_ineis[id];

	int3 gridPos = getGridPos(xi, hash._radius), nhpos;
	uint2 inei;
	uint ihash, istart, iend, i;
	for (nhpos.z = gridPos.z - 1; nhpos.z <= gridPos.z + 1; nhpos.z++) {
		for (nhpos.y = gridPos.y - 1; nhpos.y <= gridPos.y + 1; nhpos.y++) {
			for (nhpos.x = gridPos.x - 1; nhpos.x <= gridPos.x + 1; nhpos.x++) {
				ihash = getGridIndex(nhpos, hash._size);
				istart = hash._istarts[ihash];
				if (istart != 0xffffffff) {
					iend = hash._iends[ihash];
					for (i = istart; i < iend; i++) {
						inei = hash._ids[i];
						if (particlesi->_type != inei.x || inei.y != id) {
							if (inei.x == TYPE_SPH_PARTICLE)
								particlesj = &sphParticles;
							else if (inei.x == TYPE_PORE_PARTICLE)
								particlesj = &poreParticles;
							else
								particlesj = &boundaryParticles;

							phase = particlesj->_phases[inei.y];
							hj = particlesj->_radii[phase];
							if (inei.x == TYPE_SPH_PARTICLE) {
								REAL s = ((SPHParticleParam*)particlesj)->_ss[inei.y];
								hj *= S3TO1(s);
							}

							getVector(particlesj->_xs, inei.y, xj);

							if (hj < hi) hj = hi;
							hj *= SPH_RADIUS_RATIO;

							dist = LengthSquared(xi - xj);
							if (dist < hj * hj)
								particlesi->_neis[icurr++] = inei;
						}
					}
				}
			}
		}
	}
}

#endif