#ifndef __SIMULATION_CUH__
#define __SIMULATION_CUH__

#pragma once
#include "Simulation.h"
#include "../include/CUDA_Custom/DeviceManager.cuh"

__global__ void initMasses_kernel(ObjParam obj, REAL* masses, uchar* isFixeds) {
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= obj._numNodes)
		return;

	uchar isFixed = isFixeds[id];
	REAL m = masses[id];
	REAL invM = 1.0 / m;
	/*if (isFixed) {
		m = invM = 0.0;
	}*/
	obj._ms[id] = m;
	obj._invMs[id] = invM;
}
__global__ void initClothMasses_kernel(ClothParam cloth, REAL* masses, uchar* isFixeds) {
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= cloth._numNodes)
		return;

	uchar isFixed = isFixeds[id];
	uint phase = cloth._nodePhases[id];
	REAL mf = cloth._mfs[id];
	REAL m = masses[id];

	m += mf;
	REAL invM = 1.0 / m;
	/*if (isFixed) {
		m = invM = 0.0;
	}*/
	cloth._ms[id] = m;
	cloth._invMs[id] = invM;
}
__global__ void compGravityForce_kernel(REAL* forces, REAL* ms, REAL3 gravity, uint numNodes) {
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numNodes)
		return;

	uint ino = id * 3u;

	REAL m = ms[id];

	REAL3 f;
	f.x = forces[ino + 0u];
	f.y = forces[ino + 1u];
	f.z = forces[ino + 2u];

	f += m * gravity;

	forces[ino + 0u] = f.x;
	forces[ino + 1u] = f.y;
	forces[ino + 2u] = f.z;
}
__global__ void compRotationForce_kernel(
	REAL* ns, REAL* vs, REAL* forces, REAL* ms, uint* nodePhases, 
	REAL3* pivots, REAL3* degrees, REAL invdt, uint numNodes) 
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	uint phase = nodePhases[id];
	REAL3 pivot = pivots[phase];
	REAL3 degree = degrees[phase];

	REAL m = ms[id];
	id *= 3u;

	REAL3 force;
	force.x = forces[id + 0u];
	force.y = forces[id + 1u];
	force.z = forces[id + 2u];

	degree.x *= M_PI * 0.00555555555555555555555555555556;
	degree.y *= M_PI * 0.00555555555555555555555555555556;
	degree.z *= M_PI * 0.00555555555555555555555555555556;

	REAL cx = cos(degree.x);
	REAL sx = sin(degree.x);
	REAL cy = cos(degree.y);
	REAL sy = -sin(degree.y);
	REAL cz = cos(degree.z);
	REAL sz = sin(degree.z);

	REAL3 x, px;
	x.x = ns[id + 0u];
	x.y = ns[id + 1u];
	x.z = ns[id + 2u];
	x -= pivot;

	px.x = x.x * cz * cy + x.y * (cz * sy * sx - sz * cx) + x.z * (cz * sy * cx + sz * sx);
	px.y = x.x * sz * cy + x.y * (sz * sy * sx + cz * cx) + x.z * (sz * sy * cx - cz * sx);
	px.z = x.x * -sy + x.y * cy * sx + x.z * cy * cx;

	px = invdt * (px - x);
	x.x = vs[id + 0u];
	x.y = vs[id + 1u];
	x.z = vs[id + 2u];
	force += m * invdt * (px - x);
	forces[id + 0u] = force.x;
	forces[id + 1u] = force.y;
	forces[id + 2u] = force.z;
}

__global__ void applyForce_kernel(REAL* vs, REAL* forces, REAL* invMs, uchar* isFixeds, REAL dt, uint numNodes) {
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numNodes)
		return;

	uint ino = id * 3u;

	REAL invM = invMs[id];
	uchar isFixed = isFixeds[id];
	if (isFixed) invM = 0.0;

	REAL3 v, f;

	v.x = vs[ino + 0u];
	v.y = vs[ino + 1u];
	v.z = vs[ino + 2u];
	f.x = forces[ino + 0u];
	f.y = forces[ino + 1u];
	f.z = forces[ino + 2u];

	v += dt * invM * f;

	vs[ino + 0u] = v.x;
	vs[ino + 1u] = v.y;
	vs[ino + 2u] = v.z;
}
__global__ void updateVelocity_kernel(
	REAL* n0s, REAL* n1s, REAL* vs, REAL invdt, uint numNodes)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numNodes)
		return;

	uint ino = id * 3u;
	REAL3 n0, n1, v;

	n0.x = n0s[ino + 0u];
	n0.y = n0s[ino + 1u];
	n0.z = n0s[ino + 2u];
	n1.x = n1s[ino + 0u];
	n1.y = n1s[ino + 1u];
	n1.z = n1s[ino + 2u];

	v = invdt * (n1 - n0);

	vs[ino + 0u] = v.x;
	vs[ino + 1u] = v.y;
	vs[ino + 2u] = v.z;
}
__global__ void updatePosition_kernel(REAL* ns, REAL* vs, REAL dt, uint numNodes) {
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numNodes)
		return;

	uint ino = id * 3u;
	REAL3 n, v;

	n.x = ns[ino + 0u];
	n.y = ns[ino + 1u];
	n.z = ns[ino + 2u];
	v.x = vs[ino + 0u];
	v.y = vs[ino + 1u];
	v.z = vs[ino + 2u];

	n += dt * v;

	ns[ino + 0u] = n.x;
	ns[ino + 1u] = n.y;
	ns[ino + 2u] = n.z;
}

__global__ void initProject_kernel(
	REAL* ns, REAL* vs, REAL* ms, 
	REAL* Zs, REAL* Xs,
	REAL dt, REAL invdt2, uint numNodes)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numNodes)
		return;

	uint ino = id * 3u;
	REAL m = ms[id];
	REAL3 X, v;
	X.x = ns[ino + 0u];
	X.y = ns[ino + 1u];
	X.z = ns[ino + 2u];
	v.x = vs[ino + 0u];
	v.y = vs[ino + 1u];
	v.z = vs[ino + 2u];

	X += dt * v;
	Xs[ino + 0u] = X.x;
	Xs[ino + 1u] = X.y;
	Xs[ino + 2u] = X.z;

	X *= invdt2 * m;
	Zs[ino + 0u] = X.x;
	Zs[ino + 1u] = X.y;
	Zs[ino + 2u] = X.z;
}
__global__ void compErrorProject_kernel(
	REAL* Xs, REAL* newXs, REAL* Bs,
	uint* inos, REAL* ws, REAL* RLs, 
	uint* nodePhases, uint numSprings)
{
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id >= numSprings)
		return;

	uint ino = id << 1u;
	uint ino0 = inos[ino + 0u];
	uint ino1 = inos[ino + 1u];
	uint phase = nodePhases[ino0];

	REAL w = ws[phase];

	REAL3 x0, x1;
	x0.x = Xs[ino0 * 3u + 0u]; x0.y = Xs[ino0 * 3u + 1u]; x0.z = Xs[ino0 * 3u + 2u];
	x1.x = Xs[ino1 * 3u + 0u]; x1.y = Xs[ino1 * 3u + 1u]; x1.z = Xs[ino1 * 3u + 2u];

	REAL3 d = x0 - x1;
	REAL restLength = RLs[id];
	REAL length = Length(d);
	if (length > 1.0e-40) {
		REAL newL = restLength / length;
		d *= newL;
		x0 -= d;
		x1 += d;
	}
	REAL3 error0 = x1 * w;
	REAL3 error1 = x0 * w;

	atomicAdd_REAL(Bs + ino0, w);
	atomicAdd_REAL(Bs + ino1, w);
	ino0 *= 3u; ino1 *= 3u;
	atomicAdd_REAL(newXs + ino0 + 0u, error0.x);
	atomicAdd_REAL(newXs + ino0 + 1u, error0.y);
	atomicAdd_REAL(newXs + ino0 + 2u, error0.z);
	atomicAdd_REAL(newXs + ino1 + 0u, error1.x);
	atomicAdd_REAL(newXs + ino1 + 1u, error1.y);
	atomicAdd_REAL(newXs + ino1 + 2u, error1.z);
}
__global__ void updateXsProject_kernel(
	REAL* ms, uchar* isFixeds,
	REAL* Bs, REAL* Xs, REAL* prevXs, REAL* newXs,
	REAL underRelax, REAL omega, REAL invdt2,
	uint numNodes, REAL* maxError)
{
	extern __shared__ REAL s_maxError[];
	uint id = threadIdx.x + blockDim.x * blockIdx.x;
	uint ino;

	s_maxError[threadIdx.x] = 0.0;
	if (id < numNodes) {
		uchar isFixed = isFixeds[id];
		if (!isFixed > 0.0) {
			REAL m = ms[id];
			REAL b = Bs[id];
			ino = id * 3u;
			REAL3 X, prevX, newX;
			X.x = Xs[ino + 0u];
			X.y = Xs[ino + 1u];
			X.z = Xs[ino + 2u];

			if (b > 0.0) {
				prevX.x = prevXs[ino + 0u];
				prevX.y = prevXs[ino + 1u];
				prevX.z = prevXs[ino + 2u];
				newX.x = newXs[ino + 0u];
				newX.y = newXs[ino + 1u];
				newX.z = newXs[ino + 2u];

				newX *= 1.0 / (b + m * invdt2);
				newX = omega * (underRelax * (newX - X) + X - prevX) + prevX;

				Xs[ino + 0u] = newX.x;
				Xs[ino + 1u] = newX.y;
				Xs[ino + 2u] = newX.z;

				s_maxError[threadIdx.x] = Length(newX - X);
			}
			prevXs[ino + 0u] = X.x;
			prevXs[ino + 1u] = X.y;
			prevXs[ino + 2u] = X.z;
		}
	}
	for (ino = blockDim.x >> 1u; ino > 32u; ino >>= 1u) {
		__syncthreads();
		if (threadIdx.x < ino)
			if (s_maxError[threadIdx.x] < s_maxError[threadIdx.x + ino])
				s_maxError[threadIdx.x] = s_maxError[threadIdx.x + ino];
	}
	__syncthreads();
	if (threadIdx.x < 32u) {
		warpMax(s_maxError, threadIdx.x);
		if (threadIdx.x == 0u)
			atomicMax_REAL(maxError, s_maxError[0]);
	}
}

__global__ void Damping_kernel(
	REAL* vs, REAL w, uint numNodes)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numNodes)
		return;

	uint ino = id * 3u;
	REAL3 v;

	v.x = vs[ino + 0u];
	v.y = vs[ino + 1u];
	v.z = vs[ino + 2u];

	v *= w;

	vs[ino + 0u] = v.x;
	vs[ino + 1u] = v.y;
	vs[ino + 2u] = v.z;
}
__global__ void Damping_kernel(
	REAL* vs, uchar* isFixeds, REAL w, uint numNodes)
{
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= numNodes)
		return;
	if (!isFixeds[id]) {
		uint ino = id * 3u;
		REAL3 v;

		v.x = vs[ino + 0u];
		v.y = vs[ino + 1u];
		v.z = vs[ino + 2u];

		v *= w;

		vs[ino + 0u] = v.x;
		vs[ino + 1u] = v.y;
		vs[ino + 2u] = v.z;
	}
}

#endif