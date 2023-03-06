#ifndef __SPH_SOLVER_CUH__
#define __SPH_SOLVER_CUH__

#include "SPHKernel.cuh"
#include "SPHSolver.h"

__global__ void compBoundaryParticleVolume_kernel(
	BoundaryParticleParam boundaryParticles, PoreParticleParam poreParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= boundaryParticles._numParticles)
		return;

	uint phase = boundaryParticles._phases[id];
	REAL hi = boundaryParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO;
	REAL invhi = 1.0 / hi, invhj;

	REAL3 xi, xj;
	getVector(boundaryParticles._xs, id, xi);

	REAL3 grad;
	REAL w, dist;

	REAL volume = SPHKernel::WKernel(0.0, invhi);

	uint2 nei;
	researchNeighbors(boundaryParticles, id, nei,
		{
			if (nei.x != TYPE_SPH_PARTICLE) {
				if (nei.x == TYPE_PORE_PARTICLE) {
					phase = poreParticles._phases[nei.y];
					hj = poreParticles._radii[phase];
					getVector(poreParticles._xs, nei.y, xj);
				}
				else {
					phase = boundaryParticles._phases[nei.y];
					hj = boundaryParticles._radii[phase];
					getVector(boundaryParticles._xs, nei.y, xj);
				}

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;

				xj = xi - xj;
				dist = Length(xj);

				w = SPHKernel::WKernel(dist, invhi, invhj);
				volume += w;
			}
		});

	volume = 1.0 / volume;
	boundaryParticles._volumes[id] = volume;
}
__global__ void compPoreParticleVolume_kernel(
	PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= poreParticles._numParticles)
		return;

	uint phase = poreParticles._phases[id];
	REAL hi = poreParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO;
	REAL invhi = 1.0 / hi, invhj;

	REAL3 xi, xj;
	getVector(poreParticles._xs, id, xi);

	REAL w, dist;

	REAL volume = SPHKernel::WKernel(0.0, invhi);

	uint2 nei;
	researchNeighbors(poreParticles, id, nei,
		{
			if (nei.x != TYPE_SPH_PARTICLE) {
				if (nei.x == TYPE_PORE_PARTICLE) {
					phase = poreParticles._phases[nei.y];
					hj = poreParticles._radii[phase];
					getVector(poreParticles._xs, nei.y, xj);
				}
				else {
					phase = boundaryParticles._phases[nei.y];
					hj = boundaryParticles._radii[phase];
					getVector(boundaryParticles._xs, nei.y, xj);
				}

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				w = SPHKernel::WKernel(dist, invhi, invhj);
				volume += w;
			}
		});

	volume = 1.0 / volume;
	poreParticles._volumes[id] = volume;
}
__global__ void compDensity_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id], sj;

	REAL hi = sphParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO * S3TO1(si);
	REAL invhi = 1.0 / hi, invhj;

	REAL restDensityi = sphParticles._restDensities[phase];
	REAL volumei = sphParticles._restVolumes[phase], volumej;
	volumei *= si;

	REAL3 xi, xj;
	getVector(sphParticles._xs, id, xi);

	REAL w, dist;

	REAL densityi = volumei * SPHKernel::WKernel(0.0, invhi);

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];

				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);

				volumej = sphParticles._restVolumes[phase];
				volumej *= sj;
				getVector(sphParticles._xs, nei.y, xj);
			}
			else if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];
				volumej = poreParticles._volumes[nei.y];
				getVector(poreParticles._xs, nei.y, xj);
			}
			else {
				phase = boundaryParticles._phases[nei.y];
				hj = boundaryParticles._radii[phase];
				volumej = boundaryParticles._volumes[nei.y];
				getVector(boundaryParticles._xs, nei.y, xj);
			}

			hj *= SPH_RADIUS_RATIO;
			invhj = 1.0 / hj;
			xj = xi - xj;
			dist = Length(xj);

			w = SPHKernel::WKernel(dist, invhi, invhj);
			densityi += volumej * w;
		});


	densityi *= restDensityi;
	sphParticles._ds[id] = densityi;
}

__global__ void compDFSPHFactor_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id], sj;

	REAL hi = sphParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO * S3TO1(si);
	REAL invhi = 1.0 / hi, invhj;
	REAL restDensityi = sphParticles._restDensities[phase];

	REAL volumej;
	REAL3 xi, xj;
	getVector(sphParticles._xs, id, xi);

	REAL dist;

	REAL3 grads = make_REAL3(0.0), grad;
	REAL ai = 0.0;

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];

				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);

				volumej = sphParticles._restVolumes[phase];
				volumej *= sj;
				getVector(sphParticles._xs, nei.y, xj);
			}
			else if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];
				volumej = poreParticles._volumes[nei.y];
				getVector(poreParticles._xs, nei.y, xj);

				REAL w0 = poreParticles._ws[(nei.y << 1u) + 0u];
				REAL w1 = poreParticles._ws[(nei.y << 1u) + 1u];
				REAL w2 = 1.0 - w0 - w1;
				volumej *= 2.0 / (1.0 + w0 * w0 + w1 * w1 + w2 * w2);
			}
			else {
				phase = boundaryParticles._phases[nei.y];
				hj = boundaryParticles._radii[phase];
				volumej = boundaryParticles._volumes[nei.y];
				getVector(boundaryParticles._xs, nei.y, xj);
			}

			hj *= SPH_RADIUS_RATIO;
			invhj = 1.0 / hj;
			xj = xi - xj;
			dist = Length(xj);

			grad = volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
				
			if (nei.x == TYPE_SPH_PARTICLE)
				ai += LengthSquared(grad);
			grads += grad;
		});

	ai += LengthSquared(grads);
	if (ai > 1.0e-6)
		ai = 1.0 / ai;
	else ai = 0.0;

	sphParticles._as[id] = ai;
}
__global__ void compCDStiffness_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles,
	REAL dt, REAL* sumError)
{
	__shared__ REAL s_sumErrors[BLOCKSIZE];
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	s_sumErrors[threadIdx.x] = 0u;
	if (id < sphParticles._numParticles) {
		uint phase = sphParticles._phases[id];
		REAL si = sphParticles._ss[id], sj;

		REAL hi = sphParticles._radii[phase], hj;
		hi *= SPH_RADIUS_RATIO * S3TO1(si);
		REAL invhi = 1.0 / hi, invhj;

		REAL restDensityi = sphParticles._restDensities[phase];
		REAL di = sphParticles._ds[id];
		REAL ai = sphParticles._as[id];

		REAL volumej;
		REAL3 xi, xj;
		REAL3 vi, vj;
		getVector(sphParticles._xs, id, xi);
		getVector(sphParticles._vs, id, vi);

		REAL dist;

		REAL delta = 0.0;

		uint2 nei;
		researchNeighbors(sphParticles, id, nei,
			{
				if (nei.x == TYPE_SPH_PARTICLE) {
					phase = sphParticles._phases[nei.y];
					sj = sphParticles._ss[nei.y];

					hj = sphParticles._radii[phase];
					hj *= S3TO1(sj);

					volumej = sphParticles._restVolumes[phase];
					volumej *= sj;
					getVector(sphParticles._xs, nei.y, xj);
					getVector(sphParticles._vs, nei.y, vj);
				}
				else if (nei.x == TYPE_PORE_PARTICLE) {
					phase = poreParticles._phases[nei.y];
					hj = poreParticles._radii[phase];
					volumej = poreParticles._volumes[nei.y];
					getVector(poreParticles._xs, nei.y, xj);
					getVector(poreParticles._vs, nei.y, vj);
				}
				else {
					phase = boundaryParticles._phases[nei.y];
					hj = boundaryParticles._radii[phase];
					volumej = boundaryParticles._volumes[nei.y];
					getVector(boundaryParticles._xs, nei.y, xj);
					getVector(boundaryParticles._vs, nei.y, vj);
				}

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				vj = vi - vj;
				delta += volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * Dot(xj, vj);
			});


		REAL ki = di / restDensityi + dt * delta - 1.0;
		if (ki > 0.0) {
			s_sumErrors[threadIdx.x] = ki;
			ki *= ai / (dt * dt);
			sphParticles._ps[id] += ki;
		}
		else ki = 0.0;

		sphParticles._ks[id] = ki;
	}
	for (uint s = blockDim.x >> 1u; s > 32u; s >>= 1u) {
		__syncthreads();
		if (threadIdx.x < s)
			s_sumErrors[threadIdx.x] += s_sumErrors[threadIdx.x + s];
	}
	__syncthreads();
	if (threadIdx.x < 32u) {
		warpAdd(s_sumErrors, threadIdx.x);
		if (threadIdx.x == 0)
			atomicAdd_REAL(sumError, s_sumErrors[0]);
	}
}
__global__ void compDFStiffness_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles,
	REAL dt, REAL* sumError)
{
	__shared__ REAL s_sumErrors[BLOCKSIZE];
	uint id = blockDim.x * blockIdx.x + threadIdx.x;

	s_sumErrors[threadIdx.x] = 0u;
	if (id < sphParticles._numParticles) {
		uint phase = sphParticles._phases[id];
		REAL si = sphParticles._ss[id], sj;

		REAL hi = sphParticles._radii[phase], hj;
		hi *= SPH_RADIUS_RATIO * S3TO1(si);
		REAL invhi = 1.0 / hi, invhj;

		REAL restDensityi = sphParticles._restDensities[phase];
		REAL di = sphParticles._ds[id];
		REAL ai = sphParticles._as[id];

		REAL volumej;
		REAL3 xi, xj;
		REAL3 vi, vj;
		getVector(sphParticles._xs, id, xi);
		getVector(sphParticles._vs, id, vi);

		REAL dist;

		REAL delta = 0.0;

		uint2 nei;
		researchNeighbors(sphParticles, id, nei,
			{
				if (nei.x == TYPE_SPH_PARTICLE) {
					phase = sphParticles._phases[nei.y];
					sj = sphParticles._ss[nei.y];

					hj = sphParticles._radii[phase];
					hj *= S3TO1(sj);

					volumej = sphParticles._restVolumes[phase];
					volumej *= sj;
					getVector(sphParticles._xs, nei.y, xj);
					getVector(sphParticles._vs, nei.y, vj);
				}
				else if (nei.x == TYPE_PORE_PARTICLE) {
					phase = poreParticles._phases[nei.y];
					hj = poreParticles._radii[phase];
					volumej = poreParticles._volumes[nei.y];
					getVector(poreParticles._xs, nei.y, xj);
					getVector(poreParticles._vs, nei.y, vj);
				}
				else {
					phase = boundaryParticles._phases[nei.y];
					hj = boundaryParticles._radii[phase];
					volumej = boundaryParticles._volumes[nei.y];
					getVector(boundaryParticles._xs, nei.y, xj);
					getVector(boundaryParticles._vs, nei.y, vj);
				}

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				vj = vi - vj;
				delta += volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * Dot(xj, vj);
			});


		REAL ki = min(delta * dt, (di / restDensityi + dt * delta - 0.9));
		if (ki > 0.0) {
			s_sumErrors[threadIdx.x] = ki;
			ki *= ai / (dt * dt);
			sphParticles._ps[id] += ki;
		}
		else ki = 0.0;

		sphParticles._ks[id] = ki;
	}
	for (uint s = blockDim.x >> 1u; s > 32u; s >>= 1u) {
		__syncthreads();
		if (threadIdx.x < s) 
			s_sumErrors[threadIdx.x] += s_sumErrors[threadIdx.x + s];
	}
	__syncthreads();
	if (threadIdx.x < 32u) {
		warpAdd(s_sumErrors, threadIdx.x);
		if (threadIdx.x == 0) 
			atomicAdd_REAL(sumError, s_sumErrors[0]);
	}
}
__global__ void applyDFSPH_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id], sj;

	REAL hi = sphParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO * S3TO1(si);
	REAL invhi = 1.0 / hi, invhj;

	REAL restDensityi = sphParticles._restDensities[phase], restDensityj;
	REAL mi = sphParticles._masses[phase], mj;
	mi *= si;
	REAL ki = sphParticles._ks[id], kj;

	REAL volumej;
	REAL3 xi, xj;
	getVector(sphParticles._xs, id, xi);
	REAL3 forcei;
	getVector(sphParticles._forces, id, forcei);

	REAL3 forceij;
	REAL dist;

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];

				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);

				volumej = sphParticles._restVolumes[phase];
				volumej *= sj;

				restDensityj = sphParticles._restDensities[phase];
				getVector(sphParticles._xs, nei.y, xj);
				kj = sphParticles._ks[nei.y];
				kj = (ki + kj * restDensityi / restDensityj);
			}
			else if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];
				volumej = poreParticles._volumes[nei.y];
				getVector(poreParticles._xs, nei.y, xj);
				kj = ki;

				REAL w0 = poreParticles._ws[(nei.y << 1u) + 0u];
				REAL w1 = poreParticles._ws[(nei.y << 1u) + 1u];
				REAL w2 = 1.0 - w0 - w1;
				volumej *= 2.0 / (1.0 + w0 * w0 + w1 * w1 + w2 * w2);
			}
			else {
				phase = boundaryParticles._phases[nei.y];
				hj = boundaryParticles._radii[phase];
				volumej = boundaryParticles._volumes[nei.y];
				getVector(boundaryParticles._xs, nei.y, xj);
				kj = ki;
			}
			
			hj *= SPH_RADIUS_RATIO;
			invhj = 1.0 / hj;
			xj = xi - xj;
			dist = Length(xj);

			forceij = mi * volumej * kj * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
			if (nei.x == TYPE_PORE_PARTICLE)
				sumVector(poreParticles._forces, nei.y, forceij);

			forcei -= forceij;
		});

	setVector(sphParticles._forces, id, forcei);
}
__global__ void applyPressureForce_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id], sj;

	REAL hi = sphParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO * S3TO1(si);
	REAL invhi = 1.0 / hi, invhj;

	REAL restDensityi = sphParticles._restDensities[phase], restDensityj;
	REAL mi = sphParticles._masses[phase], mj;
	mi *= si;

	REAL pi = sphParticles._ps[id], pj;
	REAL relaxTi = sphParticles._relaxTs[id], relaxTj;
	pi *= relaxTi;

	REAL volumej;
	REAL3 xi, xj;
	getVector(sphParticles._xs, id, xi);
	REAL3 forcei;
	getVector(sphParticles._forces, id, forcei);

	REAL3 forceij;
	REAL dist;

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];

				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);

				volumej = sphParticles._restVolumes[phase];
				volumej *= sj;
				relaxTj = sphParticles._relaxTs[nei.y];

				restDensityj = sphParticles._restDensities[phase];
				getVector(sphParticles._xs, nei.y, xj);
				pj = sphParticles._ps[nei.y];
				pj = (pi + pj * relaxTj * restDensityi / restDensityj);
			}
			else if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];
				volumej = poreParticles._volumes[nei.y];
				getVector(poreParticles._xs, nei.y, xj);
				pj = pi;

				REAL w0 = poreParticles._ws[(nei.y << 1u) + 0u];
				REAL w1 = poreParticles._ws[(nei.y << 1u) + 1u];
				REAL w2 = 1.0 - w0 - w1;
				volumej *= 2.0 / (1.0 + w0 * w0 + w1 * w1 + w2 * w2);
			}
			else {
				phase = boundaryParticles._phases[nei.y];
				hj = boundaryParticles._radii[phase];
				volumej = boundaryParticles._volumes[nei.y];
				getVector(boundaryParticles._xs, nei.y, xj);
				pj = pi;
			}

			hj *= SPH_RADIUS_RATIO;
			invhj = 1.0 / hj;
			xj = xi - xj;
			dist = Length(xj);

			forceij = mi * volumej * pj * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
			if (nei.x == TYPE_PORE_PARTICLE)
				sumVector(poreParticles._forces, nei.y, forceij);

			forcei -= forceij;
		});

	setVector(sphParticles._forces, id, forcei);
}

__global__ void compGravityForce_kernel(SPHParticleParam sphParticles, REAL3 gravity) {
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id];
	REAL mi = sphParticles._masses[phase];
	mi *= si;

	REAL3 forcei;
	getVector(sphParticles._forces, id, forcei);

	forcei += mi * gravity;

	setVector(sphParticles._forces, id, forcei);
}
__global__ void compViscosityForce_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id], sj;

	REAL hi = sphParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO * S3TO1(si);
	REAL invhi = 1.0 / hi, invhj;

	REAL ki = sphParticles._viscosities[phase], kj;
	REAL restDensityi = sphParticles._restDensities[phase];
	REAL mi = sphParticles._masses[phase], mj;
	mi *= si;

	REAL3 xi, xj;
	REAL3 vi, vj;
	REAL di = sphParticles._ds[id], dj;
	getVector(sphParticles._xs, id, xi);
	getVector(sphParticles._vs, id, vi);

	REAL3 forcei;
	getVector(sphParticles._forces, id, forcei);

	REAL volumei = mi / di;
	const REAL Cons = 0.5 * 10.0 * volumei;

	REAL3 forceij;
	REAL dist, volumej;

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];
				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);

				mj = sphParticles._masses[phase];
				mj *= sj;

				kj = sphParticles._viscosities[phase];
				dj = sphParticles._ds[nei.y];
				getVector(sphParticles._xs, nei.y, xj);
				getVector(sphParticles._vs, nei.y, vj);
				volumej = mj / dj;
			}
			else if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];
				kj = poreParticles._viscosities[phase];
				getVector(poreParticles._xs, nei.y, xj);
				getVector(poreParticles._vs, nei.y, vj);
				volumej = poreParticles._volumes[nei.y];
				volumej = volumej * restDensityi / di;

				REAL w0 = poreParticles._ws[(nei.y << 1u) + 0u];
				REAL w1 = poreParticles._ws[(nei.y << 1u) + 1u];
				REAL w2 = 1.0 - w0 - w1;
				volumej *= 2.0 / (1.0 + w0 * w0 + w1 * w1 + w2 * w2);
			}
			else {
				phase = boundaryParticles._phases[nei.y];
				hj = boundaryParticles._radii[phase];

				kj = boundaryParticles._viscosities[phase];
				getVector(boundaryParticles._xs, nei.y, xj);
				getVector(boundaryParticles._vs, nei.y, vj);
				volumej = boundaryParticles._volumes[nei.y];
				volumej = volumej * restDensityi / di;
			}

			hj *= SPH_RADIUS_RATIO;
			invhj = 1.0 / hj;
			xj = xi - xj;
			dist = Length(xj);

			vj = vi - vj;
			forceij = Cons * (ki + kj) * volumej * Dot(xj, vj) *
				SPHKernel::LaplacianKernel(dist, invhi, invhj) / dist * xj;
			/*forceij = Cons * (ki + kj) * volumej * 
				SPHKernel::LaplacianKernel(dist, invhi, invhj) * dist * vj;*/

			if (nei.x == TYPE_PORE_PARTICLE)
				sumVector(poreParticles._forces, nei.y, forceij);

			forcei -= forceij;
		});


	setVector(sphParticles._forces, id, forcei);
}
#if 1
__global__ void compColorNormal_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id], sj;

	REAL hi = sphParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO * S3TO1(si);
	REAL invhi = 1.0 / hi, invhj;

	REAL di = sphParticles._ds[id], dj;
	REAL3 xi, xj;
	getVector(sphParticles._xs, id, xi);

	REAL dist, mj, volumej;

	REAL3 norm = make_REAL3(0.0);

	uint2 nei;

	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];
				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);

				mj = sphParticles._masses[phase];
				mj *= sj;

				dj = sphParticles._ds[nei.y];
				getVector(sphParticles._xs, nei.y, xj);
				volumej = mj / dj;

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				norm += //(hi + hj) * 0.5 * 
					volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
			}
		});

	norm *= hi;
	setVector(sphParticles._ns, id, norm);
}
__global__ void compSurfaceTensionForce_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id], sj;

	REAL hi = sphParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO * S3TO1(si);
	REAL invhi = 1.0 / hi, invhj;

	REAL ki = sphParticles._surfaceTensions[phase], kj;
	REAL restDensityi = sphParticles._restDensities[phase];
	ki *= restDensityi + restDensityi;

	REAL mi = sphParticles._masses[phase], mj;
	mi *= si;

	REAL3 xi, xj;
	REAL di = sphParticles._ds[id], dj;
	getVector(sphParticles._xs, id, xi);

	REAL3 ni, nj;
	getVector(sphParticles._ns, id, ni);

	REAL3 forcei;
	getVector(sphParticles._forces, id, forcei);

	REAL3 forceij;
	REAL dist;

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];
				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);
				
				mj = sphParticles._masses[phase];
				mj *= sj;

				dj = sphParticles._ds[nei.y];

				if (sphParticles._relaxTs[nei.y])
					dj = restDensityi;

				getVector(sphParticles._xs, nei.y, xj);
				getVector(sphParticles._ns, nei.y, nj);
			}
			else if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];

				kj = poreParticles._surfaceTensions[phase];
				getVector(poreParticles._xs, nei.y, xj);
				mj = poreParticles._volumes[nei.y];
				mj *= restDensityi;

				REAL w0 = poreParticles._ws[(nei.y << 1u) + 0u];
				REAL w1 = poreParticles._ws[(nei.y << 1u) + 1u];
				REAL w2 = 1.0 - w0 - w1;
				mj *= 2.0 / (1.0 + w0 * w0 + w1 * w1 + w2 * w2);
			}
			else {
				phase = boundaryParticles._phases[nei.y];
				hj = boundaryParticles._radii[phase];

				kj = boundaryParticles._surfaceTensions[phase];
				getVector(boundaryParticles._xs, nei.y, xj);
				mj = boundaryParticles._volumes[nei.y];
				mj *= restDensityi;
			}
			
			hj *= SPH_RADIUS_RATIO;
			invhj = 1.0 / hj;
			xj = xi - xj;
			dist = Length(xj);

			if (nei.x == TYPE_SPH_PARTICLE) {
				nj = ni - nj;
				//nj = hj / hi * ni - nj;
				//nj = (hi + hj) / (hi + hi) * ni - (hi + hj) / (hj + hj) * nj;
				REAL3 cohesion = mi * mj * SPHKernel::cohesionKernel(dist, invhi, invhj) / dist * xj;
				REAL3 curvature = mi * nj;
				forceij = ki / (di + dj) * (cohesion + curvature);
			}
			else {
				forceij = kj * mi * mj * 
					SPHKernel::addhesionKernel(dist, invhi, invhj) / dist * xj;
				if (nei.x == TYPE_PORE_PARTICLE)
					sumVector(poreParticles._forces, nei.y, forceij);
			}

			forcei -= forceij;
		});


	setVector(sphParticles._forces, id, forcei);
}
#else
__global__ void compColorField_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id], sj;
	REAL hi = sphParticles._radii[phase], hj;
	hi *= S3TO1(si) * SPH_RADIUS_RATIO;
	REAL invhi = 1.0 / hi, invhj;

	REAL restDensityi = sphParticles._restDensities[phase];
	REAL mi = sphParticles._masses[phase], mj;
	mi *= si;

	REAL di = sphParticles._ds[id], dj;
	REAL3 xi, xj;
	getVector(sphParticles._xs, id, xi);

	REAL volumei = di / mi, volumej;
	//REAL volumei = sphParticles._restVolumes[phase], volumej;

	REAL dist;

	REAL ci = mi / di * SPHKernel::WKernel(0.0, invhi * 0.5);

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];
				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);

				mj = sphParticles._masses[phase];
				mj *= sj;
				dj = sphParticles._ds[nei.y];
				getVector(sphParticles._xs, nei.y, xj);
				volumej = mj / dj;
				//volumej = sphParticles._restVolumes[phase];
			}
			else if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];

				getVector(poreParticles._xs, nei.y, xj);
				volumej = poreParticles._volumes[nei.y];
			}
			else {
				phase = boundaryParticles._phases[nei.y];
				hj = boundaryParticles._radii[phase];

				getVector(boundaryParticles._xs, nei.y, xj);
				volumej = boundaryParticles._volumes[nei.y];
			}

			hj *= SPH_RADIUS_RATIO;
			invhj = 1.0 / hj;

			xj = xi - xj;
			dist = Length(xj);
			ci += volumej * SPHKernel::WKernel(dist, invhi * 0.5, invhj * 0.5);
		});

	sphParticles._ns[id] = ci;
}
__global__ void compColorGradient_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id], sj;
	REAL hi = sphParticles._radii[phase], hj;
	hi *= S3TO1(si) * SPH_RADIUS_RATIO;
	REAL invhi = 1.0 / hi, invhj;

	REAL dj, mj;
	REAL3 xi, xj;
	getVector(sphParticles._xs, id, xi);
	REAL ci = sphParticles._ns[id], cj;

	REAL dist, volumej;

	REAL3 grads = make_REAL3(0.0);

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];
				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);

				mj = sphParticles._masses[phase];
				mj *= sj;
				dj = sphParticles._ds[nei.y];
				cj = sphParticles._ns[nei.y];
				getVector(sphParticles._xs, nei.y, xj);
				volumej = mj / dj;
				//volumej = sphParticles._restVolumes[phase];

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				grads += cj * volumej * SPHKernel::GKernel(dist, invhi * 0.5, invhj * 0.5) / dist * xj;
			}
		});

	REAL k = LengthSquared(grads) / (ci * ci + FLT_EPSILON);
	sphParticles._ks[id] = k;
}
__global__ void compSurfaceTensionForce_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	const REAL atm = 10.0;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id], sj;
	REAL hi = sphParticles._radii[phase], hj;
	hi *= S3TO1(si) * SPH_RADIUS_RATIO;
	REAL invhi = 1.0 / hi, invhj;

	REAL ki = sphParticles._surfaceTensions[phase], kj;
	REAL restDensityi = sphParticles._restDensities[phase];
	REAL mi = sphParticles._masses[phase], mj;
	mi *= si;

	REAL3 xi, xj;
	REAL di = sphParticles._ds[id], dj;
	getVector(sphParticles._xs, id, xi);

	REAL c2i = sphParticles._ks[id], c2j;
	REAL volumei = mi / di, volumej;
	//REAL volumei = sphParticles._restVolumes[phase], volumej;

	REAL3 forcei;
	getVector(sphParticles._forces, id, forcei);

	REAL3 forceij;
	REAL dist;

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];
				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);

				kj = sphParticles._surfaceTensions[phase];
				mj = sphParticles._masses[phase];
				mj *= sj;
				dj = sphParticles._ds[nei.y];
				c2j = sphParticles._ks[nei.y];
				getVector(sphParticles._xs, nei.y, xj);
				volumej = mj / dj;
				//volumej = sphParticles._restVolumes[phase];
				c2j += c2i;
			}
			else if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];

				kj = poreParticles._surfaceTensions[phase];
				getVector(poreParticles._xs, nei.y, xj);
				volumej = poreParticles._volumes[nei.y];
				c2j = c2i;

				REAL w0 = poreParticles._ws[(nei.y << 1u) + 0u];
				REAL w1 = poreParticles._ws[(nei.y << 1u) + 1u];
				REAL w2 = 1.0 - w0 - w1;
				volumej *= 2.0 / (1.0 + w0 * w0 + w1 * w1 + w2 * w2);
			}
			else {
				phase = boundaryParticles._phases[nei.y];
				hj = boundaryParticles._radii[phase];

				kj = boundaryParticles._surfaceTensions[phase];
				getVector(boundaryParticles._xs, nei.y, xj);
				volumej = boundaryParticles._volumes[nei.y];
				c2j = c2i;
			}

			hj *= SPH_RADIUS_RATIO;
			invhj = 1.0 / hj;

			xj = xi - xj;
			dist = Length(xj);

			forceij = -volumei * volumej * (0.25 * kj * c2j + atm) * SPHKernel::GKernel(dist, invhi * 0.5, invhj * 0.5) / dist * xj;

			if (nei.x == TYPE_PORE_PARTICLE)
				sumVector(poreParticles._forces, nei.y, forceij);

			forcei -= forceij;
		});


	setVector(sphParticles._forces, id, forcei);
}
#endif

__global__ void applyForce_kernel(SPHParticleParam sphParticles, REAL dt) {
	uint id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id];
	REAL mi = sphParticles._masses[phase];
	mi *= si;
	
	REAL3 vi, forcei;
	getVector(sphParticles._vs, id, vi);
	getVector(sphParticles._forces, id, forcei);

	vi += dt / mi * forcei;

	setVector(sphParticles._vs, id, vi);
}

#endif