//#ifndef __POROUS_SOLVER_CUH__
//#define __POROUS_SOLVER_CUH__
//
//#pragma once
//#include "PorousSolver.h"
//#include "SPHKernel.cuh"
//#include "../include/CUDA_Custom/DeviceManager.cuh"
//
//#define FIXED_VOLUME		0
//#define VOLUME_TEST			1.0
//
//__global__ void massStabilization_kernel(ClothParam cloth)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= cloth._numNodes)
//		return;
//
//	REAL mf = cloth._mfs[id];
//	if (mf < 1.0e-40)
//		mf = 0.0;
//}
//__global__ void lerpPoreFactorToParticle_kernel(
//	ClothParam cloth, REAL* norms, PoreParticleParam particles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= particles._numParticles)
//		return;
//
//	uint ino = id << 1u;
//	REAL w0 = particles._ws[ino + 0u];
//	REAL w1 = particles._ws[ino + 1u];
//	REAL w2 = 1.0 - w0 - w1;
//
//	ino = particles._inos[id];
//	ino *= 3u;
//	uint ino0 = cloth._fs[ino + 0u];
//	uint ino1 = cloth._fs[ino + 1u];
//	uint ino2 = cloth._fs[ino + 2u];
//	REAL3 norm;
//	norm.x = norms[ino + 0u];
//	norm.y = norms[ino + 1u];
//	norm.z = norms[ino + 2u];
//
//	REAL mf0 = cloth._mfs[ino0];
//	REAL mf1 = cloth._mfs[ino1];
//	REAL mf2 = cloth._mfs[ino2];
//	REAL nodeWeight0 = particles._nodeWeights[ino0];
//	REAL nodeWeight1 = particles._nodeWeights[ino1];
//	REAL nodeWeight2 = particles._nodeWeights[ino2];
//
//	ino = id * 3u;
//	particles._norms[ino + 0u] = norm.x;
//	particles._norms[ino + 1u] = norm.y;
//	particles._norms[ino + 2u] = norm.z;
//
//	//REAL mf = w0 * mf0 / nodeWeight0 + w1 * mf1 / nodeWeight1 + w2 * mf2 / nodeWeight2;
//	REAL mf = w0 * mf0 + w1 * mf1 + w2 * mf2;
//	particles._mfs[id] = mf;
//}
//__global__ void lerpPoreFactorToObject_kernel(
//	ClothParam cloth, PoreParticleParam particles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= particles._numParticles)
//		return;
//
//	uint ino = id << 1u;
//	REAL w0 = particles._ws[ino + 0u];
//	REAL w1 = particles._ws[ino + 1u];
//	REAL w2 = 1.0 - w0 - w1;
//
//	uint phase = particles._phases[id];
//	REAL restSolidFraction = particles._restSolidFractions[phase];
//	REAL restFluidDensity = particles._restFluidDensities[phase];
//
//	//REAL volume = particles._volumes[id];
//	REAL volume = particles._restVolumes[phase] * VOLUME_TEST;
//
//	REAL fluidMass0 = (1.0 - restSolidFraction) * volume * restFluidDensity;
//
//	ino = particles._inos[id];
//	ino *= 3u;
//	uint ino0 = cloth._fs[ino + 0u];
//	uint ino1 = cloth._fs[ino + 1u];
//	uint ino2 = cloth._fs[ino + 2u];
//	REAL nodeWeight0 = particles._nodeWeights[ino0];
//	REAL nodeWeight1 = particles._nodeWeights[ino1];
//	REAL nodeWeight2 = particles._nodeWeights[ino2];
//	atomicAdd_REAL(cloth._maxFluidMass + ino0, w0 * fluidMass0 / nodeWeight0);
//	atomicAdd_REAL(cloth._maxFluidMass + ino1, w1 * fluidMass0 / nodeWeight1);
//	atomicAdd_REAL(cloth._maxFluidMass + ino2, w2 * fluidMass0 / nodeWeight2);
//	atomicAdd_REAL(cloth._restSolidFractions + ino0, w0 * restSolidFraction / nodeWeight0);
//	atomicAdd_REAL(cloth._restSolidFractions + ino1, w1 * restSolidFraction / nodeWeight1);
//	atomicAdd_REAL(cloth._restSolidFractions + ino2, w2 * restSolidFraction / nodeWeight2);
//
//	REAL s = particles._ss[id];
//	atomicAdd_REAL(cloth._ss + ino0, s * w0 / nodeWeight0);
//	atomicAdd_REAL(cloth._ss + ino1, s * w1 / nodeWeight1);
//	atomicAdd_REAL(cloth._ss + ino2, s * w2 / nodeWeight2);
//}
//
//__global__ void initPoreFactor_kernel(
//	PoreParticleParam particles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= particles._numParticles)
//		return;
//
//	uint ino = id * 3u;
//	REAL fd = 100.0;
//	REAL rb = 31.0;
//
//	REAL visc = 0.89;
//	REAL surft = 72.0;
//
//	uint phase = particles._phases[id];
//	REAL sf = particles._restSolidFractions[phase];
//	REAL restFluidDensity = particles._restFluidDensities[phase];
//
//	REAL volume = particles._restVolumes[phase];
//	//REAL volume = particles._volumes[id];
//
//	REAL fluidMass = particles._mfs[id];
//	REAL fluidMass0 = (1.0 - sf) * volume * restFluidDensity;
//
//	REAL cc = 1.6;
//	REAL s = fluidMass / fluidMass0;
//
//	REAL kA = (-log(sf) - 1.476 + sf * (2.0 - sf * 0.5)) /
//		(16.0 * sf) * fd * fd;
//	REAL kB = (-log(sf) - 1.476 + sf * (2.0 - sf * (1.774 - 4.078 * sf))) /
//		(32.0 * sf) * fd * fd;
//
//	/*REAL cA = 1.75 / sqrt(150.0) * pow(restFluidDensity * s, cc) * pow(fd, cc - 1.0) * pow(visc, 1.0 - cc) /
//		(pow(1.0 - sf, 3.0 / 2.0) * sqrt(kA));
//	REAL cB = 1.75 / sqrt(150.0) * pow(restFluidDensity * s, cc) * pow(fd, cc - 1.0) * pow(visc, 1.0 - cc) /
//		(pow(1.0 - sf, 3.0 / 2.0) * sqrt(kB));*/
//	REAL cA = 1.75 / sqrt(150.0) * restFluidDensity / (pow(1.0 - sf, 3.0 / 2.0) * sqrt(kA));
//	REAL cB = 1.75 / sqrt(150.0) * restFluidDensity / (pow(1.0 - sf, 3.0 / 2.0) * sqrt(kB));
//
//	REAL pA = 2.0 * surft * sf * cos(40.8 * M_PI / 180.0) / ((1.0 - sf) * rb);
//	REAL pB = 0.5 * pA;
//
//	particles._Ks[ino + 0u] = kA;
//	particles._Ks[ino + 1u] = kA;
//	particles._Ks[ino + 2u] = kA;
//
//	particles._Cs[ino + 0u] = cA;
//	particles._Cs[ino + 1u] = cA;
//	particles._Cs[ino + 2u] = cA;
//
//	particles._Ps[ino + 0u] = pA;
//	particles._Ps[ino + 1u] = pA;
//	particles._Ps[ino + 2u] = pA;
//
//	particles._ss[id] = s;
//}
//__global__ void updateRelaxT_kernel(
//	SPHParticleParam sphParticles, REAL dt)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= sphParticles._numParticles)
//		return;
//
//	REAL relaxT = sphParticles._relaxTs[id];
//	if (relaxT != 1.0) {
//		REAL density = sphParticles._ds[id];
//		uint phase = sphParticles._phases[id];
//		REAL restDensity = sphParticles._restDensities[phase];
//		REAL s = sphParticles._ss[id];
//
//		if (relaxT < 1.0) {
//			relaxT += 1.0 * dt;
//
//			if (relaxT > 1.0)
//				relaxT = 1.0;
//			else {
//				REAL3 v;
//				getVector(sphParticles._vs, id, v);
//
//				REAL h = sphParticles._radii[phase];
//				h *= SPH_RADIUS_RATIO * S3TO1(s);
//
//				REAL lv = Length(v);
//				REAL maxV = h * 1.0 * (0.05 + relaxT * 0.95) / dt;
//				if (lv > maxV) {
//					v *= maxV / lv;
//					//setVector(sphParticles._vs, id, v);
//				}
//
//				/*if (density <= 0.2 * restDensity)
//					relaxT = 1.0;*/
//			}
//		}
//		else relaxT = 1.0;
//		sphParticles._relaxTs[id] = relaxT;
//	}
//}
//
//__global__ void compPoreVelocity_kernel(
//	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles, 
//	REAL3 gravity)
//{
//#if 0
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	REAL visc = 0.89;
//
//	uint phase = poreParticles._phases[id];
//	REAL si = poreParticles._ss[id], sj;
//	REAL hi = poreParticles._radii[phase], hj, invh;
//	REAL fluidDensity = poreParticles._restFluidDensities[id];
//	fluidDensity *= si;
//	REAL restSolidFraction = poreParticles._restSolidFractions[phase];
//
//	REAL3 xi, xj;
//	getVector(poreParticles._xs, id, xi);
//
//	REAL mj, dj, volumej, pj;
//	REAL dist;
//
//	REAL3 pc = make_REAL3(0.0);
//	REAL3 pp = make_REAL3(0.0);
//
//	uint2 nei;
//	researchNeighbors(id, poreParticles._neis, nei,
//		{
//			if (nei.x == TYPE_SPH_PARTICLE) {
//				phase = sphParticles._phases[nei.y];
//				sj = sphParticles._ss[nei.y];
//				hj = sphParticles._radii[phase];
//				hj *= S3TO1(sj);
//
//				mj = sphParticles._masses[phase];
//				mj *= sj;
//				dj = sphParticles._ds[nei.y];
//				volumej = mj / dj;
//
//				pj = sphParticles._ps[nei.y];
//				getVector(sphParticles._xs, nei.y, xj);
//
//				hj = (hi + hj) * SPH_RADIUS_RATIO * 0.5;
//				xj = xi - xj;
//				dist = Length(xj);
//				if (dist < hj) {
//					invh = 1.0 / hj;
//					pp -= pj * volumej * SPHKernel::GKernel(dist, invh) / dist * xj;
//				}
//			}
//			else if (nei.x == TYPE_PORE_PARTICLE) {
//				phase = poreParticles._phases[nei.y];
//				volumej = poreParticles._volumes[nei.y];
//				sj = poreParticles._ss[nei.y];
//				hj = poreParticles._radii[phase];
//				pj = poreParticles._Ps[nei.y * 3u + 0u];
//
//				getVector(poreParticles._xs, nei.y, xj);
//
//				hj = (hi + hj) * SPH_RADIUS_RATIO * 0.5;
//				xj = xi - xj;
//				dist = Length(xj);
//				if (dist < hj) {
//					invh = 1.0 / hj;
//					pc += max(1.0 - sj, 0.0) * pj * volumej * SPHKernel::GKernel(dist, invh) / dist * xj;
//				}
//			}
//			else {
//				/*phase = boundaryParticles._phases[nei.y];
//				hj = boundaryParticles._radii[phase];
//				volumej = boundaryParticles._volumes[nei.y];
//
//				getVector(boundaryParticles._xs, nei.y, xj);
//
//				hj = (hi + hj) * SPH_RADIUS_RATIO * 0.5;
//				xj = xi - xj;
//				dist = Length(xj);
//				if (dist < hj) {
//					invh = 1.0 / hj;
//					pp -= 10.0 * volumej * SPHKernel::GKernel(dist, invh) / dist * xj;
//				}*/
//			}
//		});
//
//	REAL c = visc / poreParticles._Ks[id * 3u + 0u];
//	//REAL3 vP = (1.0 - restSolidFraction) / c * (pc + pp + gravity); //fluidDensity*
//	REAL3 vP = gravity;
//	Normalize(vP);
//
//	setVector(poreParticles._vPs, id, vP);
//#else
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	REAL visc = 0.89;
//
//	uint phase = poreParticles._phases[id];
//	REAL si = poreParticles._ss[id], sj;
//
//	REAL hi = poreParticles._radii[phase], hj;
//	hi *= SPH_RADIUS_RATIO;
//	REAL invhi = 1.0 / hi, invhj;
//
//	REAL fluidDensity = poreParticles._restFluidDensities[id];
//	fluidDensity *= si;
//	REAL restSolidFractioni = poreParticles._restSolidFractions[phase], restSolidFractionj;
//
//	REAL3 xi, xj;
//	getVector(poreParticles._xs, id, xi);
//
//	REAL mj, dj, volumej, pj;
//	REAL dist;
//
//	REAL PressureK = 16.8;
//	REAL3 pc = make_REAL3(0.0);
//	REAL3 pp = make_REAL3(0.0);
//
//	uint2 nei;
//	researchNeighbors(poreParticles, id, nei,
//		{
//			if (nei.x == TYPE_SPH_PARTICLE) {
//				phase = sphParticles._phases[nei.y];
//				sj = sphParticles._ss[nei.y];
//				hj = sphParticles._radii[phase];
//				hj *= S3TO1(sj);
//
//				pj = sphParticles._relaxTs[nei.y];
//				volumej = sphParticles._restVolumes[phase];
//				volumej *= sj * pj;
//				getVector(sphParticles._xs, nei.y, xj);
//
//				hj *= SPH_RADIUS_RATIO;
//				invhj = 1.0 / hj;
//				xj = xi - xj;
//				dist = Length(xj);
//
//				pp -= PressureK * hi * volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
//			}
//			else if (nei.x == TYPE_PORE_PARTICLE) {
//				phase = poreParticles._phases[nei.y];
//				hj = poreParticles._radii[phase];
//				volumej = poreParticles._volumes[nei.y];
//				restSolidFractionj = poreParticles._restSolidFractions[phase];
//				//volumej *= (1.0 - restSolidFractionj);
//				sj = poreParticles._ss[nei.y];
//				pj = poreParticles._Ps[nei.y * 3u + 0u];
//
//				getVector(poreParticles._xs, nei.y, xj);
//
//				hj *= SPH_RADIUS_RATIO;
//				invhj = 1.0 / hj;
//				xj = xi - xj;
//				dist = Length(xj);
//
//				pc += max(1.0 - sj, 0.0) * pj * (1.0 - restSolidFractionj) * volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
//				//pp -= sj * PressureK * hi * volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
//			}
//			else {
//				phase = boundaryParticles._phases[nei.y];
//				hj = boundaryParticles._radii[phase];
//				volumej = boundaryParticles._volumes[nei.y];
//
//				getVector(boundaryParticles._xs, nei.y, xj);
//
//				hj *= SPH_RADIUS_RATIO;
//				invhj = 1.0 / hj;
//				xj = xi - xj;
//				dist = Length(xj);
//
//				pp -= PressureK * hi * volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
//			}
//		});
//
//	REAL c = visc / poreParticles._Ks[id * 3u + 0u];
//	REAL3 vP = (1.0 - restSolidFractioni) / c * (/*pc +*/ pp + gravity); //fluidDensity*
//	//REAL3 vP = gravity;
//	Normalize(vP);
//
//	setVector(poreParticles._vPs, id, vP);
//#endif
//}
//
//__global__ void lerpMassToObject_kernel(
//	ClothParam cloth, PoreParticleParam particles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= particles._numParticles)
//		return;
//
//	REAL dm = particles._dms[id];
//
//	REAL w0, w1, w2;
//	uint ino = id << 1u;
//	if (dm < 0.0) {
//		w0 = particles._mws[ino + 0u];
//		w1 = particles._mws[ino + 1u];
//	}
//	else {
//		w0 = particles._ws[ino + 0u];
//		w1 = particles._ws[ino + 1u];
//	}
//	w2 = 1.0 - w0 - w1;
//
//	ino = particles._inos[id];
//	ino *= 3u;
//	uint ino0 = cloth._fs[ino + 0u];
//	uint ino1 = cloth._fs[ino + 1u];
//	uint ino2 = cloth._fs[ino + 2u];
//	atomicAdd_REAL(cloth._mfs + ino0, w0 * dm);
//	atomicAdd_REAL(cloth._mfs + ino1, w1 * dm);
//	atomicAdd_REAL(cloth._mfs + ino2, w2 * dm);
//}
//__global__ void compAbsorption_kernel(
//	SPHParticleParam sphParticles, PoreParticleParam poreParticles, REAL dt)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= sphParticles._numParticles)
//		return;
//
//	uint phase = sphParticles._phases[id];
//	REAL si = sphParticles._ss[id], sj;
//
//	REAL hi = sphParticles._radii[phase], hj;
//	hi *= SPH_RADIUS_RATIO * S3TO1(si);
//	REAL invhi = 1.0 / hi, invhj;
//
//	REAL restDensityi = sphParticles._restDensities[phase];
//	
//	REAL mi = sphParticles._masses[phase];
//	REAL m0i = mi;
//	mi *= si;
//
//	REAL3 xi, xj;
//	getVector(sphParticles._xs, id, xi);
//
//	REAL volumei = 1.0;// sphParticles._ds[id] / restDensityi;
//	REAL volumej, solidFraction;
//	REAL3 vP;
//	REAL dist, dot, dmij;
//
//	REAL fluidMass0;
//	REAL dm = 0.0;
//
//	uint2 nei;
//	researchNeighbors(sphParticles, id, nei,
//		{
//			if (nei.x == TYPE_PORE_PARTICLE) {
//				phase = poreParticles._phases[nei.y];
//				hj = poreParticles._radii[phase];
//
//				solidFraction = poreParticles._restSolidFractions[phase];
//				sj = poreParticles._ss[nei.y];
//
//				//volumej = poreParticles._volumes[nei.y];
//				volumej = poreParticles._restVolumes[phase];
//				volumej *= (1.0 - solidFraction);
//				fluidMass0 = volumej * restDensityi;
//
//				volumej = poreParticles._volumes[nei.y];
//				//volumej = poreParticles._restVolumes[phase];
//				volumej *= (1.0 - solidFraction);
//
//				getVector(poreParticles._xs, nei.y, xj);
//				getVector(poreParticles._vPs, nei.y, vP);
//
//				hj *= SPH_RADIUS_RATIO;
//				invhj = 1.0 / hj;
//				xj = xi - xj;
//				dist = Length(xj);
//
//				dot = -ABSORPTION_N * Dot(vP, xj) / (dist + FLT_EPSILON);
//				dmij =
//					dt * ABSORPTION_K * fluidMass0 * (ABSORPTION_MAX - sj + dot) * volumei * volumej *
//					SPHKernel::LaplacianKernel(dist, invhi, invhj);
//				if (dmij > 0.0)
//					dm += dmij;
//
//				/*dot = 1.0 - 0.8 * Dot(vP, xj) / (dist + FLT_EPSILON);
//				if (dot > 0.0) {
//					dmij =
//						dt * ABSORPTION_K * m0i * (ABSORPTION_MAX - sj) * dot * volumej *
//						SPHKernel::LaplacianKernel(dist, invhi, invhj);
//					if (dmij > 0.0) {
//						dm += dmij;
//					}
//				}*/
//			}
//		});
//	if (dm > 1.0e-40) {
//		REAL invM = 1.0;
//		if (dm > mi - m0i * MIN_VOLUME) {
//			if (dm > mi - m0i * MIN_VOLUME * 0.8)
//				invM = mi / dm;
//			else
//				invM = max((mi - m0i * (MIN_VOLUME + 1.0e-5)) / dm, 0.0);
//		}
//		dm = 0.0;
//		researchNeighbors(sphParticles, id, nei,
//			{
//				if (nei.x == TYPE_PORE_PARTICLE) {
//					phase = poreParticles._phases[nei.y];
//					hj = poreParticles._radii[phase];
//
//					solidFraction = poreParticles._restSolidFractions[phase];
//					sj = poreParticles._ss[nei.y];
//
//					//volumej = poreParticles._volumes[nei.y];
//					volumej = poreParticles._restVolumes[phase];
//					volumej *= (1.0 - solidFraction);
//					fluidMass0 = volumej * restDensityi;
//
//					volumej = poreParticles._volumes[nei.y];
//					//volumej = poreParticles._restVolumes[phase];
//					volumej *= (1.0 - solidFraction);
//
//					getVector(poreParticles._xs, nei.y, xj);
//					getVector(poreParticles._vPs, nei.y, vP);
//
//					hj *= SPH_RADIUS_RATIO;
//					invhj = 1.0 / hj;
//					xj = xi - xj;
//					dist = Length(xj);
//
//					dot = -ABSORPTION_N * Dot(vP, xj) / (dist + FLT_EPSILON);
//					dmij =
//						invM * dt * ABSORPTION_K * fluidMass0 * (ABSORPTION_MAX - sj + dot) * volumei * volumej *
//						SPHKernel::LaplacianKernel(dist, invhi, invhj);
//					if (dmij > 0.0) {
//						dm -= dmij;
//						atomicAdd_REAL(poreParticles._dms + nei.y, dmij);
//					}
//
//					/*dot = 1.0 - 0.8 * Dot(vP, xj) / (dist + FLT_EPSILON);
//					if (dot > 0.0) {
//						dmij =
//							invM * dt * ABSORPTION_K * m0i * (ABSORPTION_MAX - sj) * dot * volumej *
//							SPHKernel::LaplacianKernel(dist, invhi, invhj);
//						if (dmij > 0.0) {
//							dm -= dmij;
//							atomicAdd_REAL(poreParticles._dms + nei.y, dmij);
//						}
//					}*/
//				}
//			});
//		sphParticles._ss[id] = si + dm / m0i;
//	}
//}
//
//__global__ void compIsDripping_kernel(
//	PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	uint phase = poreParticles._phases[id];
//
//	REAL hi = poreParticles._radii[phase], hj;
//	hi *= SPH_RADIUS_RATIO;
//	REAL invhi = 1.0 / hi, invhj;
//
//	REAL3 vP;
//	getVector(poreParticles._vPs, id, vP);
//
//	REAL3 xi, xj;
//	getVector(poreParticles._xs, id, xi);
//
//	REAL dist, dot, volumej;
//
//	uchar isDripping = 1u;
//	REAL obsWs = 0.0;
//
//	uint2 nei;
//	researchNeighbors(poreParticles, id, nei,
//		{
//			if (nei.x == TYPE_BOUNDARY_PARTICLE) {
//				phase = boundaryParticles._phases[nei.y];
//				hj = boundaryParticles._radii[phase];
//
//				volumej = boundaryParticles._radii[phase];
//				volumej = volumej * volumej * volumej * M_PI * 4.0 / 3.0;
//				getVector(boundaryParticles._xs, nei.y, xj);
//
//				hj *= SPH_RADIUS_RATIO;
//				invhj = 1.0 / hj;
//				xj = xi - xj;
//				dist = Length(xj);
//
//				dot = -Dot(xj, vP) / (dist + FLT_EPSILON);
//				if (dot > 0.0)
//					obsWs += dot * volumej * SPHKernel::WKernel(dist, invhi, invhj);
//				if (dot > 0.5)
//					isDripping = 0u;
//			}
//		});
//	if (obsWs > DRIPPING_THRESHOLD)
//		isDripping = 0u;
//	
//	poreParticles._isDrippings[id] = isDripping;
//}
//__global__ void compEmissionNodeWeight_kernel(
//	ClothParam cloth, PoreParticleParam particles, REAL* nodeWeights)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= particles._numParticles)
//		return;
//
//	uint isDripping = particles._isDrippings[id];
//	if (isDripping) {
//		/*REAL w = particles._volumes[id];
//
//		uint ino = particles._inos[id];
//		ino *= 3u;
//		uint ino0 = cloth._fs[ino + 0u];
//		uint ino1 = cloth._fs[ino + 1u];
//		uint ino2 = cloth._fs[ino + 2u];
//		atomicAdd_REAL(nodeWeights + ino0, w);
//		atomicAdd_REAL(nodeWeights + ino1, w);
//		atomicAdd_REAL(nodeWeights + ino2, w);*/
//		uint ino = id << 1u;
//		REAL w0 = particles._ws[ino + 0u];
//		REAL w1 = particles._ws[ino + 1u];
//		REAL w2 = 1.0 - w0 - w1;
//
//		ino = particles._inos[id];
//		ino *= 3u;
//		uint ino0 = cloth._fs[ino + 0u];
//		uint ino1 = cloth._fs[ino + 1u];
//		uint ino2 = cloth._fs[ino + 2u];
//		atomicAdd_REAL(nodeWeights + ino0, pow(w0, 1.0));
//		atomicAdd_REAL(nodeWeights + ino1, pow(w1, 1.0));
//		atomicAdd_REAL(nodeWeights + ino2, pow(w2, 1.0));
//	}
//}
//__global__ void lerpEmissionMassToParticle_kernel(
//	ClothParam cloth, PoreParticleParam particles, REAL* nodeWeights)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= particles._numParticles)
//		return;
//#if 1
//	uint isDripping = particles._isDrippings[id];
//	if (isDripping) {
//		uint ino = id << 1u;
//		REAL w0 = particles._ws[ino + 0u];
//		REAL w1 = particles._ws[ino + 1u];
//		REAL w2 = 1.0 - w0 - w1;
//
//		ino = particles._inos[id];
//		ino *= 3u;
//		uint ino0 = cloth._fs[ino + 0u];
//		uint ino1 = cloth._fs[ino + 1u];
//		uint ino2 = cloth._fs[ino + 2u];
//		REAL fluidMass0 = cloth._mfs[ino0];
//		REAL fluidMass1 = cloth._mfs[ino1];
//		REAL fluidMass2 = cloth._mfs[ino2];
//		REAL maxFluidMass0 = cloth._maxFluidMass[ino0];
//		REAL maxFluidMass1 = cloth._maxFluidMass[ino1];
//		REAL maxFluidMass2 = cloth._maxFluidMass[ino2];
//
//		REAL ws0 = particles._nodeWeights[ino0];
//		REAL ws1 = particles._nodeWeights[ino1];
//		REAL ws2 = particles._nodeWeights[ino2];
//
//		REAL dm0 = max(fluidMass0 - maxFluidMass0, 0.0);
//		REAL dm1 = max(fluidMass1 - maxFluidMass1, 0.0);
//		REAL dm2 = max(fluidMass2 - maxFluidMass2, 0.0);
//		dm0 *= w0 / ws0;
//		dm1 *= w1 / ws1;
//		dm2 *= w2 / ws2;
//
//		REAL total = dm0 + dm1 + dm2;
//		if (total > 1.0e-40) {
//			particles._mfs[id] = total;
//
//			total = 1.0 / total;
//			ino = id << 1u;
//			particles._mws[ino + 0u] = dm0 * total;
//			particles._mws[ino + 1u] = dm1 * total;
//		}
//		else {
//			particles._isDrippings[id] = 0u;
//			particles._mfs[id] = 0.0;
//		}
//		/*uint ino = id << 1u;
//		REAL w0 = particles._ws[ino + 0u];
//		REAL w1 = particles._ws[ino + 1u];
//		REAL w2 = 1.0 - w0 - w1;
//
//		ino = particles._inos[id];
//		ino *= 3u;
//		uint ino0 = cloth._fs[ino + 0u];
//		uint ino1 = cloth._fs[ino + 1u];
//		uint ino2 = cloth._fs[ino + 2u];
//		REAL fluidMass0 = cloth._mfs[ino0];
//		REAL fluidMass1 = cloth._mfs[ino1];
//		REAL fluidMass2 = cloth._mfs[ino2];
//		REAL maxFluidMass0 = cloth._maxFluidMass[ino0];
//		REAL maxFluidMass1 = cloth._maxFluidMass[ino1];
//		REAL maxFluidMass2 = cloth._maxFluidMass[ino2];
//
//		REAL dm0 = max(fluidMass0 - maxFluidMass0, 0.0);
//		REAL dm1 = max(fluidMass1 - maxFluidMass1, 0.0);
//		REAL dm2 = max(fluidMass2 - maxFluidMass2, 0.0);
//
//		if (w0 > 1.0 - 1.0e-10) {
//			particles._mfs[id] = dm0;
//		}
//		else if (w1 > 1.0 - 1.0e-10) {
//			particles._mfs[id] = dm1;
//		}
//		else if (w2 > 1.0 - 1.0e-10) {
//			particles._mfs[id] = dm2;
//		}
//		else {
//			particles._mfs[id] = 0.0;
//		}
//		ino = id << 1u;
//		particles._mws[ino + 0u] = w0;
//		particles._mws[ino + 1u] = w1;*/
//	}
//	else {
//		particles._mfs[id] = 0.0;
//	}
//#else
//	uint isDripping = particles._isDrippings[id];
//	if (isDripping) {
//		uint ino = id << 1u;
//		REAL w0 = particles._ws[ino + 0u];
//		REAL w1 = particles._ws[ino + 1u];
//		REAL w2 = 1.0 - w0 - w1;
//
//		ino = particles._inos[id];
//		ino *= 3u;
//		uint ino0 = cloth._fs[ino + 0u];
//		uint ino1 = cloth._fs[ino + 1u];
//		uint ino2 = cloth._fs[ino + 2u];
//		REAL fluidMass0 = cloth._mfs[ino0];
//		REAL fluidMass1 = cloth._mfs[ino1];
//		REAL fluidMass2 = cloth._mfs[ino2];
//		REAL maxFluidMass0 = cloth._maxFluidMass[ino0];
//		REAL maxFluidMass1 = cloth._maxFluidMass[ino1];
//		REAL maxFluidMass2 = cloth._maxFluidMass[ino2];
//
//		REAL ws0 = nodeWeights[ino0];
//		REAL ws1 = nodeWeights[ino1];
//		REAL ws2 = nodeWeights[ino2];
//
//		REAL dm0 = max(fluidMass0 - maxFluidMass0, 0.0);
//		REAL dm1 = max(fluidMass1 - maxFluidMass1, 0.0);
//		REAL dm2 = max(fluidMass2 - maxFluidMass2, 0.0);
//		dm0 *= pow(w0, 1.0) / ws0;
//		dm1 *= pow(w1, 1.0) / ws1;
//		dm2 *= pow(w2, 1.0) / ws2;
//
//		REAL total = dm0 + dm1 + dm2;
//		if (total > 1.0e-40) {
//			particles._mfs[id] = total;
//
//			total = 1.0 / total;
//			ino = id << 1u;
//			particles._mws[ino + 0u] = dm0 * total;
//			particles._mws[ino + 1u] = dm1 * total;
//		}
//		else {
//			particles._mfs[id] = 0.0;
//			particles._isDrippings[id] = 0u;
//		}
//		/*uint ino = particles._inos[id];
//		ino *= 3u;
//		uint ino0 = cloth._fs[ino + 0u];
//		uint ino1 = cloth._fs[ino + 1u];
//		uint ino2 = cloth._fs[ino + 2u];
//		REAL fluidMass0 = cloth._mfs[ino0];
//		REAL fluidMass1 = cloth._mfs[ino1];
//		REAL fluidMass2 = cloth._mfs[ino2];
//		REAL maxFluidMass0 = cloth._maxFluidMass[ino0];
//		REAL maxFluidMass1 = cloth._maxFluidMass[ino1];
//		REAL maxFluidMass2 = cloth._maxFluidMass[ino2];
//
//		REAL w = particles._volumes[id];
//		REAL ws0 = nodeWeights[ino0];
//		REAL ws1 = nodeWeights[ino1];
//		REAL ws2 = nodeWeights[ino2];
//
//		REAL dm0 = max(fluidMass0 - maxFluidMass0, 0.0);
//		REAL dm1 = max(fluidMass1 - maxFluidMass1, 0.0);
//		REAL dm2 = max(fluidMass2 - maxFluidMass2, 0.0);
//		dm0 *= w / ws0;
//		dm1 *= w / ws1;
//		dm2 *= w / ws2;
//
//		REAL total = dm0 + dm1 + dm2;
//		if (total > 1.0e-40) {
//			particles._mfs[id] = total;
//
//			total = 1.0 / total;
//			ino = id << 1u;
//			particles._mws[ino + 0u] = dm0 * total;
//			particles._mws[ino + 1u] = dm1 * total;
//		}
//		else {
//			particles._mfs[id] = 0.0;
//		}*/
//	}
//	else particles._mfs[id] = 0.0;
//#endif
//}
//__global__ void compEmissionWeight_kernel(
//	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= sphParticles._numParticles)
//		return;
//
//	uint phase = sphParticles._phases[id];
//	REAL si = sphParticles._ss[id], sj;
//
//	REAL ws = 0.0;
//	if (si > MIN_VOLUME) {
//		REAL restDensityi = sphParticles._restDensities[phase];
//
//		REAL m0i = sphParticles._masses[phase];
//		REAL mi = m0i * si;
//
//		REAL3 xi, xj;
//		getVector(sphParticles._xs, id, xi);
//
//		REAL overMass;
//		REAL3 vP;
//		REAL dist, dot, dmij;
//
//		uint2 nei;
//		researchNeighbors(sphParticles, id, nei,
//			{
//				if (nei.x == TYPE_PORE_PARTICLE) {
//					overMass = poreParticles._mfs[nei.y];
//					getVector(poreParticles._xs, nei.y, xj);
//					getVector(poreParticles._vPs, nei.y, vP);
//
//					if (overMass > 1.0e-40) {
//						xj = xi - xj;
//						dist = Length(xj);
//
//						dot = EMISSION_N + Dot(xj, vP) / (dist + FLT_EPSILON);
//						if (dot > 0.0)
//							ws += dot * overMass;
//					}
//				}
//			});
//
//		if (ws > 1.0e-20)
//			ws = (m0i - mi) / ws;
//		else
//			ws = 0.0;
//	}
//	sphParticles._ks[id] = ws;
//}
//__global__ void compEmissionToSPHParticle_kernel(
//	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	uint phase = poreParticles._phases[id];
//	REAL overMass = poreParticles._mfs[id];
//
//	if (overMass > 1.0e-40) {
//		REAL3 xi, xj;
//		getVector(poreParticles._xs, id, xi);
//		REAL3 vP;
//		getVector(poreParticles._vPs, id, vP);
//
//		REAL wsj, m0j, sj;
//		REAL dist, dot, dmij;
//
//		REAL dm = 0.0;
//
//		uint2 nei;
//		researchNeighbors(poreParticles, id, nei,
//			{
//				if (nei.x == TYPE_SPH_PARTICLE) {
//					phase = sphParticles._phases[nei.y];
//					wsj = sphParticles._ks[nei.y];
//					getVector(sphParticles._xs, nei.y, xj);
//
//					if (wsj > 0.0) {
//						xj = xi - xj;
//						dist = Length(xj);
//						dot = EMISSION_N - Dot(xj, vP) / (dist + FLT_EPSILON);
//						if (dot > 0.0) {
//							dmij = wsj * dot * overMass;
//							dm += dmij;
//						}
//					}
//				}
//			});
//		if (dm > 1.0e-40) {
//			REAL invM = 1.0;
//			if (dm > overMass)
//				invM = overMass / dm;
//
//			dm = 0.0;
//			researchNeighbors(poreParticles, id, nei,
//				{
//					if (nei.x == TYPE_SPH_PARTICLE) {
//						phase = sphParticles._phases[nei.y];
//						m0j = sphParticles._masses[phase];
//						wsj = sphParticles._ks[nei.y];
//						sj = sphParticles._ss[nei.y];
//						getVector(sphParticles._xs, nei.y, xj);
//
//						if (sj > MIN_VOLUME) {
//							xj = xi - xj;
//							dist = Length(xj);
//
//							dot = EMISSION_N - Dot(xj, vP) / (dist + FLT_EPSILON);
//							if (dot > 0.0) {
//								dmij = invM * wsj * dot * overMass;
//								dm -= dmij;
//								atomicAdd_REAL(sphParticles._ss + nei.y, dmij / m0j);
//							}
//						}
//					}
//				});
//
//			overMass += dm;
//			poreParticles._mfs[id] = overMass;
//			poreParticles._dms[id] = dm;
//		}
//	}
//}
//__global__ void compEmissionToPoreParticle_kernel(
//	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	uint phase = poreParticles._phases[id];
//
//	REAL hi = poreParticles._radii[phase], hj;
//	hi *= SPH_RADIUS_RATIO;
//	REAL invhi = 1.0 / hi, invhj;
//
//	REAL overMass = poreParticles._mfs[id];
//	REAL sphMass = sphParticles._masses[0];
//	REAL minMass = sphMass * (MIN_VOLUME + 1.0e-10);
//
//	REAL dm = 0.0;
//	uchar isDripping = poreParticles._isDrippings[id];
//	if (isDripping) {
//		if (overMass > 0.0) {
//			REAL3 vP;
//			getVector(poreParticles._vPs, id, vP);
//
//			REAL3 xi, xj;
//			getVector(poreParticles._xs, id, xi);
//
//			REAL volumej, restSolidFractionj;
//			REAL dist, dot;
//
//			REAL ws = 0.0, w;
//
//			uint2 nei;
//			researchNeighbors(poreParticles, id, nei,
//				{
//					if (nei.x == TYPE_PORE_PARTICLE) {
//						phase = poreParticles._phases[nei.y];
//						hj = poreParticles._radii[phase];
//
//						restSolidFractionj = poreParticles._restSolidFractions[phase];
//						volumej = poreParticles._volumes[nei.y];
//						volumej *= (1.0 - restSolidFractionj);
//						getVector(poreParticles._xs, nei.y, xj);
//
//						hj *= SPH_RADIUS_RATIO;
//						invhj = 1.0 / hj;
//						xj = xi - xj;
//						dist = Length(xj);
//
//						dot = (DRIPPING_N - Dot(xj, vP) / (dist + FLT_EPSILON));
//						if (dot > 0.0) {
//							w = dot * volumej * SPHKernel::WKernel(dist, invhi, invhj);
//							ws += w;
//						}
//					}
//				});
//
//			if (ws > 1.0e-40) {
//				ws = 1.0 / ws;
//
//				REAL dmij;
//				researchNeighbors(poreParticles, id, nei,
//					{
//						if (nei.x == TYPE_PORE_PARTICLE) {
//							phase = poreParticles._phases[nei.y];
//							hj = poreParticles._radii[phase];
//
//							restSolidFractionj = poreParticles._restSolidFractions[phase];
//							volumej = poreParticles._volumes[nei.y];
//							volumej *= (1.0 - restSolidFractionj);
//							getVector(poreParticles._xs, nei.y, xj);
//
//							hj *= SPH_RADIUS_RATIO;
//							invhj = 1.0 / hj;
//							xj = xi - xj;
//							dist = Length(xj);
//
//							dot = (DRIPPING_N - Dot(xj, vP) / (dist + FLT_EPSILON));
//							if (dot > 0.0) {
//								w = dot * volumej * SPHKernel::WKernel(dist, invhi, invhj);
//								dmij = w * ws * overMass;
//								dm -= dmij;
//								atomicAdd_REAL(poreParticles._dms + nei.y, dmij);
//							}
//						}
//					});
//			}
//		}
//
//		overMass += dm;
//		if (overMass > minMass * 0.8) {
//			if (overMass < minMass)
//				overMass = minMass;
//			if (overMass > sphMass)
//				overMass = sphMass;
//			dm -= overMass;
//			isDripping = 1u;
//		}
//		else isDripping = 0u;
//		/*if (overMass >= minMass) {
//			if (overMass > sphMass)
//				overMass = sphMass;
//			dm -= overMass;
//			isDripping = 1u;
//		}
//		else isDripping = 0u;*/
//
//		atomicAdd_REAL(poreParticles._dms + id, dm);
//		poreParticles._mfs[id] = overMass;
//		poreParticles._isDrippings[id] = isDripping;
//	}
//}
////__global__ void compEmissionToPoreParticle_kernel(
////	PoreParticleParam poreParticles)
////{
////	uint id = blockDim.x * blockIdx.x + threadIdx.x;
////	if (id >= poreParticles._numParticles)
////		return;
////
////	uint phase = poreParticles._phases[id];
////
////	REAL hi = poreParticles._radii[phase], hj;
////	hi *= SPH_RADIUS_RATIO;
////	REAL invhi = 1.0 / hi, invhj;
////
////	REAL overMass = poreParticles._mfs[id];
////
////	REAL dm = 0.0;
////	uchar isDripping = poreParticles._isDrippings[id];
////	if (isDripping) {
////		if (overMass > 0.0) {
////			REAL3 vP;
////			getVector(poreParticles._vPs, id, vP);
////
////			REAL3 xi, xj;
////			getVector(poreParticles._xs, id, xi);
////
////			REAL volumej, restSolidFractionj;
////			REAL dist, dot;
////
////			REAL ws = 0.0, w;
////
////			uint2 nei;
////			researchNeighbors(poreParticles, id, nei,
////				{
////					if (nei.x == TYPE_PORE_PARTICLE) {
////						phase = poreParticles._phases[nei.y];
////						hj = poreParticles._radii[phase];
////
////						restSolidFractionj = poreParticles._restSolidFractions[phase];
////						volumej = poreParticles._volumes[nei.y];
////						volumej *= (1.0 - restSolidFractionj);
////						getVector(poreParticles._xs, nei.y, xj);
////
////						hj *= SPH_RADIUS_RATIO;
////						invhj = 1.0 / hj;
////						xj = xi - xj;
////						dist = Length(xj);
////
////						dot = (DRIPPING_N - Dot(xj, vP) / (dist + FLT_EPSILON));
////						if (dot > 0.0) {
////							w = dot * volumej * SPHKernel::WKernel(dist, invhi, invhj);
////							ws += w;
////							isDripping = 0u;
////						}
////					}
////				});
////
////			if (ws > 1.0e-40) {
////				ws = 1.0 / ws;
////
////				REAL dmij;
////				researchNeighbors(poreParticles, id, nei,
////					{
////						if (nei.x == TYPE_PORE_PARTICLE) {
////							phase = poreParticles._phases[nei.y];
////							hj = poreParticles._radii[phase];
////
////							restSolidFractionj = poreParticles._restSolidFractions[phase];
////							volumej = poreParticles._volumes[nei.y];
////							volumej *= (1.0 - restSolidFractionj);
////							getVector(poreParticles._xs, nei.y, xj);
////
////							hj *= SPH_RADIUS_RATIO;
////							invhj = 1.0 / hj;
////							xj = xi - xj;
////							dist = Length(xj);
////
////							dot = (DRIPPING_N - Dot(xj, vP) / (dist + FLT_EPSILON));
////							if (dot > 0.0) {
////								w = dot * volumej * SPHKernel::WKernel(dist, invhi, invhj);
////								dmij = w * ws * overMass;
////								dm -= dmij;
////								atomicAdd_REAL(poreParticles._dms + nei.y, dmij);
////							}
////						}
////					});
////			}
////		}
////		overMass += dm;
////		atomicAdd_REAL(poreParticles._dms + id, dm);
////		poreParticles._mfs[id] = overMass;
////		poreParticles._isDrippings[id] = isDripping;
////	}
////}
////__global__ void compEmissionCheck_kernel(
////	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
////{
////	uint id = blockDim.x * blockIdx.x + threadIdx.x;
////	if (id >= poreParticles._numParticles)
////		return;
////
////	uchar isDripping = poreParticles._isDrippings[id];
////	if (isDripping) {
////		REAL overMass = poreParticles._mfs[id];
////		REAL sphMass = sphParticles._masses[0];
////		REAL minMass = sphMass * (MIN_VOLUME + 1.0e-10);
////
////		REAL dm = 0.0;
////		if (overMass > minMass * 0.8) {
////			if (overMass < minMass)
////				overMass = minMass;
////			if (overMass > sphMass)
////				overMass = sphMass;
////			dm = -overMass;
////			isDripping = 1u;
////		}
////		else isDripping = 0u;
////
////		atomicAdd_REAL(poreParticles._dms + id, dm);
////		poreParticles._mfs[id] = overMass;
////		poreParticles._isDrippings[id] = isDripping;
////	}
////}
//
//__global__ void lerpDiffusionMassToParticle_kernel(
//	ClothParam cloth, PoreParticleParam particles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= particles._numParticles)
//		return;
//
//	uint ino = id << 1u;
//	REAL w0 = particles._ws[ino + 0u];
//	REAL w1 = particles._ws[ino + 1u];
//	REAL w2 = 1.0 - w0 - w1;
//
//	ino = particles._inos[id];
//	ino *= 3u;
//	uint ino0 = cloth._fs[ino + 0u];
//	uint ino1 = cloth._fs[ino + 1u];
//	uint ino2 = cloth._fs[ino + 2u];
//	REAL fluidMass0 = cloth._mfs[ino0];
//	REAL fluidMass1 = cloth._mfs[ino1];
//	REAL fluidMass2 = cloth._mfs[ino2];
//	REAL s0 = cloth._maxFluidMass[ino0];
//	REAL s1 = cloth._maxFluidMass[ino1];
//	REAL s2 = cloth._maxFluidMass[ino2];
//
//	REAL ws0 = particles._nodeWeights[ino0];
//	REAL ws1 = particles._nodeWeights[ino1];
//	REAL ws2 = particles._nodeWeights[ino2];
//
//	s0 = fluidMass0 / (s0 + FLT_EPSILON);
//	s1 = fluidMass1 / (s1 + FLT_EPSILON);
//	s2 = fluidMass2 / (s2 + FLT_EPSILON);
//
//	fluidMass0 *= w0 / ws0;
//	fluidMass1 *= w1 / ws1;
//	fluidMass2 *= w2 / ws2;
//
//	REAL total = fluidMass0 + fluidMass1 + fluidMass2;
//	if (total > 1.0e-40) {
//		particles._mfs[id] = total;
//		particles._ss[id] = w0 * s0 + w1 * s1 + w2 * s2;
//
//		total = 1.0 / total;
//		ino = id << 1u;
//		particles._mws[ino + 0u] = fluidMass0 * total;
//		particles._mws[ino + 1u] = fluidMass1 * total;
//	}
//	else {
//		particles._ss[id] = 0.0;
//		particles._mfs[id] = 0.0;
//	}
//}
//__global__ void compDiffusion_kernel(ClothParam cloths, PoreParticleParam poreParticles, REAL dt)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	uint phase = poreParticles._phases[id];
//
//	REAL hi = poreParticles._radii[phase], hj;
//	hi *= SPH_RADIUS_RATIO;
//	REAL invhi = 1.0 / hi, invhj;
//
//	REAL restSolidFractioni = poreParticles._restSolidFractions[phase], restSolidFractionj;
//	REAL restFluidDensity = poreParticles._restFluidDensities[phase];
//
//	REAL si = poreParticles._ss[id], sj;
//
//	REAL volumei = poreParticles._volumes[id], volumej;
//	volumei *= 1.0 - restSolidFractioni;
//
//	REAL fluidMassi = poreParticles._mfs[id], fluidMassj;
//
//	REAL3 xi, xj;
//	getVector(poreParticles._xs, id, xi);
//	REAL3 vPi, vPj;
//	getVector(poreParticles._vPs, id, vPi);
//
//	REAL dist, dot, dmij;
//
//	REAL dm = 0.0;
//	REAL grad;
//
//	uint2 nei;
//
//	researchNeighbors(poreParticles, id, nei,
//		{
//			if (nei.x == TYPE_PORE_PARTICLE) {
//				phase = poreParticles._phases[nei.y];
//				hj = poreParticles._radii[phase];
//
//				sj = poreParticles._ss[nei.y];
//
//				restSolidFractionj = poreParticles._restSolidFractions[phase];
//
//				volumej = poreParticles._volumes[nei.y];
//				volumej *= 1.0 - restSolidFractionj;
//
//				fluidMassj = poreParticles._mfs[nei.y];
//				getVector(poreParticles._xs, nei.y, xj);
//				getVector(poreParticles._vPs, nei.y, vPj);
//
//				hj *= SPH_RADIUS_RATIO;
//				invhj = 1.0 / hj;
//				xj = xi - xj;
//				dist = Length(xj);
//
//				dot = -DIFFUSION_N * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
//				grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
//				dmij = dt * (si - sj + dot) *
//					(fluidMassi * volumei + fluidMassj * volumej) * grad;
//
//				if (dmij > 0.0)
//					dm += dmij;
//			}
//		});
//
//	if (dm > 1.0e-40) {
//		REAL invM = 1.0;
//		if (dm > fluidMassi)
//			invM = fluidMassi / dm;
//		dm = 0.0;
//		researchNeighbors(poreParticles, id, nei,
//			{
//				if (nei.x == TYPE_PORE_PARTICLE) {
//					phase = poreParticles._phases[nei.y];
//					hj = poreParticles._radii[phase];
//
//					sj = poreParticles._ss[nei.y];
//
//					restSolidFractionj = poreParticles._restSolidFractions[phase];
//
//					volumej = poreParticles._volumes[nei.y];
//					volumej *= 1.0 - restSolidFractionj;
//
//					fluidMassj = poreParticles._mfs[nei.y];
//					getVector(poreParticles._xs, nei.y, xj);
//					getVector(poreParticles._vPs, nei.y, vPj);
//
//					hj *= SPH_RADIUS_RATIO;
//					invhj = 1.0 / hj;
//					xj = xi - xj;
//					dist = Length(xj);
//
//					dot = -DIFFUSION_N * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
//					grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
//					dmij = invM * dt * (si - sj + dot) *
//						(fluidMassi * volumei + fluidMassj * volumej) * grad;
//
//					if (dmij > 0.0) {
//						dm -= dmij;
//						atomicAdd_REAL(poreParticles._dms + nei.y, dmij);
//					}
//				}
//			});
//
//		atomicAdd_REAL(poreParticles._dms + id, dm);
//	}
//}
////__global__ void compDiffusion_kernel(ClothParam cloths, PoreParticleParam poreParticles, REAL dt)
////{
////	uint id = blockDim.x * blockIdx.x + threadIdx.x;
////	if (id >= poreParticles._numParticles)
////		return;
////
////	uint phase = poreParticles._phases[id];
////
////	REAL hi = poreParticles._radii[phase], hj;
////	hi *= SPH_RADIUS_RATIO;
////	REAL invhi = 1.0 / hi, invhj;
////
////	REAL restSolidFractioni = poreParticles._restSolidFractions[phase], restSolidFractionj;
////	REAL restFluidDensity = poreParticles._restFluidDensities[phase];
////
////	REAL volumei = poreParticles._restVolumes[phase], volumej;
////	//REAL volumei = poreParticles._volumes[id], volumej;
////	volumei *= 1.0 - restSolidFractioni;
////	REAL fluidMass0i = volumei * restFluidDensity, fluidMass0j;
////
////	//volumei = poreParticles._restVolumes[phase];
////	volumei = poreParticles._volumes[id];
////	volumei *= 1.0 - restSolidFractioni;
////
////	REAL fluidMassi = poreParticles._mfs[id], fluidMassj;
////
////	REAL3 xi, xj;
////	getVector(poreParticles._xs, id, xi);
////	REAL3 vPi, vPj;
////	getVector(poreParticles._vPs, id, vPi);
////
////	REAL dist, dot, dmij;
////
////	REAL dm = 0.0;
////	REAL grad;
////
////	uint2 nei;
////
////	researchNeighbors(poreParticles, id, nei,
////		{
////			if (nei.x == TYPE_PORE_PARTICLE) {
////				phase = poreParticles._phases[nei.y];
////				hj = poreParticles._radii[phase];
////
////				restSolidFractionj = poreParticles._restSolidFractions[phase];
////
////				volumej = poreParticles._restVolumes[phase];
////				//volumej = poreParticles._volumes[nei.y];
////				volumej *= 1.0 - restSolidFractionj;
////				fluidMass0j = volumej * restFluidDensity;
////
////				//volumej = poreParticles._restVolumes[phase];
////				volumej = poreParticles._volumes[nei.y];
////				volumej *= 1.0 - restSolidFractionj;
////
////				fluidMassj = poreParticles._mfs[nei.y];
////				getVector(poreParticles._xs, nei.y, xj);
////				getVector(poreParticles._vPs, nei.y, vPj);
////
////				hj *= SPH_RADIUS_RATIO;
////				invhj = 1.0 / hj;
////				xj = xi - xj;
////				dist = Length(xj);
////
////				//dot = -DIFFUSION_N * Dot(vPi + vPj, xj) / (dist + FLT_EPSILON);
////				dot = -DIFFUSION_N * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
////				grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
////				dmij = dt * (fluidMassi / fluidMass0i - fluidMassj / fluidMass0j + dot) *
////					(hi * fluidMassi * volumej + hj * fluidMassj * volumei) * grad;
////
////				/*dot = 1.0 - 0.8 * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
////				grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
////				dmij = dt * (fluidMassi / fluidMass0i - fluidMassj / fluidMass0j) * dot * 
////					(hi * fluidMassi * volumej + hj * fluidMassj * volumei) * grad;*/
////
////				if (dmij > 0.0)
////					dm += dmij;
////			}
////		});
////
////	if (dm > 1.0e-40) {
////		REAL invM = 1.0;
////		if (dm > fluidMassi)
////			invM = fluidMassi / dm;
////		dm = 0.0;
////		researchNeighbors(poreParticles, id, nei,
////			{
////				if (nei.x == TYPE_PORE_PARTICLE) {
////					phase = poreParticles._phases[nei.y];
////					hj = poreParticles._radii[phase];
////
////					restSolidFractionj = poreParticles._restSolidFractions[phase];
////
////					volumej = poreParticles._restVolumes[phase];
////					//volumej = poreParticles._volumes[nei.y];
////					volumej *= 1.0 - restSolidFractionj;
////					fluidMass0j = volumej * restFluidDensity;
////
////					//volumej = poreParticles._restVolumes[phase];
////					volumej = poreParticles._volumes[nei.y];
////					volumej *= 1.0 - restSolidFractionj;
////
////					fluidMassj = poreParticles._mfs[nei.y];
////					getVector(poreParticles._xs, nei.y, xj);
////					getVector(poreParticles._vPs, nei.y, vPj);
////
////					hj *= SPH_RADIUS_RATIO;
////					invhj = 1.0 / hj;
////					xj = xi - xj;
////					dist = Length(xj);
////
////					//dot = -DIFFUSION_N * Dot(vPi + vPj, xj) / (dist + FLT_EPSILON);
////					dot = -DIFFUSION_N * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
////					grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
////					dmij = invM * dt * (fluidMassi / fluidMass0i - fluidMassj / fluidMass0j + dot) *
////						(hi * fluidMassi * volumej + hj * fluidMassj * volumei) * grad;
////
////					/*dot = 1.0 - 0.8 * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
////					grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
////					dmij = invM * dt * (fluidMassi / fluidMass0i - fluidMassj / fluidMass0j) * dot * 
////						(hi * fluidMassi * volumej + hj * fluidMassj * volumei) * grad;*/
////
////					if (dmij > 0.0) {
////						dm -= dmij;
////						atomicAdd_REAL(poreParticles._dms + nei.y, dmij);
////					}
////				}
////			});
////
////		atomicAdd_REAL(poreParticles._dms + id, dm);
////	}
////}
//
//__global__ void compNewNumParticles_kernel(
//	SPHParticleParam sphParticles, uint* ids)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= sphParticles._numParticles)
//		return;
//
//	REAL s = sphParticles._ss[id];
//	uint x = 0u;
//	if (s > MIN_VOLUME * 0.5)
//		x = 1u;
//
//	if (id == 0u)
//		ids[0] = 0u;
//	ids[id + 1u] = x;
//}
//__global__ void copyOldToNew_kernel(
//	SPHParticleParam sphParticles,
//	REAL* newXs, REAL* newVs, REAL* newSs,
//	REAL* newRelaxTs, uint* newPhases, 
//	uint* ids)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= sphParticles._numParticles)
//		return;
//
//	REAL s = sphParticles._ss[id];
//	if (s > MIN_VOLUME * 0.5) {
//		uint newId = ids[id];
//		REAL3 x, v;
//		REAL s, relaxT;
//		uint phase;
//		getVector(sphParticles._xs, id, x);
//		getVector(sphParticles._vs, id, v);
//		s = sphParticles._ss[id];
//		relaxT = sphParticles._relaxTs[id];
//		phase = sphParticles._phases[id];
//
//		setVector(newXs, newId, x);
//		setVector(newVs, newId, v);
//		newSs[newId] = s;
//		newRelaxTs[newId] = relaxT;
//		newPhases[newId] = phase;
//	}
//}
//__global__ void compDrippingNum_kernel(
//	PoreParticleParam poreParticles, uint* ids)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	REAL isDripping = poreParticles._isDrippings[id];
//	uint x = 0u;
//	if (isDripping)
//		x = 1u;
//
//	if (id == 0u)
//		ids[0] = 0u;
//	ids[id + 1u] = x;
//}
//__global__ void generateDrippingParticle_kernel(
//	SPHParticleParam sphParticles, PoreParticleParam poreParticles,
//	REAL* newXs, REAL* newVs, REAL* newSs,
//	REAL* newRelaxTs, uint* newPhases, 
//	uint* ids, uint oldNumParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	REAL isDripping = poreParticles._isDrippings[id];
//	if (isDripping) {
//		uint phase = poreParticles._phases[id];
//		REAL radius = poreParticles._radii[phase];
//		REAL overMass = poreParticles._mfs[id];
//
//		REAL3 sphX, sphV, norm;
//		getVector(poreParticles._xs, id, sphX);
//		getVector(poreParticles._vs, id, sphV);
//		getVector(poreParticles._vPs, id, norm);
//
//		uint sphPhase = 0u;
//		REAL sphRadius = sphParticles._radii[sphPhase];
//		REAL sphMass = sphParticles._masses[sphPhase];
//
//		REAL sphS = overMass / sphMass;
//		REAL sphRelaxT = 0.1;
//		sphX += (S3TO1(sphS) * sphRadius + radius) * norm;
//
//		uint sphId = ids[id];
//		sphId += oldNumParticles;
//
//		setVector(newXs, sphId, sphX);
//		setVector(newVs, sphId, sphV);
//		newSs[sphId] = sphS;
//		newRelaxTs[sphId] = sphRelaxT;
//		newPhases[sphId] = sphPhase;
//	}
//}
//
//__global__ void compPorePressureForce_kernel(
//	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= sphParticles._numParticles)
//		return;
//
//	uint phase = sphParticles._phases[id];
//	REAL si = sphParticles._ss[id], sj;
//
//	REAL hi = sphParticles._radii[phase], hj;
//	hi *= SPH_RADIUS_RATIO * S3TO1(si);
//	REAL invhi = 1.0 / hi, invhj;
//
//	REAL restDensityi = sphParticles._restDensities[phase];
//	REAL mi = sphParticles._masses[phase], mj;
//	mi *= si;
//
//	REAL3 xi, xj;
//	REAL di = sphParticles._ds[id], dj;
//	getVector(sphParticles._xs, id, xi);
//
//	REAL3 forcei;
//	getVector(sphParticles._forces, id, forcei);
//
//	REAL volumei = mi / di, volumej;
//
//	REAL3 forceij;
//	REAL dist, pp;
//
//	uint2 nei;
//	researchNeighbors(sphParticles, id, nei,
//		{
//			if (nei.x == TYPE_PORE_PARTICLE) {
//				phase = poreParticles._phases[nei.y];
//				hj = poreParticles._radii[phase];
//
//				sj = poreParticles._ss[nei.y];
//				pp = poreParticles._Ps[nei.y * 3u + 0u];
//				pp *= max(1.0 - sj, 0.0);
//				getVector(poreParticles._xs, nei.y, xj);
//				volumej = poreParticles._volumes[nei.y];
//
//				hj *= SPH_RADIUS_RATIO;
//				invhj = 1.0 / hj;
//				xj = xi - xj;
//				dist = Length(xj);
//
//				REAL w0 = poreParticles._ws[(nei.y << 1u) + 0u];
//				REAL w1 = poreParticles._ws[(nei.y << 1u) + 1u];
//				REAL w2 = 1.0 - w0 - w1;
//				volumej *= 2.0 / (1.0 + w0 * w0 + w1 * w1 + w2 * w2);
//
//				forceij = -pp * volumei * volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
//				sumVector(poreParticles._forces, nei.y, forceij);
//				forcei -= forceij;
//			}
//		});
//
//
//	setVector(sphParticles._forces, id, forcei);
//}
//__global__ void compDragForce_kernel(
//	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= sphParticles._numParticles)
//		return;
//
//	uint phase = sphParticles._phases[id];
//	REAL si = sphParticles._ss[id], sj;
//
//	REAL hi = sphParticles._radii[phase], hj;
//	hi *= SPH_RADIUS_RATIO * S3TO1(si);
//	REAL invhi = 1.0 / hi, invhj;
//
//	REAL ki = sphParticles._viscosities[phase], kj;
//	REAL restDensityi = sphParticles._restDensities[phase];
//	REAL mi = sphParticles._masses[phase], mj;
//	mi *= si;
//
//	REAL3 xi, xj;
//	REAL3 vi, vj;
//	REAL di = sphParticles._ds[id], dj;
//	getVector(sphParticles._xs, id, xi);
//	getVector(sphParticles._vs, id, vi);
//
//	REAL3 forcei;
//	getVector(sphParticles._forces, id, forcei);
//
//	REAL volumei = mi / di;
//	REAL visc = 0.89;
//	const REAL Cons = 10.0 * volumei;
//
//	REAL3 forceij;
//	REAL dist, volumej;
//	REAL restSolidFraction, kA, cA;
//
//	uint2 nei;
//	researchNeighbors(sphParticles, id, nei,
//		{
//			if (nei.x == TYPE_PORE_PARTICLE) {
//				phase = poreParticles._phases[nei.y];
//				hj = poreParticles._radii[phase];
//				restSolidFraction = poreParticles._restSolidFractions[phase];
//				getVector(poreParticles._xs, nei.y, xj);
//				getVector(poreParticles._vs, nei.y, vj);
//				volumej = poreParticles._volumes[nei.y];
//				//volumej = volumej * restDensityi / di;
//
//				hj *= SPH_RADIUS_RATIO;
//				invhj = 1.0 / hj;
//				xj = xi - xj;
//				dist = Length(xj);
//
//				vj = vi - vj;
//
//				kA = poreParticles._Ks[nei.y * 3u + 0u];
//				cA = poreParticles._Cs[nei.y * 3u + 0u];
//				kj = visc / kA;// +cA * Length(vj);
//				kj /= (1.0 - restSolidFraction);
//
//				REAL w0 = poreParticles._ws[(nei.y << 1u) + 0u];
//				REAL w1 = poreParticles._ws[(nei.y << 1u) + 1u];
//				REAL w2 = 1.0 - w0 - w1;
//				volumej *= 2.0 / (1.0 + w0 * w0 + w1 * w1 + w2 * w2);
//
//				forceij = Cons * kj * volumej * Dot(xj, vj) *
//					SPHKernel::LaplacianKernel(dist, invhi, invhj) / dist * xj;
//				//forceij = Cons * kj * volumej *
//				//	SPHKernel::LaplacianKernel(dist, invhi, invhj) * dist * vj;
//
//				sumVector(poreParticles._forces, nei.y, forceij);
//
//				forcei -= forceij;
//			}
//		});
//
//
//	setVector(sphParticles._forces, id, forcei);
//}
//__global__ void compPoreAttractionForce_kernel(
//	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= sphParticles._numParticles)
//		return;
//
//	uint phase = sphParticles._phases[id];
//	REAL si = sphParticles._ss[id], sj;
//
//	REAL hi = sphParticles._radii[phase], hj;
//	hi *= SPH_RADIUS_RATIO * S3TO1(si);
//	REAL invhi = 1.0 / hi, invhj;
//
//	REAL ki = sphParticles._viscosities[phase], kj;
//	REAL restDensityi = sphParticles._restDensities[phase];
//	REAL mi = sphParticles._masses[phase], mj;
//	mi *= si;
//
//	REAL3 xi, xj;
//	getVector(sphParticles._xs, id, xi);
//
//	REAL di = sphParticles._ds[id], dj;
//
//	REAL3 forcei;
//	getVector(sphParticles._forces, id, forcei);
//
//	REAL volumei = mi / di;
//	const REAL Cons = 1000.0;
//
//	REAL3 forceij;
//	REAL dist, volumej;
//	REAL restSolidFraction, kA, cA;
//
//	uint2 nei;
//	researchNeighbors(sphParticles, id, nei,
//		{
//			if (nei.x == TYPE_PORE_PARTICLE) {
//				phase = poreParticles._phases[nei.y];
//				hj = poreParticles._radii[phase];
//				restSolidFraction = poreParticles._restSolidFractions[phase];
//				getVector(poreParticles._xs, nei.y, xj);
//				volumej = poreParticles._volumes[nei.y];
//				mj = volumej * restDensityi;
//
//				hj *= SPH_RADIUS_RATIO;
//				invhj = 1.0 / hj;
//				xj = xi - xj;
//				dist = Length(xj);
//
//				REAL s = poreParticles._ss[nei.y];
//				REAL force = Cons * (mi + s * 0.06) * mj * SPHKernel::WKernel(dist, invhi, invhj);
//
//				REAL w0 = poreParticles._ws[(nei.y << 1u) + 0u];
//				REAL w1 = poreParticles._ws[(nei.y << 1u) + 1u];
//				REAL w2 = 1.0 - w0 - w1;
//				volumej *= 2.0 / (1.0 + w0 * w0 + w1 * w1 + w2 * w2);
//
//				forceij = -force * volumei * volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
//				sumVector(poreParticles._forces, nei.y, forceij);
//
//				forcei -= forceij;
//			}
//		});
//
//
//	setVector(sphParticles._forces, id, forcei);
//}
//__global__ void compPoreAdhesionForce_kernel(
//	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	uint phase = poreParticles._phases[id];
//	REAL si = poreParticles._ss[id], sj;
//
//	REAL hi = poreParticles._radii[phase], hj;
//	hi *= SPH_RADIUS_RATIO;
//	REAL invhi = 1.0 / hi, invhj;
//
//	REAL restFluidDensityi = poreParticles._restFluidDensities[phase];
//	REAL mi = poreParticles._volumes[id], mj;
//	mi *= restFluidDensityi;
//
//	REAL wi0 = poreParticles._ws[(id << 1u) + 0u], wj0;
//	REAL wi1 = poreParticles._ws[(id << 1u) + 1u], wj1;
//	REAL wi2 = 1.0 - wi0 - wi1, wj2;
//	REAL iPti = wi0 * wi0 + wi1 * wi1 + wi2 * wi2, iPtj;
//
//	REAL3 xi, xj;
//	getVector(poreParticles._xs, id, xi);
//
//	REAL3 forcei;
//	getVector(poreParticles._forces, id, forcei);
//
//	REAL Ui = 0.0;
//	REAL3 forceij;
//	REAL dist;
//
//	uint2 nei;
//	researchNeighbors(poreParticles, id, nei,
//		{
//			if (nei.x == TYPE_SPH_PARTICLE) {
//				phase = sphParticles._phases[nei.y];
//				sj = sphParticles._ss[nei.y];
//				hj = sphParticles._radii[phase];
//				hj *= S3TO1(sj);
//
//				mj = sphParticles._restVolumes[phase];
//				mj *= sj;
//				getVector(sphParticles._xs, nei.y, xj);
//
//				hj *= SPH_RADIUS_RATIO;
//				invhj = 1.0 / hj;
//				xj = xi - xj;
//				dist = Length(xj);
//
//				Ui += mj * SPHKernel::WKernel(dist, invhi, invhj);
//			}
//		});
//
//	Ui = 1.0 - min(Ui, 1.0);
//	const REAL Const = Ui * 0.16;
//
//	researchNeighbors(poreParticles, id, nei,
//		{
//			if (nei.x != TYPE_SPH_PARTICLE) {
//				if (nei.x == TYPE_PORE_PARTICLE) {
//					phase = poreParticles._phases[nei.y];
//					sj = poreParticles._ss[nei.y];
//					hj = poreParticles._radii[phase];
//
//					mj = poreParticles._volumes[nei.y];
//					getVector(poreParticles._xs, nei.y, xj);
//
//					mj *= restFluidDensityi;
//					wj0 = poreParticles._ws[(nei.y << 1u) + 0u];
//					wj1 = poreParticles._ws[(nei.y << 1u) + 1u];
//					wj2 = 1.0 - wj0 - wj1;
//					iPtj = wj0 * wj0 + wj1 * wj1 + wj2 * wj2;
//					sj += si;
//				}
//				else {
//					phase = boundaryParticles._phases[nei.y];
//					hj = boundaryParticles._radii[phase];
//
//					mj = boundaryParticles._volumes[nei.y];
//					getVector(boundaryParticles._xs, nei.y, xj);
//
//					mj *= restFluidDensityi;
//					iPtj = 1.0;
//					sj = si;
//				}
//
//				hj *= SPH_RADIUS_RATIO;
//				invhj = 1.0 / hj;
//				xj = xi - xj;
//				dist = Length(xj);
//
//				forceij = Const / (iPti + iPtj) * sj * mi * mj *
//					SPHKernel::selfCohesionKernel(dist, invhi, invhj) / dist * xj;
//
//				forcei -= forceij;
//			}
//		});
//
//
//	setVector(poreParticles._forces, id, forcei);
//}
//
//#endif
#ifndef __POROUS_SOLVER_CUH__
#define __POROUS_SOLVER_CUH__

#pragma once
#include "PorousSolver.h"
#include "SPHKernel.cuh"
#include "../include/CUDA_Custom/DeviceManager.cuh"

#define FIXED_VOLUME		0
#define VOLUME_TEST			4.0

__global__ void massStabilization_kernel(ClothParam cloth)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= cloth._numNodes)
		return;

	REAL mf = cloth._mfs[id];
	if (mf < 1.0e-40)
		mf = 0.0;
}
__global__ void lerpPoreFactorToParticle_kernel(
	ClothParam cloth, REAL* norms, PoreParticleParam particles)
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
	REAL3 norm;
	norm.x = norms[ino + 0u];
	norm.y = norms[ino + 1u];
	norm.z = norms[ino + 2u];

	REAL mf0 = cloth._mfs[ino0];
	REAL mf1 = cloth._mfs[ino1];
	REAL mf2 = cloth._mfs[ino2];
	/*REAL s0 = cloth._ss[ino0];
	REAL s1 = cloth._ss[ino1];
	REAL s2 = cloth._ss[ino2];*/
	REAL s0 = cloth._maxFluidMass[ino0];
	REAL s1 = cloth._maxFluidMass[ino1];
	REAL s2 = cloth._maxFluidMass[ino2];
	s0 = mf0 / (s0 + FLT_EPSILON);
	s1 = mf1 / (s1 + FLT_EPSILON);
	s2 = mf2 / (s2 + FLT_EPSILON);
	REAL nodeWeight0 = particles._nodeWeights[ino0];
	REAL nodeWeight1 = particles._nodeWeights[ino1];
	REAL nodeWeight2 = particles._nodeWeights[ino2];

	ino = id * 3u;
	particles._norms[ino + 0u] = norm.x;
	particles._norms[ino + 1u] = norm.y;
	particles._norms[ino + 2u] = norm.z;

	//REAL mf = w0 * mf0 / nodeWeight0 + w1 * mf1 / nodeWeight1 + w2 * mf2 / nodeWeight2;
	REAL mf = w0 * mf0 + w1 * mf1 + w2 * mf2;
	particles._mfs[id] = mf;
	REAL s = w0 * s0 + w1 * s1 + w2 * s2;
	particles._ss[id] = s;
}
__global__ void lerpPoreFactorToObject_kernel(
	ClothParam cloth, PoreParticleParam particles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= particles._numParticles)
		return;

	uint ino = id << 1u;
	REAL w0 = particles._ws[ino + 0u];
	REAL w1 = particles._ws[ino + 1u];
	REAL w2 = 1.0 - w0 - w1;

	uint phase = particles._phases[id];
	REAL restSolidFraction = particles._restSolidFractions[phase];
	REAL restFluidDensity = particles._restFluidDensities[phase];

	REAL volume = particles._volumes[id];
	//REAL volume = particles._restVolumes[phase];

	REAL fluidMass0 = (1.0 - restSolidFraction) * volume * restFluidDensity;

	ino = particles._inos[id];
	ino *= 3u;
	uint ino0 = cloth._fs[ino + 0u];
	uint ino1 = cloth._fs[ino + 1u];
	uint ino2 = cloth._fs[ino + 2u];
	REAL nodeWeight0 = particles._nodeWeights[ino0];
	REAL nodeWeight1 = particles._nodeWeights[ino1];
	REAL nodeWeight2 = particles._nodeWeights[ino2];
	atomicAdd_REAL(cloth._maxFluidMass + ino0, w0 * fluidMass0);
	atomicAdd_REAL(cloth._maxFluidMass + ino1, w1 * fluidMass0);
	atomicAdd_REAL(cloth._maxFluidMass + ino2, w2 * fluidMass0);
	atomicAdd_REAL(cloth._restSolidFractions + ino0, w0 * restSolidFraction / nodeWeight0);
	atomicAdd_REAL(cloth._restSolidFractions + ino1, w1 * restSolidFraction / nodeWeight1);
	atomicAdd_REAL(cloth._restSolidFractions + ino2, w2 * restSolidFraction / nodeWeight2);

	//REAL s = particles._ss[id];
	volume = particles._restVolumes[phase];
	fluidMass0 = (1.0 - restSolidFraction) * volume * restFluidDensity;
	REAL s0 = cloth._mfs[ino0] / fluidMass0;
	REAL s1 = cloth._mfs[ino1] / fluidMass0;
	REAL s2 = cloth._mfs[ino2] / fluidMass0;
	atomicAdd_REAL(cloth._ss + ino0, s0 * w0);
	atomicAdd_REAL(cloth._ss + ino1, s1 * w1);
	atomicAdd_REAL(cloth._ss + ino2, s2 * w2);
}

__global__ void initPoreFactor_kernel(
	PoreParticleParam particles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= particles._numParticles)
		return;

	uint ino = id * 3u;
	REAL fd = 100.0;
	REAL rb = 31.0;

	REAL visc = 0.89;
	REAL surft = 72.0;

	uint phase = particles._phases[id];
	REAL sf = particles._restSolidFractions[phase];
	REAL restFluidDensity = particles._restFluidDensities[phase];

	//REAL volume = particles._restVolumes[phase];
	//REAL volume = particles._volumes[id];

	REAL fluidMass = particles._mfs[id];
	//REAL fluidMass0 = (1.0 - sf) * VOLUME_TEST * volume * restFluidDensity;

	REAL cc = 1.6;
	//REAL s = fluidMass / fluidMass0;

	REAL kA = (-log(sf) - 1.476 + sf * (2.0 - sf * 0.5)) /
		(16.0 * sf) * fd * fd;
	REAL kB = (-log(sf) - 1.476 + sf * (2.0 - sf * (1.774 - 4.078 * sf))) /
		(32.0 * sf) * fd * fd;

	/*REAL cA = 1.75 / sqrt(150.0) * pow(restFluidDensity * s, cc) * pow(fd, cc - 1.0) * pow(visc, 1.0 - cc) /
		(pow(1.0 - sf, 3.0 / 2.0) * sqrt(kA));
	REAL cB = 1.75 / sqrt(150.0) * pow(restFluidDensity * s, cc) * pow(fd, cc - 1.0) * pow(visc, 1.0 - cc) /
		(pow(1.0 - sf, 3.0 / 2.0) * sqrt(kB));*/
	REAL cA = 1.75 / sqrt(150.0) * restFluidDensity / (pow(1.0 - sf, 3.0 / 2.0) * sqrt(kA));
	REAL cB = 1.75 / sqrt(150.0) * restFluidDensity / (pow(1.0 - sf, 3.0 / 2.0) * sqrt(kB));

	REAL pA = 2.0 * surft * sf * cos(40.8 * M_PI / 180.0) / ((1.0 - sf) * rb);
	REAL pB = 0.5 * pA;

	particles._Ks[ino + 0u] = kA;
	particles._Ks[ino + 1u] = kA;
	particles._Ks[ino + 2u] = kA;

	particles._Cs[ino + 0u] = cA;
	particles._Cs[ino + 1u] = cA;
	particles._Cs[ino + 2u] = cA;

	particles._Ps[ino + 0u] = pA;
	particles._Ps[ino + 1u] = pA;
	particles._Ps[ino + 2u] = pA;

	//particles._ss[id] = s;
}
__global__ void updateRelaxT_kernel(
	SPHParticleParam sphParticles, REAL dt)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	REAL relaxT = sphParticles._relaxTs[id];
	if (relaxT != 1.0) {
		REAL density = sphParticles._ds[id];
		uint phase = sphParticles._phases[id];
		REAL restDensity = sphParticles._restDensities[phase];
		REAL s = sphParticles._ss[id];

		if (relaxT < 1.0) {
			relaxT += 1.0 * dt;

			if (relaxT > 1.0)
				relaxT = 1.0;
			else {
				REAL3 v;
				getVector(sphParticles._vs, id, v);

				REAL h = sphParticles._radii[phase];
				h *= SPH_RADIUS_RATIO * S3TO1(s);

				REAL lv = Length(v);
				REAL maxV = h * 1.0 * (0.05 + relaxT * 0.95) / dt;
				if (lv > maxV) {
					v *= maxV / lv;
					//setVector(sphParticles._vs, id, v);
				}

				/*if (density <= 0.2 * restDensity)
					relaxT = 1.0;*/
			}
		}
		else relaxT = 1.0;
		sphParticles._relaxTs[id] = relaxT;
	}
}

__global__ void compPoreVelocity_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles, 
	REAL3 gravity)
{
#if 0
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= poreParticles._numParticles)
		return;

	REAL visc = 0.89;

	uint phase = poreParticles._phases[id];
	REAL si = poreParticles._ss[id], sj;
	REAL hi = poreParticles._radii[phase], hj, invh;
	REAL fluidDensity = poreParticles._restFluidDensities[id];
	fluidDensity *= si;
	REAL restSolidFraction = poreParticles._restSolidFractions[phase];

	REAL3 xi, xj;
	getVector(poreParticles._xs, id, xi);

	REAL mj, dj, volumej, pj;
	REAL dist;

	REAL3 pc = make_REAL3(0.0);
	REAL3 pp = make_REAL3(0.0);

	uint2 nei;
	researchNeighbors(id, poreParticles._neis, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];
				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);

				mj = sphParticles._masses[phase];
				mj *= sj;
				dj = sphParticles._ds[nei.y];
				volumej = mj / dj;

				pj = sphParticles._ps[nei.y];
				getVector(sphParticles._xs, nei.y, xj);

				hj = (hi + hj) * SPH_RADIUS_RATIO * 0.5;
				xj = xi - xj;
				dist = Length(xj);
				if (dist < hj) {
					invh = 1.0 / hj;
					pp -= pj * volumej * SPHKernel::GKernel(dist, invh) / dist * xj;
				}
			}
			else if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				volumej = poreParticles._volumes[nei.y];
				sj = poreParticles._ss[nei.y];
				hj = poreParticles._radii[phase];
				pj = poreParticles._Ps[nei.y * 3u + 0u];

				getVector(poreParticles._xs, nei.y, xj);

				hj = (hi + hj) * SPH_RADIUS_RATIO * 0.5;
				xj = xi - xj;
				dist = Length(xj);
				if (dist < hj) {
					invh = 1.0 / hj;
					pc += max(1.0 - sj, 0.0) * pj * volumej * SPHKernel::GKernel(dist, invh) / dist * xj;
				}
			}
			else {
				/*phase = boundaryParticles._phases[nei.y];
				hj = boundaryParticles._radii[phase];
				volumej = boundaryParticles._volumes[nei.y];

				getVector(boundaryParticles._xs, nei.y, xj);

				hj = (hi + hj) * SPH_RADIUS_RATIO * 0.5;
				xj = xi - xj;
				dist = Length(xj);
				if (dist < hj) {
					invh = 1.0 / hj;
					pp -= 10.0 * volumej * SPHKernel::GKernel(dist, invh) / dist * xj;
				}*/
			}
		});

	REAL c = visc / poreParticles._Ks[id * 3u + 0u];
	//REAL3 vP = (1.0 - restSolidFraction) / c * (pc + pp + gravity); //fluidDensity*
	REAL3 vP = gravity;
	Normalize(vP);

	setVector(poreParticles._vPs, id, vP);
#else
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= poreParticles._numParticles)
		return;

	REAL visc = 0.89;

	uint phase = poreParticles._phases[id];
	REAL si = poreParticles._ss[id], sj;

	REAL hi = poreParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO;
	REAL invhi = 1.0 / hi, invhj;

	REAL fluidDensity = poreParticles._restFluidDensities[id];
	fluidDensity *= si;
	REAL restSolidFractioni = poreParticles._restSolidFractions[phase], restSolidFractionj;

	REAL3 xi, xj;
	getVector(poreParticles._xs, id, xi);

	REAL mj, dj, volumej, pj;
	REAL dist;

	REAL PressureK = 16.8;
	REAL3 pc = make_REAL3(0.0);
	REAL3 pp = make_REAL3(0.0);

	uint2 nei;
	researchNeighbors(poreParticles, id, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];
				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);

				pj = sphParticles._relaxTs[nei.y];
				volumej = sphParticles._restVolumes[phase];
				volumej *= sj * pj;
				getVector(sphParticles._xs, nei.y, xj);

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				pp -= PressureK * hi * volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
			}
			else if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];
				volumej = poreParticles._volumes[nei.y];
				restSolidFractionj = poreParticles._restSolidFractions[phase];
				//volumej *= (1.0 - restSolidFractionj);
				sj = poreParticles._ss[nei.y];
				pj = poreParticles._Ps[nei.y * 3u + 0u];

				getVector(poreParticles._xs, nei.y, xj);

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				pc += max(1.0 - sj, 0.0) * pj * (1.0 - restSolidFractionj) * volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
				//pp -= sj * PressureK * hi * volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
			}
			else {
				phase = boundaryParticles._phases[nei.y];
				hj = boundaryParticles._radii[phase];
				volumej = boundaryParticles._volumes[nei.y];

				getVector(boundaryParticles._xs, nei.y, xj);

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				pp -= PressureK * hi * volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
			}
		});

	REAL c = visc / poreParticles._Ks[id * 3u + 0u];
	REAL3 vP = (1.0 - restSolidFractioni) / c * (/*pc +*/ pp + gravity); //fluidDensity*
	//REAL3 vP = gravity;
	Normalize(vP);

	setVector(poreParticles._vPs, id, vP);
#endif
}

__global__ void lerpMassToObject_kernel(
	ClothParam cloth, PoreParticleParam particles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= particles._numParticles)
		return;

	REAL dm = particles._dms[id];

	REAL w0, w1, w2;
	uint ino = id << 1u;
	if (dm < 0.0) {
		w0 = particles._mws[ino + 0u];
		w1 = particles._mws[ino + 1u];
	}
	else {
		w0 = particles._ws[ino + 0u];
		w1 = particles._ws[ino + 1u];
	}
	w2 = 1.0 - w0 - w1;

	ino = particles._inos[id];
	ino *= 3u;
	uint ino0 = cloth._fs[ino + 0u];
	uint ino1 = cloth._fs[ino + 1u];
	uint ino2 = cloth._fs[ino + 2u];
	atomicAdd_REAL(cloth._mfs + ino0, w0 * dm);
	atomicAdd_REAL(cloth._mfs + ino1, w1 * dm);
	atomicAdd_REAL(cloth._mfs + ino2, w2 * dm);
}
__global__ void compAbsorption_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, REAL dt)
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
	
	REAL mi = sphParticles._masses[phase];
	REAL m0i = mi;
	mi *= si;

	REAL3 xi, xj;
	getVector(sphParticles._xs, id, xi);

	REAL volumei = 1.0;// / sphParticles._ds[id];
	REAL volumej, solidFraction;
	REAL3 vP;
	REAL dist, dot, dmij;

	REAL fluidMass0;
	REAL dm = 0.0;

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];

				solidFraction = poreParticles._restSolidFractions[phase];
				sj = poreParticles._ss[nei.y];

				volumej = poreParticles._volumes[nei.y];
				//volumej = poreParticles._restVolumes[phase];
				volumej *= (1.0 - solidFraction);
				fluidMass0 = volumej * restDensityi;

				volumej = poreParticles._volumes[nei.y];
				//volumej = poreParticles._restVolumes[phase];
				volumej *= (1.0 - solidFraction);

				getVector(poreParticles._xs, nei.y, xj);
				getVector(poreParticles._vPs, nei.y, vP);

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				dot = -ABSORPTION_N * Dot(vP, xj) / (dist + FLT_EPSILON);
				dmij =
					dt * ABSORPTION_K * fluidMass0 * (ABSORPTION_MAX - sj + dot) * volumei * volumej *
					SPHKernel::LaplacianKernel(dist, invhi, invhj);
				if (dmij > 0.0)
					dm += dmij;

				/*dot = 1.0 - 0.8 * Dot(vP, xj) / (dist + FLT_EPSILON);
				if (dot > 0.0) {
					dmij =
						dt * ABSORPTION_K * m0i * (ABSORPTION_MAX - sj) * dot * volumej *
						SPHKernel::LaplacianKernel(dist, invhi, invhj);
					if (dmij > 0.0) {
						dm += dmij;
					}
				}*/
			}
		});
	if (dm > 1.0e-40) {
		REAL invM = 1.0;
		if (dm > mi - m0i * MIN_VOLUME) {
			if (dm > mi - m0i * MIN_VOLUME * 0.8)
				invM = mi / dm;
			else
				invM = max((mi - m0i * (MIN_VOLUME + 1.0e-5)) / dm, 0.0);
		}
		dm = 0.0;
		researchNeighbors(sphParticles, id, nei,
			{
				if (nei.x == TYPE_PORE_PARTICLE) {
					phase = poreParticles._phases[nei.y];
					hj = poreParticles._radii[phase];

					solidFraction = poreParticles._restSolidFractions[phase];
					sj = poreParticles._ss[nei.y];

					volumej = poreParticles._volumes[nei.y];
					//volumej = poreParticles._restVolumes[phase];
					volumej *= (1.0 - solidFraction);
					fluidMass0 = volumej * restDensityi;

					volumej = poreParticles._volumes[nei.y];
					//volumej = poreParticles._restVolumes[phase];
					volumej *= (1.0 - solidFraction);

					getVector(poreParticles._xs, nei.y, xj);
					getVector(poreParticles._vPs, nei.y, vP);

					hj *= SPH_RADIUS_RATIO;
					invhj = 1.0 / hj;
					xj = xi - xj;
					dist = Length(xj);

					dot = -ABSORPTION_N * Dot(vP, xj) / (dist + FLT_EPSILON);
					dmij =
						invM * dt * ABSORPTION_K * fluidMass0 * (ABSORPTION_MAX - sj + dot) * volumei * volumej *
						SPHKernel::LaplacianKernel(dist, invhi, invhj);
					if (dmij > 0.0) {
						dm -= dmij;
						atomicAdd_REAL(poreParticles._dms + nei.y, dmij);
					}

					/*dot = 1.0 - 0.8 * Dot(vP, xj) / (dist + FLT_EPSILON);
					if (dot > 0.0) {
						dmij =
							invM * dt * ABSORPTION_K * m0i * (ABSORPTION_MAX - sj) * dot * volumej *
							SPHKernel::LaplacianKernel(dist, invhi, invhj);
						if (dmij > 0.0) {
							dm -= dmij;
							atomicAdd_REAL(poreParticles._dms + nei.y, dmij);
						}
					}*/
				}
			});
		sphParticles._ss[id] = si + dm / m0i;
	}
}

__global__ void compIsDripping_kernel(
	PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= poreParticles._numParticles)
		return;

	uint phase = poreParticles._phases[id];

	REAL hi = poreParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO;
	REAL invhi = 1.0 / hi, invhj;

	REAL3 vP;
	getVector(poreParticles._vPs, id, vP);

	REAL3 xi, xj;
	getVector(poreParticles._xs, id, xi);

	REAL dist, dot, volumej;

	uchar isDripping = 1u;
	//REAL obsWs = 0.0;

	uint2 nei;
	researchNeighbors(poreParticles, id, nei,
		{
			if (nei.x == TYPE_BOUNDARY_PARTICLE) {
				phase = boundaryParticles._phases[nei.y];
				hj = boundaryParticles._radii[phase];

				volumej = boundaryParticles._radii[phase];
				volumej = volumej * volumej * volumej * M_PI * 4.0 / 3.0;
				getVector(boundaryParticles._xs, nei.y, xj);

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				dot = -Dot(xj, vP) / (dist + FLT_EPSILON);
				//if (dot > 0.0)
				//	obsWs += dot * volumej * SPHKernel::WKernel(dist, invhi, invhj);
				if (dot > 0.5)
					isDripping = 0u;
			}
		});
	//if (obsWs > DRIPPING_THRESHOLD)
	//	isDripping = 0u;
	
	poreParticles._isDrippings[id] = isDripping;
}
__global__ void compEmissionNodeWeight_kernel(
	ClothParam cloth, PoreParticleParam particles, REAL* nodeWeights)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= particles._numParticles)
		return;

	uint isDripping = particles._isDrippings[id];
	if (isDripping) {
		/*REAL w = particles._volumes[id];

		uint ino = particles._inos[id];
		ino *= 3u;
		uint ino0 = cloth._fs[ino + 0u];
		uint ino1 = cloth._fs[ino + 1u];
		uint ino2 = cloth._fs[ino + 2u];
		atomicAdd_REAL(nodeWeights + ino0, w);
		atomicAdd_REAL(nodeWeights + ino1, w);
		atomicAdd_REAL(nodeWeights + ino2, w);*/
		uint ino = id << 1u;
		REAL w0 = particles._ws[ino + 0u];
		REAL w1 = particles._ws[ino + 1u];
		REAL w2 = 1.0 - w0 - w1;

		ino = particles._inos[id];
		ino *= 3u;
		uint ino0 = cloth._fs[ino + 0u];
		uint ino1 = cloth._fs[ino + 1u];
		uint ino2 = cloth._fs[ino + 2u];
		atomicAdd_REAL(nodeWeights + ino0, pow(w0, 1.0));
		atomicAdd_REAL(nodeWeights + ino1, pow(w1, 1.0));
		atomicAdd_REAL(nodeWeights + ino2, pow(w2, 1.0));
	}
}
__global__ void lerpEmissionMassToParticle_kernel(
	ClothParam cloth, PoreParticleParam particles, REAL* nodeWeights)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= particles._numParticles)
		return;
#if 1
	uint isDripping = particles._isDrippings[id];
	if (isDripping) {
		uint ino = id << 1u;
		REAL w0 = particles._ws[ino + 0u];
		REAL w1 = particles._ws[ino + 1u];
		REAL w2 = 1.0 - w0 - w1;

		ino = particles._inos[id];
		ino *= 3u;
		uint ino0 = cloth._fs[ino + 0u];
		uint ino1 = cloth._fs[ino + 1u];
		uint ino2 = cloth._fs[ino + 2u];
		REAL fluidMass0 = cloth._mfs[ino0];
		REAL fluidMass1 = cloth._mfs[ino1];
		REAL fluidMass2 = cloth._mfs[ino2];
		REAL maxFluidMass0 = cloth._maxFluidMass[ino0];
		REAL maxFluidMass1 = cloth._maxFluidMass[ino1];
		REAL maxFluidMass2 = cloth._maxFluidMass[ino2];

		REAL ws0 = particles._nodeWeights[ino0];
		REAL ws1 = particles._nodeWeights[ino1];
		REAL ws2 = particles._nodeWeights[ino2];

		REAL dm0 = max(fluidMass0 - maxFluidMass0, 0.0);
		REAL dm1 = max(fluidMass1 - maxFluidMass1, 0.0);
		REAL dm2 = max(fluidMass2 - maxFluidMass2, 0.0);
		dm0 *= w0 / ws0;
		dm1 *= w1 / ws1;
		dm2 *= w2 / ws2;

		REAL total = dm0 + dm1 + dm2;
		if (total > 1.0e-40) {
			particles._mfs[id] = total;

			total = 1.0 / total;
			ino = id << 1u;
			particles._mws[ino + 0u] = dm0 * total;
			particles._mws[ino + 1u] = dm1 * total;
		}
		else {
			particles._isDrippings[id] = 0u;
			particles._mfs[id] = 0.0;
		}
		/*uint ino = id << 1u;
		REAL w0 = particles._ws[ino + 0u];
		REAL w1 = particles._ws[ino + 1u];
		REAL w2 = 1.0 - w0 - w1;

		ino = particles._inos[id];
		ino *= 3u;
		uint ino0 = cloth._fs[ino + 0u];
		uint ino1 = cloth._fs[ino + 1u];
		uint ino2 = cloth._fs[ino + 2u];
		REAL fluidMass0 = cloth._mfs[ino0];
		REAL fluidMass1 = cloth._mfs[ino1];
		REAL fluidMass2 = cloth._mfs[ino2];
		REAL maxFluidMass0 = cloth._maxFluidMass[ino0];
		REAL maxFluidMass1 = cloth._maxFluidMass[ino1];
		REAL maxFluidMass2 = cloth._maxFluidMass[ino2];

		REAL dm0 = max(fluidMass0 - maxFluidMass0, 0.0);
		REAL dm1 = max(fluidMass1 - maxFluidMass1, 0.0);
		REAL dm2 = max(fluidMass2 - maxFluidMass2, 0.0);

		if (w0 > 1.0 - 1.0e-10) {
			particles._mfs[id] = dm0;
		}
		else if (w1 > 1.0 - 1.0e-10) {
			particles._mfs[id] = dm1;
		}
		else if (w2 > 1.0 - 1.0e-10) {
			particles._mfs[id] = dm2;
		}
		else {
			particles._mfs[id] = 0.0;
		}
		ino = id << 1u;
		particles._mws[ino + 0u] = w0;
		particles._mws[ino + 1u] = w1;*/
	}
	else {
		particles._mfs[id] = 0.0;
	}
#else
	uint isDripping = particles._isDrippings[id];
	if (isDripping) {
		uint ino = id << 1u;
		REAL w0 = particles._ws[ino + 0u];
		REAL w1 = particles._ws[ino + 1u];
		REAL w2 = 1.0 - w0 - w1;

		ino = particles._inos[id];
		ino *= 3u;
		uint ino0 = cloth._fs[ino + 0u];
		uint ino1 = cloth._fs[ino + 1u];
		uint ino2 = cloth._fs[ino + 2u];
		REAL fluidMass0 = cloth._mfs[ino0];
		REAL fluidMass1 = cloth._mfs[ino1];
		REAL fluidMass2 = cloth._mfs[ino2];
		REAL maxFluidMass0 = cloth._maxFluidMass[ino0];
		REAL maxFluidMass1 = cloth._maxFluidMass[ino1];
		REAL maxFluidMass2 = cloth._maxFluidMass[ino2];

		REAL ws0 = nodeWeights[ino0];
		REAL ws1 = nodeWeights[ino1];
		REAL ws2 = nodeWeights[ino2];

		REAL dm0 = max(fluidMass0 - maxFluidMass0, 0.0);
		REAL dm1 = max(fluidMass1 - maxFluidMass1, 0.0);
		REAL dm2 = max(fluidMass2 - maxFluidMass2, 0.0);
		dm0 *= pow(w0, 1.0) / ws0;
		dm1 *= pow(w1, 1.0) / ws1;
		dm2 *= pow(w2, 1.0) / ws2;

		REAL total = dm0 + dm1 + dm2;
		if (total > 1.0e-40) {
			particles._mfs[id] = total;

			total = 1.0 / total;
			ino = id << 1u;
			particles._mws[ino + 0u] = dm0 * total;
			particles._mws[ino + 1u] = dm1 * total;
		}
		else {
			particles._mfs[id] = 0.0;
			particles._isDrippings[id] = 0u;
		}
		/*uint ino = particles._inos[id];
		ino *= 3u;
		uint ino0 = cloth._fs[ino + 0u];
		uint ino1 = cloth._fs[ino + 1u];
		uint ino2 = cloth._fs[ino + 2u];
		REAL fluidMass0 = cloth._mfs[ino0];
		REAL fluidMass1 = cloth._mfs[ino1];
		REAL fluidMass2 = cloth._mfs[ino2];
		REAL maxFluidMass0 = cloth._maxFluidMass[ino0];
		REAL maxFluidMass1 = cloth._maxFluidMass[ino1];
		REAL maxFluidMass2 = cloth._maxFluidMass[ino2];

		REAL w = particles._volumes[id];
		REAL ws0 = nodeWeights[ino0];
		REAL ws1 = nodeWeights[ino1];
		REAL ws2 = nodeWeights[ino2];

		REAL dm0 = max(fluidMass0 - maxFluidMass0, 0.0);
		REAL dm1 = max(fluidMass1 - maxFluidMass1, 0.0);
		REAL dm2 = max(fluidMass2 - maxFluidMass2, 0.0);
		dm0 *= w / ws0;
		dm1 *= w / ws1;
		dm2 *= w / ws2;

		REAL total = dm0 + dm1 + dm2;
		if (total > 1.0e-40) {
			particles._mfs[id] = total;

			total = 1.0 / total;
			ino = id << 1u;
			particles._mws[ino + 0u] = dm0 * total;
			particles._mws[ino + 1u] = dm1 * total;
		}
		else {
			particles._mfs[id] = 0.0;
		}*/
	}
	else particles._mfs[id] = 0.0;
#endif
}
__global__ void compEmissionWeight_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	uint phase = sphParticles._phases[id];
	REAL si = sphParticles._ss[id], sj;

	REAL ws = 0.0;
	if (si > MIN_VOLUME) {
		REAL restDensityi = sphParticles._restDensities[phase];

		REAL m0i = sphParticles._masses[phase];
		REAL mi = m0i * si;

		REAL3 xi, xj;
		getVector(sphParticles._xs, id, xi);

		REAL overMass;
		REAL3 vP;
		REAL dist, dot, dmij;

		uint2 nei;
		researchNeighbors(sphParticles, id, nei,
			{
				if (nei.x == TYPE_PORE_PARTICLE) {
					overMass = poreParticles._mfs[nei.y];
					getVector(poreParticles._xs, nei.y, xj);
					getVector(poreParticles._vPs, nei.y, vP);

					if (overMass > 1.0e-40) {
						xj = xi - xj;
						dist = Length(xj);

						dot = EMISSION_N + Dot(xj, vP) / (dist + FLT_EPSILON);
						if (dot > 0.0)
							ws += dot * overMass;
					}
				}
			});

		if (ws > 1.0e-20)
			ws = (m0i - mi) / ws;
		else
			ws = 0.0;
	}
	sphParticles._ks[id] = ws;
}
__global__ void compEmissionToSPHParticle_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= poreParticles._numParticles)
		return;

	uint phase = poreParticles._phases[id];
	REAL overMass = poreParticles._mfs[id];

	if (overMass > 1.0e-40) {
		REAL3 xi, xj;
		getVector(poreParticles._xs, id, xi);
		REAL3 vP;
		getVector(poreParticles._vPs, id, vP);

		REAL wsj, m0j, sj;
		REAL dist, dot, dmij;

		REAL dm = 0.0;

		uint2 nei;
		researchNeighbors(poreParticles, id, nei,
			{
				if (nei.x == TYPE_SPH_PARTICLE) {
					phase = sphParticles._phases[nei.y];
					wsj = sphParticles._ks[nei.y];
					getVector(sphParticles._xs, nei.y, xj);

					if (wsj > 0.0) {
						xj = xi - xj;
						dist = Length(xj);
						dot = EMISSION_N - Dot(xj, vP) / (dist + FLT_EPSILON);
						if (dot > 0.0) {
							dmij = wsj * dot * overMass;
							dm += dmij;
						}
					}
				}
			});
		if (dm > 1.0e-40) {
			REAL invM = 1.0;
			if (dm > overMass)
				invM = overMass / dm;

			dm = 0.0;
			researchNeighbors(poreParticles, id, nei,
				{
					if (nei.x == TYPE_SPH_PARTICLE) {
						phase = sphParticles._phases[nei.y];
						m0j = sphParticles._masses[phase];
						wsj = sphParticles._ks[nei.y];
						sj = sphParticles._ss[nei.y];
						getVector(sphParticles._xs, nei.y, xj);

						if (sj > MIN_VOLUME) {
							xj = xi - xj;
							dist = Length(xj);

							dot = EMISSION_N - Dot(xj, vP) / (dist + FLT_EPSILON);
							if (dot > 0.0) {
								dmij = invM * wsj * dot * overMass;
								dm -= dmij;
								atomicAdd_REAL(sphParticles._ss + nei.y, dmij / m0j);
							}
						}
					}
				});

			overMass += dm;
			poreParticles._mfs[id] = overMass;
			poreParticles._dms[id] = dm;
		}
	}
}
__global__ void compEmissionToPoreParticle_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= poreParticles._numParticles)
		return;

	uint phase = poreParticles._phases[id];

	REAL hi = poreParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO;
	REAL invhi = 1.0 / hi, invhj;

	REAL overMass = poreParticles._mfs[id];
	REAL sphMass = sphParticles._masses[0];
	REAL minMass = sphMass * (MIN_VOLUME + 1.0e-10);

	REAL dm = 0.0;
	uchar isDripping = poreParticles._isDrippings[id];
	if (isDripping) {
		if (overMass > 0.0) {
			REAL3 vP;
			getVector(poreParticles._vPs, id, vP);

			REAL3 xi, xj;
			getVector(poreParticles._xs, id, xi);

			REAL volumej, restSolidFractionj;
			REAL dist, dot;

			REAL ws = 0.0, w;

			uint2 nei;
			researchNeighbors(poreParticles, id, nei,
				{
					if (nei.x == TYPE_PORE_PARTICLE) {
						phase = poreParticles._phases[nei.y];
						hj = poreParticles._radii[phase];

						restSolidFractionj = poreParticles._restSolidFractions[phase];
						volumej = poreParticles._volumes[nei.y];
						volumej *= (1.0 - restSolidFractionj);
						getVector(poreParticles._xs, nei.y, xj);

						hj *= SPH_RADIUS_RATIO;
						invhj = 1.0 / hj;
						xj = xi - xj;
						dist = Length(xj);

						dot = (DRIPPING_N - Dot(xj, vP) / (dist + FLT_EPSILON));
						if (dot > 0.0) {
							w = dot * volumej * SPHKernel::WKernel(dist, invhi, invhj);
							ws += w;
						}
					}
				});

			if (ws > 1.0e-40) {
				ws = 1.0 / ws;

				REAL dmij;
				researchNeighbors(poreParticles, id, nei,
					{
						if (nei.x == TYPE_PORE_PARTICLE) {
							phase = poreParticles._phases[nei.y];
							hj = poreParticles._radii[phase];

							restSolidFractionj = poreParticles._restSolidFractions[phase];
							volumej = poreParticles._volumes[nei.y];
							volumej *= (1.0 - restSolidFractionj);
							getVector(poreParticles._xs, nei.y, xj);

							hj *= SPH_RADIUS_RATIO;
							invhj = 1.0 / hj;
							xj = xi - xj;
							dist = Length(xj);

							dot = (DRIPPING_N - Dot(xj, vP) / (dist + FLT_EPSILON));
							if (dot > 0.0) {
								w = dot * volumej * SPHKernel::WKernel(dist, invhi, invhj);
								dmij = w * ws * overMass;
								dm -= dmij;
								atomicAdd_REAL(poreParticles._dms + nei.y, dmij);
							}
						}
					});
			}
		}

		overMass += dm;
		if (overMass > minMass * 0.8) {
			if (overMass < minMass)
				overMass = minMass;
			if (overMass > sphMass)
				overMass = sphMass;
			dm -= overMass;
			isDripping = 1u;
		}
		else isDripping = 0u;
		/*if (overMass >= minMass) {
			if (overMass > sphMass)
				overMass = sphMass;
			dm -= overMass;
			isDripping = 1u;
		}
		else isDripping = 0u;*/

		atomicAdd_REAL(poreParticles._dms + id, dm);
		poreParticles._mfs[id] = overMass;
		poreParticles._isDrippings[id] = isDripping;
	}
}
//__global__ void compEmissionToPoreParticle_kernel(
//	PoreParticleParam poreParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	uint phase = poreParticles._phases[id];
//
//	REAL hi = poreParticles._radii[phase], hj;
//	hi *= SPH_RADIUS_RATIO;
//	REAL invhi = 1.0 / hi, invhj;
//
//	REAL overMass = poreParticles._mfs[id];
//
//	REAL dm = 0.0;
//	uchar isDripping = poreParticles._isDrippings[id];
//	if (isDripping) {
//		if (overMass > 0.0) {
//			REAL3 vP;
//			getVector(poreParticles._vPs, id, vP);
//
//			REAL3 xi, xj;
//			getVector(poreParticles._xs, id, xi);
//
//			REAL volumej, restSolidFractionj;
//			REAL dist, dot;
//
//			REAL ws = 0.0, w;
//
//			uint2 nei;
//			researchNeighbors(poreParticles, id, nei,
//				{
//					if (nei.x == TYPE_PORE_PARTICLE) {
//						phase = poreParticles._phases[nei.y];
//						hj = poreParticles._radii[phase];
//
//						restSolidFractionj = poreParticles._restSolidFractions[phase];
//						volumej = poreParticles._volumes[nei.y];
//						volumej *= (1.0 - restSolidFractionj);
//						getVector(poreParticles._xs, nei.y, xj);
//
//						hj *= SPH_RADIUS_RATIO;
//						invhj = 1.0 / hj;
//						xj = xi - xj;
//						dist = Length(xj);
//
//						dot = (DRIPPING_N - Dot(xj, vP) / (dist + FLT_EPSILON));
//						if (dot > 0.0) {
//							w = dot * volumej * SPHKernel::WKernel(dist, invhi, invhj);
//							ws += w;
//							isDripping = 0u;
//						}
//					}
//				});
//
//			if (ws > 1.0e-40) {
//				ws = 1.0 / ws;
//
//				REAL dmij;
//				researchNeighbors(poreParticles, id, nei,
//					{
//						if (nei.x == TYPE_PORE_PARTICLE) {
//							phase = poreParticles._phases[nei.y];
//							hj = poreParticles._radii[phase];
//
//							restSolidFractionj = poreParticles._restSolidFractions[phase];
//							volumej = poreParticles._volumes[nei.y];
//							volumej *= (1.0 - restSolidFractionj);
//							getVector(poreParticles._xs, nei.y, xj);
//
//							hj *= SPH_RADIUS_RATIO;
//							invhj = 1.0 / hj;
//							xj = xi - xj;
//							dist = Length(xj);
//
//							dot = (DRIPPING_N - Dot(xj, vP) / (dist + FLT_EPSILON));
//							if (dot > 0.0) {
//								w = dot * volumej * SPHKernel::WKernel(dist, invhi, invhj);
//								dmij = w * ws * overMass;
//								dm -= dmij;
//								atomicAdd_REAL(poreParticles._dms + nei.y, dmij);
//							}
//						}
//					});
//			}
//		}
//		overMass += dm;
//		atomicAdd_REAL(poreParticles._dms + id, dm);
//		poreParticles._mfs[id] = overMass;
//		poreParticles._isDrippings[id] = isDripping;
//	}
//}
//__global__ void compEmissionCheck_kernel(
//	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	uchar isDripping = poreParticles._isDrippings[id];
//	if (isDripping) {
//		REAL overMass = poreParticles._mfs[id];
//		REAL sphMass = sphParticles._masses[0];
//		REAL minMass = sphMass * (MIN_VOLUME + 1.0e-10);
//
//		REAL dm = 0.0;
//		if (overMass > minMass * 0.8) {
//			if (overMass < minMass)
//				overMass = minMass;
//			if (overMass > sphMass)
//				overMass = sphMass;
//			dm = -overMass;
//			isDripping = 1u;
//		}
//		else isDripping = 0u;
//
//		atomicAdd_REAL(poreParticles._dms + id, dm);
//		poreParticles._mfs[id] = overMass;
//		poreParticles._isDrippings[id] = isDripping;
//	}
//}

__global__ void lerpDiffusionMassToParticle_kernel(
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
	REAL fluidMass0 = cloth._mfs[ino0];
	REAL fluidMass1 = cloth._mfs[ino1];
	REAL fluidMass2 = cloth._mfs[ino2];
	REAL s0 = cloth._maxFluidMass[ino0];
	REAL s1 = cloth._maxFluidMass[ino1];
	REAL s2 = cloth._maxFluidMass[ino2];

	REAL ws0 = particles._nodeWeights[ino0];
	REAL ws1 = particles._nodeWeights[ino1];
	REAL ws2 = particles._nodeWeights[ino2];

	s0 = fluidMass0 / (s0 + FLT_EPSILON);
	s1 = fluidMass1 / (s1 + FLT_EPSILON);
	s2 = fluidMass2 / (s2 + FLT_EPSILON);

	fluidMass0 *= w0 / ws0;
	fluidMass1 *= w1 / ws1;
	fluidMass2 *= w2 / ws2;

	REAL total = fluidMass0 + fluidMass1 + fluidMass2;
	if (total > 1.0e-40) {
		particles._mfs[id] = total;
		particles._ss[id] = w0 * s0 + w1 * s1 + w2 * s2;

		total = 1.0 / total;
		ino = id << 1u;
		particles._mws[ino + 0u] = fluidMass0 * total;
		particles._mws[ino + 1u] = fluidMass1 * total;
	}
	else {
		particles._ss[id] = 0.0;
		particles._mfs[id] = 0.0;
	}
}
__global__ void compDiffusion_kernel(ClothParam cloths, PoreParticleParam poreParticles, REAL dt)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= poreParticles._numParticles)
		return;

	uint phase = poreParticles._phases[id];

	REAL hi = poreParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO;
	REAL invhi = 1.0 / hi, invhj;

	REAL restSolidFractioni = poreParticles._restSolidFractions[phase], restSolidFractionj;
	REAL restFluidDensity = poreParticles._restFluidDensities[phase];

	REAL si = poreParticles._ss[id], sj;

	REAL volumei = poreParticles._volumes[id], volumej;
	volumei *= 1.0 - restSolidFractioni;

	REAL fluidMassi = poreParticles._mfs[id], fluidMassj;

	REAL3 xi, xj;
	getVector(poreParticles._xs, id, xi);
	REAL3 vPi, vPj;
	getVector(poreParticles._vPs, id, vPi);

	REAL dist, dot, dmij;

	REAL dm = 0.0;
	REAL grad;

	uint2 nei;

	researchNeighbors(poreParticles, id, nei,
		{
			if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];

				sj = poreParticles._ss[nei.y];

				restSolidFractionj = poreParticles._restSolidFractions[phase];

				volumej = poreParticles._volumes[nei.y];
				volumej *= 1.0 - restSolidFractionj;

				fluidMassj = poreParticles._mfs[nei.y];
				getVector(poreParticles._xs, nei.y, xj);
				getVector(poreParticles._vPs, nei.y, vPj);

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				dot = -DIFFUSION_N * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
				//dot = -DIFFUSION_N * 0.5 * Dot(vPi + vPj, xj) / (dist + FLT_EPSILON);
				grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
				dmij = (si - sj + dot) *
					(fluidMassi * volumej + fluidMassj * volumei) * grad;

				dm -= dmij;
		}
		});
	poreParticles._dms[id] = dt * dm;
	/*uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= poreParticles._numParticles)
		return;

	uint phase = poreParticles._phases[id];

	REAL hi = poreParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO;
	REAL invhi = 1.0 / hi, invhj;

	REAL restSolidFractioni = poreParticles._restSolidFractions[phase], restSolidFractionj;
	REAL restFluidDensity = poreParticles._restFluidDensities[phase];

	REAL si = poreParticles._ss[id], sj;

	REAL volumei = poreParticles._volumes[id], volumej;
	volumei *= 1.0 - restSolidFractioni;

	REAL fluidMassi = poreParticles._mfs[id], fluidMassj;

	REAL3 xi, xj;
	getVector(poreParticles._xs, id, xi);
	REAL3 vPi, vPj;
	getVector(poreParticles._vPs, id, vPi);

	REAL dist, dot, dmij;

	REAL dm = 0.0;
	REAL grad;

	uint2 nei;

	researchNeighbors(poreParticles, id, nei,
		{
			if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];

				sj = poreParticles._ss[nei.y];

				restSolidFractionj = poreParticles._restSolidFractions[phase];

				volumej = poreParticles._volumes[nei.y];
				volumej *= 1.0 - restSolidFractionj;

				fluidMassj = poreParticles._mfs[nei.y];
				getVector(poreParticles._xs, nei.y, xj);
				getVector(poreParticles._vPs, nei.y, vPj);

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				dot = -DIFFUSION_N * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
				grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
				dmij = dt * (si - sj + dot) *
					(fluidMassi * volumej + fluidMassj * volumei) * grad;

				if (dmij > 0.0)
					dm += dmij;
			}
		});

	if (dm > 1.0e-40) {
		REAL invM = 1.0;
		if (dm > fluidMassi)
			invM = fluidMassi / dm;
		dm = 0.0;
		researchNeighbors(poreParticles, id, nei,
			{
				if (nei.x == TYPE_PORE_PARTICLE) {
					phase = poreParticles._phases[nei.y];
					hj = poreParticles._radii[phase];

					sj = poreParticles._ss[nei.y];

					restSolidFractionj = poreParticles._restSolidFractions[phase];

					volumej = poreParticles._volumes[nei.y];
					volumej *= 1.0 - restSolidFractionj;

					fluidMassj = poreParticles._mfs[nei.y];
					getVector(poreParticles._xs, nei.y, xj);
					getVector(poreParticles._vPs, nei.y, vPj);

					hj *= SPH_RADIUS_RATIO;
					invhj = 1.0 / hj;
					xj = xi - xj;
					dist = Length(xj);

					dot = -DIFFUSION_N * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
					grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
					dmij = invM * dt * (si - sj + dot) *
						(fluidMassi * volumej + fluidMassj * volumei) * grad;

					if (dmij > 0.0) {
						dm -= dmij;
						atomicAdd_REAL(poreParticles._dms + nei.y, dmij);
					}
				}
			});

		atomicAdd_REAL(poreParticles._dms + id, dm);
	}*/
}
//__global__ void compDiffusion_kernel(ClothParam cloths, PoreParticleParam poreParticles, REAL dt)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= poreParticles._numParticles)
//		return;
//
//	uint phase = poreParticles._phases[id];
//
//	REAL hi = poreParticles._radii[phase], hj;
//	hi *= SPH_RADIUS_RATIO;
//	REAL invhi = 1.0 / hi, invhj;
//
//	REAL restSolidFractioni = poreParticles._restSolidFractions[phase], restSolidFractionj;
//	REAL restFluidDensity = poreParticles._restFluidDensities[phase];
//
//	REAL volumei = poreParticles._restVolumes[phase], volumej;
//	//REAL volumei = poreParticles._volumes[id], volumej;
//	volumei *= 1.0 - restSolidFractioni;
//	REAL fluidMass0i = volumei * restFluidDensity, fluidMass0j;
//
//	//volumei = poreParticles._restVolumes[phase];
//	volumei = poreParticles._volumes[id];
//	volumei *= 1.0 - restSolidFractioni;
//
//	REAL fluidMassi = poreParticles._mfs[id], fluidMassj;
//
//	REAL3 xi, xj;
//	getVector(poreParticles._xs, id, xi);
//	REAL3 vPi, vPj;
//	getVector(poreParticles._vPs, id, vPi);
//
//	REAL dist, dot, dmij;
//
//	REAL dm = 0.0;
//	REAL grad;
//
//	uint2 nei;
//
//	researchNeighbors(poreParticles, id, nei,
//		{
//			if (nei.x == TYPE_PORE_PARTICLE) {
//				phase = poreParticles._phases[nei.y];
//				hj = poreParticles._radii[phase];
//
//				restSolidFractionj = poreParticles._restSolidFractions[phase];
//
//				volumej = poreParticles._restVolumes[phase];
//				//volumej = poreParticles._volumes[nei.y];
//				volumej *= 1.0 - restSolidFractionj;
//				fluidMass0j = volumej * restFluidDensity;
//
//				//volumej = poreParticles._restVolumes[phase];
//				volumej = poreParticles._volumes[nei.y];
//				volumej *= 1.0 - restSolidFractionj;
//
//				fluidMassj = poreParticles._mfs[nei.y];
//				getVector(poreParticles._xs, nei.y, xj);
//				getVector(poreParticles._vPs, nei.y, vPj);
//
//				hj *= SPH_RADIUS_RATIO;
//				invhj = 1.0 / hj;
//				xj = xi - xj;
//				dist = Length(xj);
//
//				//dot = -DIFFUSION_N * Dot(vPi + vPj, xj) / (dist + FLT_EPSILON);
//				dot = -DIFFUSION_N * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
//				grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
//				dmij = dt * (fluidMassi / fluidMass0i - fluidMassj / fluidMass0j + dot) *
//					(hi * fluidMassi * volumej + hj * fluidMassj * volumei) * grad;
//
//				/*dot = 1.0 - 0.8 * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
//				grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
//				dmij = dt * (fluidMassi / fluidMass0i - fluidMassj / fluidMass0j) * dot * 
//					(hi * fluidMassi * volumej + hj * fluidMassj * volumei) * grad;*/
//
//				if (dmij > 0.0)
//					dm += dmij;
//			}
//		});
//
//	if (dm > 1.0e-40) {
//		REAL invM = 1.0;
//		if (dm > fluidMassi)
//			invM = fluidMassi / dm;
//		dm = 0.0;
//		researchNeighbors(poreParticles, id, nei,
//			{
//				if (nei.x == TYPE_PORE_PARTICLE) {
//					phase = poreParticles._phases[nei.y];
//					hj = poreParticles._radii[phase];
//
//					restSolidFractionj = poreParticles._restSolidFractions[phase];
//
//					volumej = poreParticles._restVolumes[phase];
//					//volumej = poreParticles._volumes[nei.y];
//					volumej *= 1.0 - restSolidFractionj;
//					fluidMass0j = volumej * restFluidDensity;
//
//					//volumej = poreParticles._restVolumes[phase];
//					volumej = poreParticles._volumes[nei.y];
//					volumej *= 1.0 - restSolidFractionj;
//
//					fluidMassj = poreParticles._mfs[nei.y];
//					getVector(poreParticles._xs, nei.y, xj);
//					getVector(poreParticles._vPs, nei.y, vPj);
//
//					hj *= SPH_RADIUS_RATIO;
//					invhj = 1.0 / hj;
//					xj = xi - xj;
//					dist = Length(xj);
//
//					//dot = -DIFFUSION_N * Dot(vPi + vPj, xj) / (dist + FLT_EPSILON);
//					dot = -DIFFUSION_N * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
//					grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
//					dmij = invM * dt * (fluidMassi / fluidMass0i - fluidMassj / fluidMass0j + dot) *
//						(hi * fluidMassi * volumej + hj * fluidMassj * volumei) * grad;
//
//					/*dot = 1.0 - 0.8 * Dot(make_REAL3(0.0, -1.0, 0.0), xj) / (dist + FLT_EPSILON);
//					grad = DIFFUSION_K * SPHKernel::LaplacianKernel(dist, invhi, invhj);
//					dmij = invM * dt * (fluidMassi / fluidMass0i - fluidMassj / fluidMass0j) * dot * 
//						(hi * fluidMassi * volumej + hj * fluidMassj * volumei) * grad;*/
//
//					if (dmij > 0.0) {
//						dm -= dmij;
//						atomicAdd_REAL(poreParticles._dms + nei.y, dmij);
//					}
//				}
//			});
//
//		atomicAdd_REAL(poreParticles._dms + id, dm);
//	}
//}

__global__ void compNewNumParticles_kernel(
	SPHParticleParam sphParticles, uint* ids)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	REAL s = sphParticles._ss[id];
	uint x = 0u;
	if (s > MIN_VOLUME * 0.5)
		x = 1u;

	if (id == 0u)
		ids[0] = 0u;
	ids[id + 1u] = x;
}
__global__ void copyOldToNew_kernel(
	SPHParticleParam sphParticles,
	REAL* newXs, REAL* newVs, REAL* newSs,
	REAL* newRelaxTs, uint* newPhases, 
	uint* ids)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= sphParticles._numParticles)
		return;

	REAL s = sphParticles._ss[id];
	if (s > MIN_VOLUME * 0.5) {
		uint newId = ids[id];
		REAL3 x, v;
		REAL s, relaxT;
		uint phase;
		getVector(sphParticles._xs, id, x);
		getVector(sphParticles._vs, id, v);
		s = sphParticles._ss[id];
		relaxT = sphParticles._relaxTs[id];
		phase = sphParticles._phases[id];

		setVector(newXs, newId, x);
		setVector(newVs, newId, v);
		newSs[newId] = s;
		newRelaxTs[newId] = relaxT;
		newPhases[newId] = phase;
	}
}
__global__ void compDrippingNum_kernel(
	PoreParticleParam poreParticles, uint* ids)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= poreParticles._numParticles)
		return;

	REAL isDripping = poreParticles._isDrippings[id];
	uint x = 0u;
	if (isDripping)
		x = 1u;

	if (id == 0u)
		ids[0] = 0u;
	ids[id + 1u] = x;
}
__global__ void generateDrippingParticle_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles,
	REAL* newXs, REAL* newVs, REAL* newSs,
	REAL* newRelaxTs, uint* newPhases, 
	uint* ids, uint oldNumParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= poreParticles._numParticles)
		return;

	REAL isDripping = poreParticles._isDrippings[id];
	if (isDripping) {
		uint phase = poreParticles._phases[id];
		REAL radius = poreParticles._radii[phase];
		REAL overMass = poreParticles._mfs[id];

		REAL3 sphX, sphV, norm;
		getVector(poreParticles._xs, id, sphX);
		getVector(poreParticles._vs, id, sphV);
		getVector(poreParticles._vPs, id, norm);

		uint sphPhase = 0u;
		REAL sphRadius = sphParticles._radii[sphPhase];
		REAL sphMass = sphParticles._masses[sphPhase];

		REAL sphS = overMass / sphMass;
		REAL sphRelaxT = 0.1;
		sphX += (S3TO1(sphS) * sphRadius + radius) * norm;

		uint sphId = ids[id];
		sphId += oldNumParticles;

		setVector(newXs, sphId, sphX);
		setVector(newVs, sphId, sphV);
		newSs[sphId] = sphS;
		newRelaxTs[sphId] = sphRelaxT;
		newPhases[sphId] = sphPhase;
	}
}

__global__ void compPorePressureForce_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
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
	REAL mi = sphParticles._masses[phase], mj;
	mi *= si;

	REAL3 xi, xj;
	REAL di = sphParticles._ds[id], dj;
	getVector(sphParticles._xs, id, xi);

	REAL3 forcei;
	getVector(sphParticles._forces, id, forcei);

	REAL volumei = mi / di, volumej;

	REAL3 forceij;
	REAL dist, pp;

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];

				sj = poreParticles._ss[nei.y];
				pp = poreParticles._Ps[nei.y * 3u + 0u];
				pp *= max(1.0 - sj, 0.0);
				getVector(poreParticles._xs, nei.y, xj);
				volumej = poreParticles._volumes[nei.y];

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				REAL w0 = poreParticles._ws[(nei.y << 1u) + 0u];
				REAL w1 = poreParticles._ws[(nei.y << 1u) + 1u];
				REAL w2 = 1.0 - w0 - w1;
				volumej *= 2.0 / (1.0 + w0 * w0 + w1 * w1 + w2 * w2);

				forceij = -pp * volumei * volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
				sumVector(poreParticles._forces, nei.y, forceij);
				forcei -= forceij;
			}
		});


	setVector(sphParticles._forces, id, forcei);
}
__global__ void compDragForce_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
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
	REAL visc = 0.89;
	const REAL Cons = 10.0 * volumei;

	REAL3 forceij;
	REAL dist, volumej;
	REAL restSolidFraction, kA, cA;

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];
				restSolidFraction = poreParticles._restSolidFractions[phase];
				getVector(poreParticles._xs, nei.y, xj);
				getVector(poreParticles._vs, nei.y, vj);
				volumej = poreParticles._volumes[nei.y];
				//volumej = volumej * restDensityi / di;

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				vj = vi - vj;

				kA = poreParticles._Ks[nei.y * 3u + 0u];
				cA = poreParticles._Cs[nei.y * 3u + 0u];
				kj = visc / kA;// +cA * Length(vj);
				kj /= (1.0 - restSolidFraction);

				REAL w0 = poreParticles._ws[(nei.y << 1u) + 0u];
				REAL w1 = poreParticles._ws[(nei.y << 1u) + 1u];
				REAL w2 = 1.0 - w0 - w1;
				volumej *= 2.0 / (1.0 + w0 * w0 + w1 * w1 + w2 * w2);

				forceij = Cons * kj * volumej * Dot(xj, vj) *
					SPHKernel::LaplacianKernel(dist, invhi, invhj) / dist * xj;
				//forceij = Cons * kj * volumej *
				//	SPHKernel::LaplacianKernel(dist, invhi, invhj) * dist * vj;

				sumVector(poreParticles._forces, nei.y, forceij);

				forcei -= forceij;
			}
		});


	setVector(sphParticles._forces, id, forcei);
}
__global__ void compPoreAttractionForce_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles)
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
	getVector(sphParticles._xs, id, xi);

	REAL di = sphParticles._ds[id], dj;

	REAL3 forcei;
	getVector(sphParticles._forces, id, forcei);

	REAL volumei = mi / di;
	const REAL Cons = 1000.0;

	REAL3 forceij;
	REAL dist, volumej;
	REAL restSolidFraction, kA, cA;

	uint2 nei;
	researchNeighbors(sphParticles, id, nei,
		{
			if (nei.x == TYPE_PORE_PARTICLE) {
				phase = poreParticles._phases[nei.y];
				hj = poreParticles._radii[phase];
				restSolidFraction = poreParticles._restSolidFractions[phase];
				getVector(poreParticles._xs, nei.y, xj);
				volumej = poreParticles._volumes[nei.y];
				mj = volumej * restDensityi;

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				REAL s = poreParticles._ss[nei.y];
				REAL force = Cons * (mi + s * 0.06) * mj * SPHKernel::WKernel(dist, invhi, invhj);

				REAL w0 = poreParticles._ws[(nei.y << 1u) + 0u];
				REAL w1 = poreParticles._ws[(nei.y << 1u) + 1u];
				REAL w2 = 1.0 - w0 - w1;
				volumej *= 2.0 / (1.0 + w0 * w0 + w1 * w1 + w2 * w2);

				forceij = -force * volumei * volumej * SPHKernel::GKernel(dist, invhi, invhj) / dist * xj;
				sumVector(poreParticles._forces, nei.y, forceij);

				forcei -= forceij;
			}
		});


	setVector(sphParticles._forces, id, forcei);
}
__global__ void compPoreAdhesionForce_kernel(
	SPHParticleParam sphParticles, PoreParticleParam poreParticles, BoundaryParticleParam boundaryParticles)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= poreParticles._numParticles)
		return;

	uint phase = poreParticles._phases[id];
	REAL si = poreParticles._ss[id], sj;

	REAL hi = poreParticles._radii[phase], hj;
	hi *= SPH_RADIUS_RATIO;
	REAL invhi = 1.0 / hi, invhj;

	REAL restFluidDensityi = poreParticles._restFluidDensities[phase];
	REAL mi = poreParticles._volumes[id], mj;
	mi *= restFluidDensityi;

	REAL wi0 = poreParticles._ws[(id << 1u) + 0u], wj0;
	REAL wi1 = poreParticles._ws[(id << 1u) + 1u], wj1;
	REAL wi2 = 1.0 - wi0 - wi1, wj2;
	REAL iPti = wi0 * wi0 + wi1 * wi1 + wi2 * wi2, iPtj;

	REAL3 xi, xj;
	getVector(poreParticles._xs, id, xi);

	REAL3 forcei;
	getVector(poreParticles._forces, id, forcei);

	REAL Ui = 0.0;
	REAL3 forceij;
	REAL dist;

	uint2 nei;
	researchNeighbors(poreParticles, id, nei,
		{
			if (nei.x == TYPE_SPH_PARTICLE) {
				phase = sphParticles._phases[nei.y];
				sj = sphParticles._ss[nei.y];
				hj = sphParticles._radii[phase];
				hj *= S3TO1(sj);

				mj = sphParticles._restVolumes[phase];
				mj *= sj;
				getVector(sphParticles._xs, nei.y, xj);

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				Ui += mj * SPHKernel::WKernel(dist, invhi, invhj);
			}
		});

	Ui = 1.0 - min(Ui, 1.0);
	const REAL Const = Ui * 0.16;

	researchNeighbors(poreParticles, id, nei,
		{
			if (nei.x != TYPE_SPH_PARTICLE) {
				if (nei.x == TYPE_PORE_PARTICLE) {
					phase = poreParticles._phases[nei.y];
					sj = poreParticles._ss[nei.y];
					hj = poreParticles._radii[phase];

					mj = poreParticles._volumes[nei.y];
					getVector(poreParticles._xs, nei.y, xj);

					mj *= restFluidDensityi;
					wj0 = poreParticles._ws[(nei.y << 1u) + 0u];
					wj1 = poreParticles._ws[(nei.y << 1u) + 1u];
					wj2 = 1.0 - wj0 - wj1;
					iPtj = wj0 * wj0 + wj1 * wj1 + wj2 * wj2;
					sj += si;
				}
				else {
					phase = boundaryParticles._phases[nei.y];
					hj = boundaryParticles._radii[phase];

					mj = boundaryParticles._volumes[nei.y];
					getVector(boundaryParticles._xs, nei.y, xj);

					mj *= restFluidDensityi;
					iPtj = 1.0;
					sj = si;
				}

				hj *= SPH_RADIUS_RATIO;
				invhj = 1.0 / hj;
				xj = xi - xj;
				dist = Length(xj);

				forceij = Const / (iPti + iPtj) * sj * mi * mj *
					SPHKernel::selfCohesionKernel(dist, invhi, invhj) / dist * xj;

				forcei -= forceij;
			}
		});


	setVector(poreParticles._forces, id, forcei);
}

#endif