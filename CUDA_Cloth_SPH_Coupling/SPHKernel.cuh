#ifndef __SPH_KERNEL_CUH__
#define __SPH_KERNEL_CUH__

#include "Params.h"
#include "../include/CUDA_Custom/DeviceManager.cuh"

#define researchNeighbors(PARAM, ID, INO, X) \
	{uint ISTART = PARAM._ineis[ID]; \
	uint IEND = PARAM._ineis[ID + 1u]; \
	for (uint I = ISTART; I < IEND; I++) { \
		INO = PARAM._neis[I]; \
		X }}

namespace SPHKernel {
	static inline __forceinline__ __host__ __device__
		REAL WKernel(REAL ratio)
	{
		if (ratio < 0.0 || ratio >= 1.0)
			return 0.0;

#if 1
		REAL tmp = 1.0 - ratio * ratio;
		return CONST_W * tmp * tmp * tmp;
#else
		REAL result;
		if (ratio <= 0.5) {
			REAL tmp2 = ratio * ratio;
			result = 6.0 * tmp2 * (ratio - 1.0) + 1.0;
		}
		else {
			result = 1.0 - ratio;
			result = 2.0 * result * result * result;
		}
		return CONST_CUBICW * result;
#endif
	}
	static inline __forceinline__ __host__ __device__
		REAL GKernel(REAL ratio)
	{
		if (ratio < 1.0e-40 || ratio >= 1.0)
			return 0.0;
#if 1
		REAL tmp = 1.0 - ratio;
		return CONST_G * tmp * tmp;
#else
		REAL result;
		if (ratio <= 0.5)
			result = ratio * (3.0 * ratio - 2.0);
		else {
			result = 1.0 - ratio;
			result = -result * result;
		}
		return CONST_CUBICG * result;
#endif
	}
	static inline __forceinline__ __host__ __device__
		REAL LaplacianKernel(REAL ratio)
	{
		if (ratio < 0.0 || ratio >= 1.0)
			return 0.0;

		REAL tmp = 1.0 - ratio;
		return CONST_LAPLACIAN * tmp;
	}
	static inline __forceinline__ __host__ __device__
		REAL cohesionKernel(REAL ratio)
	{
		if (ratio <= 1.0e-40 || ratio >= 1.0)
			return 0.0;

		REAL result = (1.0 - ratio) * ratio;
		result = result * result * result;
		if (ratio <= 0.5)
			result += result - 0.015625;

		return CONST_COHESION * result;
	}
	static inline __forceinline__ __host__ __device__
		REAL addhesionKernel(REAL ratio)
	{
		if (ratio <= 0.5 || ratio >= 1.0)
			return 0.0;

		REAL result = pow(-4.0 * ratio * ratio + 6.0 * ratio - 2.0, 0.25);
		return CONST_ADHESION * result;
	}
	static inline __forceinline__ __host__ __device__
		REAL selfCohesionKernel(REAL ratio)
	{
		if (ratio <= 0.5 || ratio >= 1.0)
			return 0.0;

		REAL result = (1.0 - ratio * 0.5) * ratio * 0.5;
		result = result * result * result;

		return CONST_COHESION * result;
	}

	static inline __forceinline__ __host__ __device__
		REAL WKernel(REAL dist, REAL invh) 
	{
		return WKernel(dist * invh) * invh * invh * invh;
	}
	static inline __forceinline__ __host__ __device__
		REAL GKernel(REAL dist, REAL invh) 
	{
		REAL tmp = invh * invh;
		return GKernel(dist * invh) * tmp * tmp;
	}
	static inline __forceinline__ __host__ __device__
		REAL LaplacianKernel(REAL dist, REAL invh) 
	{
		REAL tmp = invh * invh;

		//return LaplacianKernel(dist * invh) * tmp * tmp * invh;

		dist = dist * invh;
		return -GKernel(dist) / (dist * dist + 0.01) * tmp * tmp * invh;
	}
	static inline __forceinline__ __host__ __device__
		REAL cohesionKernel(REAL dist, REAL invh)
	{
		return cohesionKernel(dist * invh) * invh * invh * invh;
	}
	static inline __forceinline__ __host__ __device__
		REAL addhesionKernel(REAL dist, REAL invh) 
	{
		return addhesionKernel(dist * invh) * invh * invh * invh;
	}
	static inline __forceinline__ __host__ __device__
		REAL selfCohesionKernel(REAL dist, REAL invh)
	{
		return selfCohesionKernel(dist * invh) * invh * invh * invh;
	}

	static inline __forceinline__ __host__ __device__
		REAL WKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return (WKernel(dist, invhi) + WKernel(dist, invhj)) * 0.5;
	}
	static inline __forceinline__ __host__ __device__
		REAL GKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return (GKernel(dist, invhi) + GKernel(dist, invhj)) * 0.5;
	}
	static inline __forceinline__ __host__ __device__
		REAL LaplacianKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return (LaplacianKernel(dist, invhi) + LaplacianKernel(dist, invhj)) * 0.5;
	}
	static inline __forceinline__ __host__ __device__
		REAL cohesionKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return (cohesionKernel(dist, invhi) + cohesionKernel(dist, invhj)) * 0.5;
	}
	static inline __forceinline__ __host__ __device__
		REAL addhesionKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return (addhesionKernel(dist, invhi) + addhesionKernel(dist, invhj)) * 0.5;
	}
	static inline __forceinline__ __host__ __device__
		REAL selfCohesionKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return (selfCohesionKernel(dist, invhi) + selfCohesionKernel(dist, invhj)) * 0.5;
	}
	static inline __forceinline__ __host__ __device__
		REAL GKernel(REAL dist, REAL hi, REAL hj, REAL invhi, REAL invhj)
	{
		return (hi * GKernel(dist, invhi) + hj * GKernel(dist, invhj)) * 0.5;
	}
	/*static inline __forceinline__ __host__ __device__
		REAL WKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return WKernel(dist, 2.0 / (1.0 / invhi + 1.0 / invhj));
	}
	static inline __forceinline__ __host__ __device__
		REAL GKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return GKernel(dist, 2.0 / (1.0 / invhi + 1.0 / invhj));
	}
	static inline __forceinline__ __host__ __device__
		REAL LaplacianKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return LaplacianKernel(dist, 2.0 / (1.0 / invhi + 1.0 / invhj));
	}
	static inline __forceinline__ __host__ __device__
		REAL cohesionKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return cohesionKernel(dist, 2.0 / (1.0 / invhi + 1.0 / invhj));
	}
	static inline __forceinline__ __host__ __device__
		REAL addhesionKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return addhesionKernel(dist, 2.0 / (1.0 / invhi + 1.0 / invhj));
	}
	static inline __forceinline__ __host__ __device__
		REAL selfCohesionKernel(REAL dist, REAL invhi, REAL invhj)
	{
		return selfCohesionKernel(dist, 2.0 / (1.0 / invhi + 1.0 / invhj));
	}*/
}

#endif