#ifndef __HASH_FUNCTIONS_CUH__
#define __HASH_FUNCTIONS_CUH__

#include "../include/CUDA_Custom/DeviceManager.cuh"

inline __forceinline__ __device__ uint splitBy3(uint x) {
	/*x = x & 0x1fffff;
	x = (x | x << 32ull) & 0x1f00000000ffff;
	x = (x | x << 16ull) & 0x1f0000ff0000ff;
	x = (x | x << 8ull) & 0x100f00f00f00f00f;
	x = (x | x << 4ull) & 0x10c30c30c30c30c3;
	x = (x | x << 2ull) & 0x1249249249249249;*/
	if (x == 1024u) x--;
	x = (x | x << 16u) & 0b00000011000000000000000011111111;
	x = (x | x << 8u) & 0b00000011000000001111000000001111;
	x = (x | x << 4u) & 0b00000011000011000011000011000011;
	x = (x | x << 2u) & 0b00001001001001001001001001001001;
	return x;
}
inline __forceinline__ __device__ uint getGridIndex(int3 p, uint3 size) {
	return __umul24(__umul24(((uint)p.z) & (size.z - 1), size.y) + (((uint)p.y) & (size.y - 1)), size.x) + (((uint)p.x) & (size.x - 1));
}
inline __forceinline__ __device__ uint getZindex(int3 p, uint3 size) {
	uint x = ((uint)(p.x + (size.x >> 1u))) & (size.x - 1u);
	uint y = ((uint)(p.y + (size.y >> 1u))) & (size.y - 1u);
	uint z = ((uint)(p.z + (size.z >> 1u))) & (size.z - 1u);

	return splitBy3(x) | splitBy3(y) << 1u | splitBy3(z) << 2u;

	/*uint id = 0u, i = 0u, bin0, bin1;
	while ((1u << i) <= x || (1u << i) <= y || (1u << i) <= z) {
		bin0 = 1u << i;
		bin1 = i << 1u;
		id |= (x & bin0) << bin1;
		id |= (y & bin0) << bin1 + 1u;
		id |= (z & bin0) << bin1 + 2u;
		i++;
	}

	return id;*/
}
inline __forceinline__ __device__ int3 getGridPos(const REAL3& x, REAL radius) {
	radius = 1.0 / radius;
	int3 p = make_int3(
		int(x.x * radius),
		int(x.y * radius),
		int(x.z * radius));
	return p;
}

#endif