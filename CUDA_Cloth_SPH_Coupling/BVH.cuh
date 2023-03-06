#ifndef __BVH_CUH__
#define __BVH_CUH__

#include "HashFunc.cuh"
#include "BVH.h"

//-------------------------------------------------------------------

//inline __global__ void initBVHInfo_kernel(
//	uint* fs, REAL* ns, TriInfo* infos, BVHParam bvh)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= bvh._numFaces)
//		return;
//
//	if (id == 0u) {
//		bvh._mins[0][0] = DBL_MAX;
//		bvh._mins[1][0] = DBL_MAX;
//		bvh._mins[2][0] = DBL_MAX;
//		bvh._maxs[0][0] = -DBL_MAX;
//		bvh._maxs[1][0] = -DBL_MAX;
//		bvh._maxs[2][0] = -DBL_MAX;
//		bvh._levels[0] = 0u;
//	}
//
//	uint ino = id * 3u;
//	uint ino0 = fs[ino + 0u];
//	uint ino1 = fs[ino + 1u];
//	uint ino2 = fs[ino + 2u];
//	REAL3 p0, p1, p2;
//	ino0 *= 3u; ino1 *= 3u; ino2 *= 3u;
//	p0.x = ns[ino0 + 0u]; p0.y = ns[ino0 + 1u]; p0.z = ns[ino0 + 2u];
//	p1.x = ns[ino1 + 0u]; p1.y = ns[ino1 + 1u]; p1.z = ns[ino1 + 2u];
//	p2.x = ns[ino2 + 0u]; p2.y = ns[ino2 + 1u]; p2.z = ns[ino2 + 2u];
//	p0 += p1 + p2;
//
//	ino = bvh._size - bvh._numFaces + id;
//	bvh._mins[0][ino] = p0.x;
//	bvh._mins[1][ino] = p0.y;
//	bvh._mins[2][ino] = p0.z;
//
//	TriInfo info;
//	info._face = id;
//	info._id = 0u;
//	infos[id] = info;
//}
//inline __global__ void InitMinMaxKernel(BVHParam bvh)
//{
//	__shared__ REAL s_mins[3][MAX_BLOCKSIZE];
//	__shared__ REAL s_maxs[3][MAX_BLOCKSIZE];
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= bvh._numFaces) {
//		s_mins[0][threadIdx.x] = DBL_MAX;
//		s_mins[1][threadIdx.x] = DBL_MAX;
//		s_mins[2][threadIdx.x] = DBL_MAX;
//		s_maxs[0][threadIdx.x] = -DBL_MAX;
//		s_maxs[1][threadIdx.x] = -DBL_MAX;
//		s_maxs[2][threadIdx.x] = -DBL_MAX;
//		return;
//	}
//
//	uint ibvh = bvh._size - bvh._numFaces + id;
//	s_mins[0][threadIdx.x] = bvh._mins[0][ibvh];
//	s_mins[1][threadIdx.x] = bvh._mins[1][ibvh];
//	s_mins[2][threadIdx.x] = bvh._mins[2][ibvh];
//	s_maxs[0][threadIdx.x] = s_mins[0][threadIdx.x];
//	s_maxs[1][threadIdx.x] = s_mins[1][threadIdx.x];
//	s_maxs[2][threadIdx.x] = s_mins[2][threadIdx.x];
//	for (uint s = BLOCKSIZE >> 1u; s > 32u; s >>= 1u) {
//		__syncthreads();
//		if (threadIdx.x < s) {
//			for (uint i = 0u; i < 3u; i++) {
//				if (s_mins[i][threadIdx.x] > s_mins[i][threadIdx.x + s])
//					s_mins[i][threadIdx.x] = s_mins[i][threadIdx.x + s];
//				if (s_maxs[i][threadIdx.x] < s_maxs[i][threadIdx.x + s])
//					s_maxs[i][threadIdx.x] = s_maxs[i][threadIdx.x + s];
//			}
//		}
//	}
//	__syncthreads();
//
//	if (threadIdx.x < 32u) {
//		warpMin(s_mins[0], threadIdx.x);
//		warpMin(s_mins[1], threadIdx.x);
//		warpMin(s_mins[2], threadIdx.x);
//		warpMax(s_maxs[0], threadIdx.x);
//		warpMax(s_maxs[1], threadIdx.x);
//		warpMax(s_maxs[2], threadIdx.x);
//		if (threadIdx.x == 0) {
//			atomicMin_REAL(bvh._mins[0], s_mins[0][threadIdx.x]);
//			atomicMin_REAL(bvh._mins[1], s_mins[1][threadIdx.x]);
//			atomicMin_REAL(bvh._mins[2], s_mins[2][threadIdx.x]);
//			atomicMax_REAL(bvh._maxs[0], s_maxs[0][threadIdx.x]);
//			atomicMax_REAL(bvh._maxs[1], s_maxs[1][threadIdx.x]);
//			atomicMax_REAL(bvh._maxs[2], s_maxs[2][threadIdx.x]);
//		}
//	}
//}
//inline __global__ void updateBVHInfo_kernel(
//	TriInfo* infos, BVHParam bvh)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= bvh._numFaces)
//		return;
//
//	TriInfo info = infos[id];
//	uint ino = info._id;
//
//	REAL3 min, max;
//	min.x = bvh._mins[0][ino];
//	min.y = bvh._mins[1][ino];
//	min.z = bvh._mins[2][ino];
//	max.x = bvh._maxs[0][ino];
//	max.y = bvh._maxs[1][ino];
//	max.z = bvh._maxs[2][ino];
//	max -= min;
//
//	uint elem = 0u;
//	if (max.x < max.y) {
//		max.x = max.y;
//		elem = 1u;
//	}
//	if (max.x < max.z)
//		elem = 2u;
//
//	ino = bvh._size - bvh._numFaces + info._face;
//	info._pos = bvh._mins[elem][ino];
//	infos[id] = info;
//}
//inline __global__ void updateMinMax_kernel(
//	BVHParam bvh, uint level, uint size)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= size)
//		return;
//
//	uint ino;
//	ino = (1u << level) - 1u + id;
//
//	uint istart = bvh._levels[ino];
//	uint iend;
//	uint half;
//	if (id != size - 1u) 
//		iend = bvh._levels[ino + 1u];
//	else 
//		iend = bvh._numFaces;
//	half = iend - istart;
//
//	uint minHalf = 1u << bvh._maxLevel - level - 2u;
//	uint maxHalf = 1u << bvh._maxLevel - level - 1u;
//	if (level > bvh._maxLevel - 2u) {
//		minHalf = 0;
//		if (level > bvh._maxLevel - 1u)
//			maxHalf = 0;
//	}
//	if (half < minHalf + maxHalf)
//		half -= minHalf;
//	else
//		half = maxHalf;
//
//	REAL3 min, max;
//	min.x = bvh._mins[0][ino];
//	min.y = bvh._mins[1][ino];
//	min.z = bvh._mins[2][ino];
//	max.x = bvh._maxs[0][ino];
//	max.y = bvh._maxs[1][ino];
//	max.z = bvh._maxs[2][ino];
//
//	ino = (ino << 1u) + 1u;
//	bvh._levels[ino] = istart;
//	bvh._mins[0][ino] = min.x;
//	bvh._mins[1][ino] = min.y;
//	bvh._mins[2][ino] = min.z;
//	bvh._maxs[0][ino] = max.x;
//	bvh._maxs[1][ino] = max.y;
//	bvh._maxs[2][ino] = max.z;
//	ino++;
//	bvh._levels[ino] = istart + half;
//	bvh._mins[0][ino] = min.x;
//	bvh._mins[1][ino] = min.y;
//	bvh._mins[2][ino] = min.z;
//	bvh._maxs[0][ino] = max.x;
//	bvh._maxs[1][ino] = max.y;
//	bvh._maxs[2][ino] = max.z;
//}
//inline __global__ void subdivBVH_kernel(
//	TriInfo* infos, BVHParam bvh)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= bvh._numFaces)
//		return;
//
//	TriInfo info = infos[id];
//	uint ino = info._id;
//
//	REAL3 min, max;
//	min.x = bvh._mins[0][ino];
//	min.y = bvh._mins[1][ino];
//	min.z = bvh._mins[2][ino];
//	max.x = bvh._maxs[0][ino];
//	max.y = bvh._maxs[1][ino];
//	max.z = bvh._maxs[2][ino];
//	max -= min;
//
//	uint elem = 0u;
//	if (max.x < max.y) {
//		max.x = max.y;
//		elem = 1u;
//	}
//	if (max.x < max.z)
//		elem = 2u;
//
//	ino = (ino << 1u) + 2u;
//	uint pivot = bvh._levels[ino];
//	if (id < pivot) {
//		info._id = ino - 1u;
//		if (id == pivot - 1u)
//			bvh._maxs[elem][info._id] = info._pos;
//	}
//	else {
//		info._id = ino;
//		if (id == pivot)
//			bvh._mins[elem][info._id] = info._pos;
//	}
//	infos[id] = info;
//}
//inline __global__ void buildBVH_kernel(
//	TriInfo* infos, BVHParam bvh)
//{
//	uint id = blockDim.x * blockIdx.x + threadIdx.x;
//	if (id >= bvh._size)
//		return;
//
//	bvh._levels[id] = Log2(id + 1u);
//	
//	uint ileaf = bvh._size - bvh._numFaces;
//	if (id >= ileaf) {
//		uint fid;
//		id -= ileaf;
//		fid = id + bvh._pivot;
//		if (fid >= bvh._numFaces)
//			fid -= bvh._numFaces;
//
//		TriInfo info = infos[fid];
//		bvh._faces[id] = info._face;
//	}
//}

inline __global__ void compDiameter_kernel(
	uint* fs, REAL* ns, REAL* diameter, uint numFaces)
{
	__shared__ REAL s_diameter[BLOCKSIZE];
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numFaces) {
		s_diameter[threadIdx.x] = 0.0;
		return;
	}

	uint ino = id * 3u;
	uint ino0 = fs[ino + 0u];
	uint ino1 = fs[ino + 1u];
	uint ino2 = fs[ino + 2u];
	REAL3 n0, n1, n2;
	getVector(ns, ino0, n0);
	getVector(ns, ino1, n1);
	getVector(ns, ino2, n2);

	float xs[3], ys[3], zs[3];
	xs[0] = fabsf((float)n0.x - (float)n1.x);
	xs[1] = fabsf((float)n0.x - (float)n2.x);
	xs[2] = fabsf((float)n1.x - (float)n2.x);
	ys[0] = fabsf((float)n0.y - (float)n1.y);
	ys[1] = fabsf((float)n0.y - (float)n2.y);
	ys[2] = fabsf((float)n1.y - (float)n2.y);
	zs[0] = fabsf((float)n0.z - (float)n1.z);
	zs[1] = fabsf((float)n0.z - (float)n2.z);
	zs[2] = fabsf((float)n1.z - (float)n2.z);
	for (int i = 1; i < 3; i++) {
		if (xs[0] < xs[i])
			xs[0] = xs[i];
		if (ys[0] < ys[i])
			ys[0] = ys[i];
		if (zs[0] < zs[i])
			zs[0] = zs[i];
	}
	if (xs[0] > ys[0] && xs[0] > zs[0])
		s_diameter[threadIdx.x] = xs[0];
	else if (ys[0] > zs[0])
		s_diameter[threadIdx.x] = ys[0];
	else
		s_diameter[threadIdx.x] = zs[0];
	/*REAL3 n01 = n1 - n0;
	REAL3 n02 = n2 - n0;
	REAL area = Length(Cross(n01, n02));
	REAL l01 = Length(n01);
	REAL l02 = Length(n02);
	REAL l12 = Length(n01 - n02);
	if (l01 > l02 && l01 > l12) 
		s_diameter[threadIdx.x] = area / l01;
	else if (l02 > l12) 
		s_diameter[threadIdx.x] = area / l02;
	else 
		s_diameter[threadIdx.x] = area / l12;*/

	for (uint s = blockDim.x >> 1u; s > 32u; s >>= 1u) {
		__syncthreads();
		if (threadIdx.x < s) {
			if (s_diameter[threadIdx.x] < s_diameter[threadIdx.x + s])
				s_diameter[threadIdx.x] = s_diameter[threadIdx.x + s];
		}
	}
	__syncthreads();

	if (threadIdx.x < 32u) {
		warpMax(s_diameter, threadIdx.x);
		if (threadIdx.x == 0)
			atomicMax_REAL(diameter, s_diameter[0]);
	}
}
inline __global__ void initTriInfo_kernel(
	uint* fs, REAL* ns, TriInfo* infos, REAL* diameter, uint numFaces)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= numFaces)
		return;

	uint ino = id * 3u;
	uint ino0 = fs[ino + 0u];
	uint ino1 = fs[ino + 1u];
	uint ino2 = fs[ino + 2u];
	REAL3 n0, n1, n2;
	getVector(ns, ino0, n0);
	getVector(ns, ino1, n1);
	getVector(ns, ino2, n2);
	
	REAL3 cen = (n0 + n1 + n2) / 3.0;
	int3 gridPos = getGridPos(cen, 0.005);
	TriInfo info;
	info._id = id;
	info._pos = getZindex(gridPos, make_uint3(512u, 512u, 512u));
	infos[id] = info;
}
inline __global__ void buildBVH_kernel(
	TriInfo* infos, BVHParam bvh)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= bvh._size)
		return;

	bvh._levels[id] = Log2(id + 1u);

	uint ileaf = bvh._size - bvh._numFaces;
	if (id >= ileaf) {
		uint fid;
		id -= ileaf;
		fid = id + bvh._pivot;
		if (fid >= bvh._numFaces)
			fid -= bvh._numFaces;

		TriInfo info = infos[fid];
		bvh._faces[id] = info._id;
	}
}
//-------------------------------------------------------------------
inline __device__ void RefitBVHLeaf(
	uint fid, AABB& aabb,
	const uint* fs, const REAL* ns,
	const uint* nodePhases, const REAL* thicknesses)
{
	fid *= 3u;
	uint ino0 = fs[fid + 0u];
	uint ino1 = fs[fid + 1u];
	uint ino2 = fs[fid + 2u];
	uint phase = nodePhases[ino0];
	REAL delta = thicknesses[phase];
	delta *= COL_CLEARANCE_RATIO;
	REAL3 p0, p1, p2;
	getVector(ns, ino0, p0);
	getVector(ns, ino1, p1);
	getVector(ns, ino2, p2);

	setAABB(aabb, p0, delta);
	addAABB(aabb, p1, delta);
	addAABB(aabb, p2, delta);
}
inline __device__ void RefitBVHLeaf(
	uint fid, AABB& aabb,
	const uint* fs, const REAL* ns, const REAL* vs,
	const uint* nodePhases, const REAL* thicknesses,
	const REAL dt)
{
	fid *= 3u;
	uint ino0 = fs[fid + 0u];
	uint ino1 = fs[fid + 1u];
	uint ino2 = fs[fid + 2u];
	uint phase = nodePhases[ino0];
	REAL delta = thicknesses[phase];

	REAL3 p0, p1, p2;
	getVector(ns, ino0, p0);
	getVector(ns, ino1, p1);
	getVector(ns, ino2, p2);

	setAABB(aabb, p0, COL_CCD_THICKNESS);
	addAABB(aabb, p1, COL_CCD_THICKNESS);
	addAABB(aabb, p2, COL_CCD_THICKNESS);

	REAL3 v0, v1, v2;
	getVector(vs, ino0, v0);
	getVector(vs, ino1, v1);
	getVector(vs, ino2, v2);
	p0 += v0 * dt; p1 += v1 * dt; p2 += v2 * dt;

	addAABB(aabb, p0, delta);
	addAABB(aabb, p1, delta);
	addAABB(aabb, p2, delta);
}
inline __global__ void RefitBVHKernel(
	BVHParam bvh, uint num)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= num)
		return;

	uint ind = num - 1u + id;
	uint ichild = (ind << 1) + 1u;
	AABB parent, lchild, rchild;
	getBVHAABB(lchild, bvh, ichild);
	getBVHAABB(rchild, bvh, ichild + 1u);

	setAABB(parent, lchild);
	addAABB(parent, rchild);
	updateBVHAABB(bvh, parent, ind);
}
inline __global__ void RefitNodeBVHKernel(
	BVHParam bvh, uint level)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	uint currLev = level;
	uint ind0, ind, ichild;
	AABB parent, lchild, rchild;
	while (currLev > 5u) {
		ind0 = (1u << currLev--);
		if (id < ind0--) {
			ind = ind0 + id;
			ichild = (ind << 1u) + 1u;
			getBVHAABB(lchild, bvh, ichild);
			getBVHAABB(rchild, bvh, ichild + 1);

			setAABB(parent, lchild);
			addAABB(parent, rchild);
			updateBVHAABB(bvh, parent, ind);
		}
		__syncthreads();
	}
	while (currLev != 0xffffffff) {
		ind0 = (1u << currLev--);
		if (id < ind0--) {
			ind = ind0 + id;
			ichild = (ind << 1u) + 1u;
			getBVHAABB(lchild, bvh, ichild);
			getBVHAABB(rchild, bvh, ichild + 1u);

			setAABB(parent, lchild);
			addAABB(parent, rchild);
			updateBVHAABB(bvh, parent, ind);
		}
	}
}

inline __global__ void RefitLeafBVHKernel(
	uint* fs, REAL* ns, uint* nodePhases, REAL* thicknesses, BVHParam bvh, uint num)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= num)
		return;

	uint ind = num - 1u + id;
	uint ileaf = bvh._size - bvh._numFaces;
	AABB node;
	if (ind < ileaf) {
		uint ichild = (ind << 1u) + 1u;
		AABB lchild, rchild;
		uint lfid = bvh._faces[ichild - ileaf];
		uint rfid = bvh._faces[ichild + 1u - ileaf];
		RefitBVHLeaf(lfid, lchild, fs, ns, nodePhases, thicknesses);
		RefitBVHLeaf(rfid, rchild, fs, ns, nodePhases, thicknesses);
		updateBVHAABB(bvh, lchild, ichild);
		updateBVHAABB(bvh, rchild, ichild + 1u);

		setAABB(node, lchild);
		addAABB(node, rchild);
	}
	else {
		uint fid = bvh._faces[ind - ileaf];
		RefitBVHLeaf(fid, node, fs, ns, nodePhases, thicknesses);
	}
	updateBVHAABB(bvh, node, ind);
}
inline __global__ void RefitLeafBVHKernel(
	uint* fs, REAL* ns, REAL* vs, uint* nodePhases, REAL* thicknesses, BVHParam bvh, uint num,
	const REAL dt)
{
	uint id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id >= num)
		return;

	uint ind = num - 1u + id;
	uint ileaf = bvh._size - bvh._numFaces;
	AABB node;
	if (ind < ileaf) {
		uint ichild = (ind << 1u) + 1u;
		AABB lchild, rchild;
		uint lfid = bvh._faces[ichild - ileaf];
		uint rfid = bvh._faces[ichild + 1u - ileaf];
		RefitBVHLeaf(lfid, lchild, fs, ns, vs, nodePhases, thicknesses, dt);
		RefitBVHLeaf(rfid, rchild, fs, ns, vs, nodePhases, thicknesses, dt);
		updateBVHAABB(bvh, lchild, ichild);
		updateBVHAABB(bvh, rchild, ichild + 1u);

		setAABB(node, lchild);
		addAABB(node, rchild);
	}
	else {
		uint fid = bvh._faces[ind - ileaf];
		RefitBVHLeaf(fid, node, fs, ns, vs, nodePhases, thicknesses, dt);
	}
	updateBVHAABB(bvh, node, ind);
}

#endif