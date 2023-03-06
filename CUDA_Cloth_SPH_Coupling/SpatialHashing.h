#ifndef __SPATIAL_HASHING_H__
#define __SPATIAL_HASHING_H__

#pragma once
#include "SPHParticle.h"
#include "PoreParticle.h"
#include "BoundaryParticle.h"

//#define HASH_TIMER

class SpatialHashing {
public:
	SpatialHashParam	_param;
public:
	Dvector<uint2>		_ids;
	Dvector<uint>		_keys;
	Dvector<uint>		_istarts;
	Dvector<uint>		_iends;
public:
	uint3				_size;
	REAL				_radius;
private:
	Particle			*_particles;
	uint				_numHash;
	uint				_numParticles;
public:
	SpatialHashing() {}
	SpatialHashing(const uint3& size) {
		init(size);
	}
	~SpatialHashing() {}
public:
	inline void setParam(void) {
		_param._ids = _ids._list;
		_param._keys = _keys._list;
		_param._istarts = _istarts._list;
		_param._iends = _iends._list;
		_param._size = _size;
		_param._radius = _radius;
		_param._numParticles = _numParticles;
	}
	inline void init(const uint3& size) {
		_size = size;
		_radius = 0.0;
		_numHash = size.x * size.y * size.z;
		_istarts.resize(_numHash);
		_iends.resize(_numHash);

		setParam();
	}
	inline void setRadius(const REAL radius) {
		_radius = radius;
		_param._radius = radius;
		printf("%f\n", radius);
	}
	inline void initParticle(uint numParticles) {
		_keys.resize(numParticles);
		_ids.resize(numParticles);
		_istarts.memset(0xffffffff);

		_numParticles = numParticles;

		setParam();
	}
	void freeParticleIds(void) {
		_ids.clear();
		_keys.clear();
		_numParticles = 0u;
	}
	inline void clear(void) {
		freeParticleIds();
		_istarts.clear();
		_iends.clear();
	}
public:
	void sort(SPHParticle* sphParticles);
	void insert(SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles);
	void getNeighbors(SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles);
};

#endif