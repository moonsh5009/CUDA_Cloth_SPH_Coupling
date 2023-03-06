#ifndef __SYSTEM_H__
#define __SYSTEM_H__

#pragma once
#include "CollisionSolver.h"
#include "ParticleSampling.h"
#include "PorousSolver.h"
#include "SPHSolver.h"

class System {
public:
	Cloth				*_cloths;
	Obstacle			*_obstacles;
public:
	BoundaryParticle	*_boundaryParticles;
	SPHParticle			*_sphParticles;
	PoreParticle		*_poreParticles;
public:
	SpatialHashing		*_hash;
public:
	AABB				_boundary;
public:
	REAL3				_gravity;
	REAL				_dt;
	REAL				_invdt;
	uint				_frame;
	uint				_subStep;
public:
	ContactElems		_ceParam;
public:
	System() {}
	System(REAL3& gravity, REAL dt) {
		init(gravity, dt);
	}
	~System() {}
public:
	uint	numFaces(void) const {
		return _cloths->_numFaces + _obstacles->_numFaces;
	}
	uint	numParticles(void) const {
		return _boundaryParticles->_numParticles + _sphParticles->_numParticles + _poreParticles->_numParticles;
	}
public:
	void	init(REAL3& gravity, REAL dt);
public:
	void	addCloth(
		Mesh* mesh, REAL friction,
		REAL radius, REAL restDensity, REAL restFluidDensity, REAL restSolidFraction,
		REAL viscosity, REAL surfaceTension,
		float4 frontColor, float4 backColor, bool isSaved = true);
	void	addObstacle(
		Mesh* mesh, REAL mass, REAL friction,
		REAL3& pivot, REAL3& rotation,
		REAL radius, REAL viscosity, REAL surfaceTension,
		float4 frontColor, float4 backColor, bool isSaved = true);
	void	addSPHModel(
		REAL radius, REAL restDensity,
		REAL viscosity, REAL surfaceTension, float4 color);
	void	spawnSPHParticle(uchar spawnType);
public:
	void	getParticleNeighors(bool sorting);
	void	compDivergenceFree(REAL dt);
	void	compPressureForce(REAL dt);
public:
	void	compProjectiveDynamics(REAL dt);
public:
	void	update(void);
	void	simulation(void);
	void	reset(void);
public:
	void	draw(void);
	void	drawBoundary(void);
};

#endif