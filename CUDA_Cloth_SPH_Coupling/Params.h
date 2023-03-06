#ifndef __PARAMATERS_H__
#define __PARAMATERS_H__

#pragma once
#include <fstream>
#include <string>
#include "../GL/freeglut.h"
#include "../include/CUDA_Custom/PrefixArray.h"

#define SCENE						3
#define QUALITY						1	// 0: Low, 1: Medium, 2: High

#define SMOOTHED_RENDERING			1

#define TYPE_MESH_OBSTACLE			1
#define TYPE_MESH_CLOTH				2
#define TYPE_SPH_PARTICLE			3
#define TYPE_PORE_PARTICLE			4
#define TYPE_BOUNDARY_PARTICLE		5

//----------------------------------------------
#if QUALITY==0
#define COLLISION_SMOOTHING		5
#elif QUALITY==1
#define COLLISION_SMOOTHING		10
#else
#define COLLISION_SMOOTHING		15
#endif
//----------------------------------------------
#define LAPLACIAN_SMOOTHING		0 // 0: uniform, 1: taubin
#if LAPLACIAN_SMOOTHING == 1
#define LAPLACIAN_LAMBDA	0.65
#define LAPLACIAN_MU		-0.75
#else
#define LAPLACIAN_LAMBDA	0.25
#endif
//----------------------------------------------

#define COL_CCD_THICKNESS			1.0e-6
#define COL_CLEARANCE_RATIO			2.0
#define SPH_RADIUS_RATIO			4.0

#define MIN_VOLUME					0.05

#define CONST_W						1.5666814710608447114749495456982 // 315.0 / (64.0 * M_PI)
#define CONST_G						-14.323944878270580219199538703526 // -45.0 / M_PI
#define CONST_CUBICW				2.5464790894703253723021402139602 // 8.0 / M_PI
#define CONST_CUBICG				15.278874536821952233812841283761 // 48.0 / M_PI
#define CONST_LAPLACIAN				14.323944878270580219199538703526 // 45.0 / M_PI
#define CONST_COHESION				10.185916357881301489208560855841 // 32.0 / (M_PI);
#define CONST_ADHESION				0.007

#define MAX_NEIGHBORS				512

#define S3TO1(X)	pow(X, 0.33333333333333333333333333333333333333333333333333333333333333334)

struct ObjParam {
	REAL	*_impulses;
	REAL	*_colWs;
	REAL	*_thicknesses;
	REAL	*_frictions;

	uint	*_fs;
	REAL	*_ns;
	REAL	*_vs;
	REAL	*_invMs;
	REAL	*_ms;
	uint	*_nodePhases;
	uchar	*_isFixeds;
	REAL	*_forces;

	uint	_numNodes;
	uint	_numFaces;
	uchar	_type;
};
struct ClothParam : public ObjParam {
	REAL	*_restSolidFractions;
	REAL	*_maxFluidMass;
	REAL	*_mfs;
	REAL	*_ss;
};

struct ParticleParam {
	REAL	*_radii;
	REAL	*_viscosities;
	REAL	*_surfaceTensions;

	REAL	*_xs;
	REAL	*_vs;
	REAL	*_forces;
	uint	*_phases;

	uint2	*_neis;
	uint	*_ineis;

	uint	_numParticles;
	uchar	_type;
};
struct BoundaryParticleParam : public ParticleParam {
	REAL	*_ws;
	REAL	*_volumes;
	uint	*_inos;
};
struct SPHParticleParam : public ParticleParam {
	REAL	*_impulses;
	REAL	*_colWs;

	REAL	*_masses;
	REAL	*_restVolumes;
	REAL	*_restDensities;

	REAL	*_ds;
	REAL	*_ps;
	REAL	*_as;
	REAL	*_ks;

	REAL	*_ns;
	REAL	*_vPs;
	REAL	*_ss;
	REAL	*_relaxTs;
};
struct PoreParticleParam : public BoundaryParticleParam {
	REAL	*_restVolumes;
	REAL	*_restDensities;
	REAL	*_restFluidDensities;
	REAL	*_restSolidFractions;

	REAL	*_nodeWeights;
	REAL	*_mws;

	REAL	*_mfs;
	REAL	*_ss;
	REAL	*_vPs;
	uchar	*_isDrippings;
	REAL	*_dms;

	REAL	*_norms;

	REAL	*_Ps;
	REAL	*_Cs;
	REAL	*_Ks;
};

struct SpatialHashParam {
	uint2	*_ids;
	uint	*_keys;
	uint	*_istarts;
	uint	*_iends;
	uint3	_size;
	REAL	_radius;
	uint	_numHash;
	uint	_numParticles;
};

#endif