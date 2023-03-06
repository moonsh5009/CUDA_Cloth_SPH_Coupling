#ifndef __SPH_SOLVER_H__
#define __SPH_SOLVER_H__

#pragma once
#include "PoreParticle.h"
#include "SPHParticle.h"
#include "SpatialHashing.h"

namespace SPHSolver {
	void compVolume(
		PoreParticle* poreParticles, BoundaryParticle* boundaryParticles);
	void compDensity(
		SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles);
	void compDFSPHFactor(
		SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles);

	void compCDStiffness(
		SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles, 
		REAL dt, REAL* d_sumError);
	void compDFStiffness(
		SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles, 
		REAL dt, REAL* d_sumError);
	void applyDFSPH(
		SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles);
	void applyPressureForce(
		SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles);

	void compGravityForce(SPHParticle* sphParticles, const REAL3& gravity);
	void compViscosityForce(SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles);
	void compSurfaceTensionForce(SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles);
	void applyForce(SPHParticle* sphParticles, const REAL dt);
}

#endif