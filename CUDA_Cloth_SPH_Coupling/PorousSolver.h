#ifndef __POROUS_SOLVER_H__
#define __POROUS_SOLVER_H__

#pragma once
#include "Cloth.h"
#include "Obstacle.h"
#include "PoreParticle.h"
#include "SPHParticle.h"

#define ABSORPTION_MAX			1.18
#define ABSORPTION_K			0.01
#define ABSORPTION_N			0.18

#define EMISSION_N				(-0.85)
#define DRIPPING_N				(-0.85)

#define DIFFUSION_K				0.1
#define DIFFUSION_N				0.01

namespace PorousSolver {
	void lerpPoreFactorToParticle(Cloth* cloth, PoreParticle* poreParticles);
	void lerpPoreFactorToObject(Cloth* cloth, PoreParticle* poreParticles);

	void initPoreFactor(
		PoreParticle* poreParticles);
	void updateRelaxT(
		SPHParticle* sphParticles, REAL dt);
	void compPoreVelocity(
		SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles,
		const REAL3& gravity);
	void compAbsorption(
		Cloth* cloths, SPHParticle* sphParticles, PoreParticle* poreParticles, const REAL dt);
	void compEmission(
		Cloth* cloths, SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles);
	void deleteAbsorbedParticle(SPHParticle* sphParticles);
	void generateDrippingParticle(SPHParticle* sphParticles, PoreParticle* poreParticles);
	void compDiffusion(Cloth* cloths, PoreParticle* poreParticles, REAL dt);

	void compPorePressureForce(SPHParticle* sphParticles, PoreParticle* poreParticles);
	void compDragForce(SPHParticle* sphParticles, PoreParticle* poreParticles);
	void compPoreAttractionForce(SPHParticle* sphParticles, PoreParticle* poreParticles);
	void compPoreAdhesionForce(SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles);
}

#endif