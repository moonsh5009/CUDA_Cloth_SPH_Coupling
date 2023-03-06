#ifndef __PARTICLE_SAMPLING_H__
#define __PARTICLE_SAMPLING_H__

#pragma once
#include "Cloth.h"
#include "Obstacle.h"
#include "PoreParticle.h"
#include "SPHParticle.h"

namespace ParticleSampling {
	void compParticleNum(
		MeshObject* obj, BoundaryParticle* boundaryParticles,
		Dvector<uint>& prevInds, Dvector<uint>& currInds, Dvector<bool>& isGenerateds, bool& isApplied);
	void particleSampling(MeshObject* obj, BoundaryParticle* boundaryParticles);

	void compNodeWeights(Cloth* cloth, PoreParticle* poreParticles);
	void lerpPosition(MeshObject* obj, BoundaryParticle* boundaryParticles);
	void lerpVelocity(MeshObject* obj, BoundaryParticle* boundaryParticles);
	void lerpForce(Cloth* cloth, PoreParticle* poreParticles);
}

#endif