#ifndef __SIMULATION_H__
#define __SIMULATION_H__

#pragma once
#include "Cloth.h"
#include "Obstacle.h"

static __inline__ __device__
void getIfromParam(const uint* is, uint i, uint* ino) {
	i *= 3u;
	ino[0] = is[i++];
	ino[1] = is[i++];
	ino[2] = is[i];
}
static __inline__ __device__
void getXfromParam(const REAL* X, uint* ino, REAL3* ps) {
	uint i = ino[0] * 3u;
	ps[0].x = X[i++]; ps[0].y = X[i++]; ps[0].z = X[i];
	i = ino[1] * 3u;
	ps[1].x = X[i++]; ps[1].y = X[i++]; ps[1].z = X[i];
	i = ino[2] * 3u;
	ps[2].x = X[i++]; ps[2].y = X[i++]; ps[2].z = X[i];
}

namespace Simulation {
	void initMasses(Cloth* cloth, Obstacle* obstacle);
	void compGravityForce(MeshObject* obj, const REAL3& gravity);
	void compRotationForce(Obstacle* obj, const REAL dt);

	void applyForce(MeshObject* obj, const REAL dt);

	void updateVelocity(Dvector<REAL>& n0s, Dvector<REAL>& n1s, Dvector<REAL>& vs, const REAL invdt);
	void updatePosition(Dvector<REAL>& ns, Dvector<REAL>& vs, const REAL dt);

	void initProject(Cloth* obj, const REAL dt, const REAL invdt2);
	void compErrorProject(Cloth* obj);
	void updateXsProject(
		Cloth* obj, const REAL invdt2,
		const REAL underRelax, const REAL omega, REAL* maxError);

	void Damping(Dvector<REAL>& vs, REAL w);
	void Damping(Dvector<REAL>& vs, Dvector<uchar>& isFixeds, REAL w);
}

#endif