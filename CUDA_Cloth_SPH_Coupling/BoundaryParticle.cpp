#include "BoundaryParticle.h"

void BoundaryParticle::init(void) {
	_param = new BoundaryParticleParam();
	_type = TYPE_BOUNDARY_PARTICLE;
}
void BoundaryParticle::reset(void) {
	Particle::reset();
	d_volumes.clear();
	d_ws.clear();
	d_inos.clear();

	d_shortEs = h_shortEs0;
	d_sampNums = h_sampNums0;
	setParam();
}
void BoundaryParticle::resizeDevice(void) {
	Particle::resizeDevice();
	d_volumes.resize(_numParticles);
	d_ws.resize(_numParticles << 1u);
	d_inos.resize(_numParticles);
}