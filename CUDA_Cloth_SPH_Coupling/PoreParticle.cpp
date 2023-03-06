#include "PoreParticle.h"

void PoreParticle::init(void) {
	_param = new PoreParticleParam();
	_type = TYPE_PORE_PARTICLE;
}
void PoreParticle::reset(void) {
	BoundaryParticle::reset();
	d_nodeWeights.clear();
	d_mws.clear();

	d_mfs.clear();
	d_ss.clear();
	d_vPs.clear();
	d_isDrippings.clear();
	d_dms.clear();
	d_norms.clear();

	d_Ps.clear();
	d_Cs.clear();
	d_Ks.clear();
	setParam();
}
void PoreParticle::resizeDevice(void) {
	BoundaryParticle::resizeDevice();
	d_mws.resize(_numParticles << 1u);

	d_mfs.resize(_numParticles);
	d_ss.resize(_numParticles);
	d_vPs.resize(_numParticles * 3u);
	d_isDrippings.resize(_numParticles);
	d_dms.resize(_numParticles);
	d_norms.resize(_numParticles * 3u);

	d_Ps.resize(_numParticles * 3u);
	d_Cs.resize(_numParticles * 3u);
	d_Ks.resize(_numParticles * 3u);
}

void PoreParticle::addModel(
	REAL radius, REAL restDensity, REAL restFluidDensity, REAL restSolidFraction,
	REAL viscosity, REAL surfaceTension, float4 color)
{
	Particle::addModel(radius, viscosity, surfaceTension, color);
	h_restDensities.push_back(restDensity);
	h_restFluidDensities.push_back(restFluidDensity);
	h_restSolidFractions.push_back(restSolidFraction);
	h_restVolumes.push_back(radius * radius * radius * M_PI * 4.0 / 3.0);
	d_restDensities = h_restDensities;
	d_restFluidDensities = h_restFluidDensities;
	d_restSolidFractions = h_restSolidFractions;
	d_restVolumes = h_restVolumes;
}

void PoreParticle::drawPoreVelocity(void) {
	vector<REAL> h_vPs;
	d_vPs.copyToHost(h_vPs);

	glDisable(GL_LIGHTING);
	glPointSize(2.5f);
	glLineWidth(2.5f);

	for (uint i = 0u; i < _numParticles; i++) {
		REAL3 p = make_REAL3(h_xs[i * 3u + 0u], h_xs[i * 3u + 1u], h_xs[i * 3u + 2u]);
		REAL3 vP = make_REAL3(h_vPs[i * 3u + 0u], h_vPs[i * 3u + 1u], h_vPs[i * 3u + 2u]);
		vP *= 0.05;

		glColor3f(0.6f, 0.0f, 0.0f);
		glBegin(GL_POINTS);
		glVertex3f(p.x, p.y, p.z);
		glEnd();

		glColor3f(0.0f, 0.0f, 0.6f);
		glBegin(GL_LINES);
		glVertex3f(p.x, p.y, p.z);
		glVertex3f(p.x + vP.x, p.y + vP.y, p.z + vP.z);
		glEnd();
	}

	glEnable(GL_LIGHTING);
}