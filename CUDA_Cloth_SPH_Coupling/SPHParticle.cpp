#include "SPHParticle.h"

void SPHParticle::init(void) {
	_param = new SPHParticleParam();
	_type = TYPE_SPH_PARTICLE;
}
void SPHParticle::reset(void) {
	Particle::reset();
	d_impulses.clear();
	d_colWs.clear();

	d_ds.clear();
	d_ps.clear();
	d_as.clear();
	d_ks.clear();

	d_ns.clear();
	d_vPs.clear();
	d_ss.clear();
	d_relaxTs.clear();
	setParam(); 
}
void SPHParticle::resizeHost(void) {
	Particle::resizeHost();
	h_ss.resize(_numParticles);
	h_relaxTs.resize(_numParticles);
}
void SPHParticle::resizeDevice(void) {
	Particle::resizeDevice();
	d_impulses.resize(_numParticles * 3u);
	d_colWs.resize(_numParticles);
	d_ds.resize(_numParticles);
	d_ps.resize(_numParticles);
	d_as.resize(_numParticles);
	d_ks.resize(_numParticles);
	d_ns.resize(_numParticles * 3u);
	d_vPs.resize(_numParticles * 3u);
	d_ss = h_ss;
	d_relaxTs = h_relaxTs;
}

void SPHParticle::addParticle(REAL3 x, REAL3 v, uint phase) {
	Particle::addParticle(x, v, phase);
	h_ss.push_back(1.0);
	//h_ss.push_back(max((REAL)rand() / (REAL)RAND_MAX, 0.1));
	h_relaxTs.push_back(1.0);
}
void SPHParticle::addModel(
	REAL radius, REAL restDensity,
	REAL viscosity, REAL surfaceTension, float4 color)
{
	Particle::addModel(radius, viscosity, surfaceTension, color);
	REAL restVolume = radius * radius * radius * M_PI * 4.0 / 3.0;
	h_masses.push_back(restDensity * restVolume);
	h_restDensities.push_back(restDensity);
	h_restVolumes.push_back(restVolume);
	d_masses = h_masses;
	d_restDensities = h_restDensities;
	d_restVolumes = h_restVolumes;
}

void SPHParticle::copyToDevice(void) {
	Particle::copyToDevice();
	d_ss.copyFromHost(h_ss);
	d_relaxTs.copyFromHost(h_relaxTs);
}
void SPHParticle::copyToHost(void) {
	Particle::copyToHost();
	d_ss.copyToHost(h_ss);
	d_relaxTs.copyToHost(h_relaxTs);
}

void SPHParticle::draw(bool isSphere) {
#ifdef SPEED_COLOR
	REAL invVel = _sim->_h->_invh * 0.02;
#endif

	if (isSphere) {
		glEnable(GL_CULL_FACE);
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);

		for (uint i = 0u; i < _numParticles; i++) {
			if (h_colors[h_phases[i]].w == 0.0)
				continue;

			REAL3 p = make_REAL3(h_xs[i * 3u + 0u], h_xs[i * 3u + 1u], h_xs[i * 3u + 2u]);
#ifdef SPEED_COLOR
			float specular[4];
			float ambient[4];
			float diffuse[4];
			float c = sqrtf(min(
				(float)Length(make_REAL3(h_vs[i * 3u + 0u], h_vs[i * 3u + 1u], h_vs[i * 3u + 2u])) * invVel, 1.f));
			specular[0] = ambient[0] = diffuse[0] = c;
			specular[1] = ambient[1] = diffuse[1] = c;
			specular[2] = ambient[2] = diffuse[2] = 1.0f;
			specular[3] = ambient[3] = diffuse[3] = 1.0f;

			glMaterialfv(GL_FRONT, GL_SPECULAR, diffuse);
			glMaterialf(GL_FRONT, GL_SHININESS, 35);
			glMaterialfv(GL_FRONT, GL_AMBIENT, diffuse);
			glMaterialfv(GL_FRONT, GL_DIFFUSE, diffuse);
#else
			glMaterialfv(GL_FRONT, GL_SPECULAR, &h_colors[h_phases[i]].x);
			glMaterialf(GL_FRONT, GL_SHININESS, 35);
			glMaterialfv(GL_FRONT, GL_AMBIENT, &h_colors[h_phases[i]].x);
			glMaterialfv(GL_FRONT, GL_DIFFUSE, &h_colors[h_phases[i]].x);
#endif

			glPushMatrix();
			glTranslatef(p.x, p.y, p.z);
			glutSolidSphere(h_radii[h_phases[i]] * S3TO1(h_ss[i]), 5, 5);
			glPopMatrix();
		}
		glDisable(GL_CULL_FACE);
	}
	else {
		Particle::draw(false);
	}
}