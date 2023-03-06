#include "Particle.h"

void Particle::init(void) {
	_param = new ParticleParam();
	_numParticles = 0u;
}
void Particle::reset(void) {
	d_ineis.clear();
	d_neis.clear();

	d_xs.clear();
	d_x0s.clear();
	d_vs.clear();
	d_phases.clear();
	d_forces.clear();
	h_xs.clear();
	h_vs.clear();
	h_phases.clear();
	_numParticles = 0u;
}
void Particle::resize(uint size) {
	_numParticles = size;
	resizeHost();
	resizeDevice();
}
void Particle::resizeHost(void) {
	h_xs.resize(_numParticles * 3u);
	h_vs.resize(_numParticles * 3u);
	h_phases.resize(_numParticles);
}
void Particle::resizeDevice(void) {
	d_ineis.resize(_numParticles + 1u);
	d_xs = h_xs;
	d_x0s.resize(_numParticles * 3u);
	d_vs = h_vs;
	d_phases = h_phases;
	d_forces.resize(_numParticles * 3u);
}
void Particle::updateParticle(void) {
	resize(h_xs.size() / 3u);
	setParam();
}
void Particle::addParticle(REAL3 x, REAL3 v, uint phase) {
	h_xs.push_back(x.x);
	h_xs.push_back(x.y);
	h_xs.push_back(x.z);
	h_vs.push_back(v.x);
	h_vs.push_back(v.y);
	h_vs.push_back(v.z);
	h_phases.push_back(phase);
}
void Particle::addModel(REAL radius, REAL viscosity, REAL surfaceTension, float4 color) {
	h_colors.push_back(color);
	h_radii.push_back(radius);
	h_viscosities.push_back(viscosity);
	h_surfaceTensions.push_back(surfaceTension);
	d_radii = h_radii;
	d_viscosities = h_viscosities;
	d_surfaceTensions = h_surfaceTensions;
}
#define SPEED_COLOR
void Particle::draw(bool isSphere) {

	if (isSphere) {
		glEnable(GL_CULL_FACE);
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);

		for (uint i = 0u; i < _numParticles; i++) {
			//if (h_colors[h_phases[i]].w == 0.0)
			//	continue;

			REAL3 p = make_REAL3(h_xs[i * 3u + 0u], h_xs[i * 3u + 1u], h_xs[i * 3u + 2u]);
			//if (fabs(p.x) > 0.3 || fabs(p.y) > 0.3)
			//	continue;

			glMaterialfv(GL_FRONT, GL_SPECULAR, &h_colors[h_phases[i]].x);
			glMaterialf(GL_FRONT, GL_SHININESS, 35);
			glMaterialfv(GL_FRONT, GL_AMBIENT, &h_colors[h_phases[i]].x);
			glMaterialfv(GL_FRONT, GL_DIFFUSE, &h_colors[h_phases[i]].x);

			glPushMatrix();
			glTranslatef(p.x, p.y, p.z);
			glutSolidSphere(h_radii[h_phases[i]], 5, 5);
			glPopMatrix();
		}
		glDisable(GL_CULL_FACE);
	}
	else {
		glDisable(GL_LIGHTING);
		glPointSize(2.5f);
		glBegin(GL_POINTS);

		for (uint i = 0u; i < _numParticles; i++) {
			if (h_colors[h_phases[i]].w == 0.0)
				continue;
			REAL3 p = make_REAL3(h_xs[i * 3u + 0u], h_xs[i * 3u + 1u], h_xs[i * 3u + 2u]);
#ifdef SPEED_COLOR
			uint phase = h_phases[i];
			REAL h = h_radii[phase] * SPH_RADIUS_RATIO;
			REAL invVel = 1.0 / h * 0.005;
			float c = min(sqrtf(min(
				(float)Length(make_REAL3(h_vs[i * 3u + 0u], h_vs[i * 3u + 1u], h_vs[i * 3u + 2u])) * invVel, 1.f)), 0.9f);
			glColor3f(c, c, 1.f);

			/*float c = (float)i / (float)(_numParticles + 1);
			glColor3f(c, c, 1.f);*/
#else
			glColor3fv(&h_colors[h_phases[i]].x);
#endif
			glVertex3f(p.x, p.y, p.z);
		}
		glEnd();
		glEnable(GL_LIGHTING);
	}
}

void Particle::copyToDevice(void) {
	d_xs.copyFromHost(h_xs);
	d_vs.copyFromHost(h_vs);
}
void Particle::copyToHost(void) {
	d_xs.copyToHost(h_xs);
	d_vs.copyToHost(h_vs);
}