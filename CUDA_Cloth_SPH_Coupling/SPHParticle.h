#ifndef __SPH_PARTICLE_H__
#define __SPH_PARTICLE_H__

#pragma once
#include "Particle.h"

class SPHParticle :public Particle
{
public:
	Dvector<REAL>			d_impulses;
	Dvector<REAL>			d_colWs;
public:
	Dvector<REAL>			d_ds;
	Dvector<REAL>			d_ps;
	Dvector<REAL>			d_as;
	Dvector<REAL>			d_ks;
public:
	Dvector<REAL>			d_ns;
	Dvector<REAL>			d_vPs;
	Dvector<REAL>			d_ss;
	vector<REAL>			h_ss;
public:
	Dvector<REAL>			d_relaxTs;
	vector<REAL>			h_relaxTs;
public:
	Dvector<REAL>			d_masses;
	Dvector<REAL>			d_restVolumes;
	Dvector<REAL>			d_restDensities;
public:
	vector<REAL>			h_masses;
	vector<REAL>			h_restVolumes;
	vector<REAL>			h_restDensities;
public:
	SPHParticle() {
		init();
	}
	virtual ~SPHParticle() {}
public:
	virtual void setParam(void) {
		Particle::setParam();
		((SPHParticleParam*)_param)->_impulses = d_impulses._list;
		((SPHParticleParam*)_param)->_colWs = d_colWs._list;

		((SPHParticleParam*)_param)->_masses = d_masses._list;
		((SPHParticleParam*)_param)->_restVolumes = d_restVolumes._list;
		((SPHParticleParam*)_param)->_restDensities = d_restDensities._list;

		((SPHParticleParam*)_param)->_ds = d_ds._list;
		((SPHParticleParam*)_param)->_ps = d_ps._list;
		((SPHParticleParam*)_param)->_as = d_as._list;
		((SPHParticleParam*)_param)->_ks = d_ks._list;

		((SPHParticleParam*)_param)->_ns = d_ns._list;
		((SPHParticleParam*)_param)->_vPs = d_vPs._list;
		((SPHParticleParam*)_param)->_ss = d_ss._list;
		((SPHParticleParam*)_param)->_relaxTs = d_relaxTs._list;
	}
public:
	virtual void	init(void);
	virtual void	reset(void);
	virtual void	resizeHost(void);
	virtual void	resizeDevice(void);
public:
	virtual void	addParticle(REAL3 x, REAL3 v, uint phase);
	virtual void	addModel(
		REAL radius, REAL restDensity, 
		REAL viscosity, REAL surfaceTension, float4 color);
public:
	virtual void	copyToDevice(void);
	virtual void	copyToHost(void);
public:
	virtual void	draw(bool isSphere = false);
};

#endif