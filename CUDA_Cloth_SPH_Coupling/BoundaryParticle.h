#ifndef __BOUNDARY_PARTICLE_H__
#define __BOUNDARY_PARTICLE_H__

#pragma once
#include "Particle.h"

class BoundaryParticle :public Particle
{
public:
	Dvector<REAL>			d_volumes;
	Dvector<REAL>			d_ws;
	Dvector<uint>			d_inos;
public:
	Dvector<uint>			d_shortEs;
	Dvector<uint>			d_sampNums;
	vector<uint>			h_shortEs0;
	vector<uint>			h_sampNums0;
public:
	BoundaryParticle() {
		init();
	}
	virtual ~BoundaryParticle() {}
public:
	virtual void setParam(void) {
		Particle::setParam();
		((BoundaryParticleParam*)_param)->_volumes = d_volumes._list;
		((BoundaryParticleParam*)_param)->_ws = d_ws._list;
		((BoundaryParticleParam*)_param)->_inos = d_inos._list;
	}
public:
	virtual void	init(void);
	virtual void	reset(void);
	virtual void	resizeDevice(void);
};

#endif