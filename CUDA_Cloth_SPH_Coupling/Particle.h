#ifndef __PARTICLE_H__
#define __PARTICLE_H__

#pragma once
#include "Params.h"

class Particle
{
public:
	Dvector<REAL>			d_xs;
	Dvector<REAL>			d_x0s;
	Dvector<REAL>			d_vs;
	Dvector<uint>			d_phases;
	Dvector<REAL>			d_forces;
public:
	vector<REAL>			h_xs;
	vector<REAL>			h_vs;
	vector<uint>			h_phases;
public:
	Dvector<REAL>			d_radii;
	Dvector<REAL>			d_viscosities;
	Dvector<REAL>			d_surfaceTensions;
public:
	vector<REAL>			h_radii;
	vector<REAL>			h_viscosities;
	vector<REAL>			h_surfaceTensions;
	vector<float4>			h_colors;
public:
	Dvector<uint2>			d_neis;
	Dvector<uint>			d_ineis;
public:
	ParticleParam			*_param;
public:
	uchar					_type;
	uint					_numParticles;
public:
	Particle() {
		init();
	}
	virtual ~Particle() {}
public:
	virtual void setParam(void) {
		_param->_radii = d_radii._list;
		_param->_viscosities = d_viscosities._list;
		_param->_surfaceTensions = d_surfaceTensions._list;

		_param->_ineis = d_ineis._list;

		_param->_xs = d_xs._list;
		_param->_vs = d_vs._list;
		_param->_phases = d_phases._list;
		_param->_forces = d_forces._list;
		_param->_numParticles = _numParticles;
		_param->_type = _type;
	}
public:
	virtual void	init(void);
	virtual void	reset(void);
	virtual void	resize(uint size);
	virtual void	resizeHost(void);
	virtual void	resizeDevice(void);
	virtual void	updateParticle(void);
public:
	virtual void	addParticle(REAL3 x, REAL3 v, uint phase);
	virtual void	addModel(REAL radius, REAL viscosity, REAL surfaceTension, float4 color);
public:
	virtual void	draw(bool isSphere = false);
public:
	virtual void	copyToDevice(void);
	virtual void	copyToHost(void);
};
#endif