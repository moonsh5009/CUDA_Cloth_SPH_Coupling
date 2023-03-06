#ifndef __PORE_PARTICLE_H__
#define __PORE_PARTICLE_H__

#pragma once
#include "BoundaryParticle.h"

class PoreParticle :public BoundaryParticle
{
public:
	Dvector<REAL>			d_nodeWeights;
	Dvector<REAL>			d_mws;
public:
	Dvector<REAL>			d_mfs;
	Dvector<REAL>			d_ss;
	Dvector<REAL>			d_vPs;
	Dvector<uchar>			d_isDrippings;
	Dvector<REAL>			d_dms;
	Dvector<REAL>			d_norms;
public:
	Dvector<REAL>			d_Ps;
	Dvector<REAL>			d_Cs;
	Dvector<REAL>			d_Ks;
public:
	Dvector<REAL>			d_restVolumes;
	Dvector<REAL>			d_restDensities;
	Dvector<REAL>			d_restFluidDensities;
	Dvector<REAL>			d_restSolidFractions;
public:
	vector<REAL>			h_restVolumes;
	vector<REAL>			h_restDensities;
	vector<REAL>			h_restFluidDensities;
	vector<REAL>			h_restSolidFractions;
public:
	PoreParticle() {
		init();
	}
	virtual ~PoreParticle() {}
public:
	virtual void setParam(void) {
		BoundaryParticle::setParam();
		((PoreParticleParam*)_param)->_restVolumes = d_restVolumes._list;
		((PoreParticleParam*)_param)->_restDensities = d_restDensities._list;
		((PoreParticleParam*)_param)->_restFluidDensities = d_restFluidDensities._list;
		((PoreParticleParam*)_param)->_restSolidFractions = d_restSolidFractions._list;

		((PoreParticleParam*)_param)->_nodeWeights = d_nodeWeights._list;
		((PoreParticleParam*)_param)->_mws = d_mws._list;

		((PoreParticleParam*)_param)->_mfs = d_mfs._list;
		((PoreParticleParam*)_param)->_ss = d_ss._list;
		((PoreParticleParam*)_param)->_vPs = d_vPs._list;
		((PoreParticleParam*)_param)->_isDrippings = d_isDrippings._list;
		((PoreParticleParam*)_param)->_dms = d_dms._list;
		((PoreParticleParam*)_param)->_norms = d_norms._list;

		((PoreParticleParam*)_param)->_Ps = d_Ps._list;
		((PoreParticleParam*)_param)->_Cs = d_Cs._list;
		((PoreParticleParam*)_param)->_Ks = d_Ks._list;
	}
public:
	virtual void	init(void);
	virtual void	reset(void);
	virtual void	resizeDevice(void);
public:
	virtual void	addModel(
		REAL radius, REAL restDensity, REAL restFluidDensity, REAL restSolidFraction,
		REAL viscosity, REAL surfaceTension, float4 color);
public:
	void	drawPoreVelocity(void);
};

#endif