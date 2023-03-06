#ifndef __CLOTH_H__
#define __CLOTH_H__

#pragma once
#include "MeshObject.h"

class Cloth : public MeshObject
{
public:
	Dvector<REAL>	d_mfs;
	Dvector<REAL>	d_ss;
	Dvector<REAL>	d_restSolidFractions;
	Dvector<REAL>	d_maxFluidMasses;
	vector<REAL>	h_mfs;
	vector<REAL>	h_ss;
	vector<REAL>	h_restSolidFractions;
	vector<REAL>	h_maxFluidMasses;
public:
	Dvector<REAL>	d_stretchRLs;
	Dvector<REAL>	d_bendingRLs;
	Dvector<REAL>	d_stretchWs;
	Dvector<REAL>	d_bendingWs;
public:
	vector<REAL>	h_stretchRLs;
	vector<REAL>	h_bendingRLs;
	vector<REAL>	h_stretchWs;
	vector<REAL>	h_bendingWs;
public:
	Dvector<REAL>	d_Bs;
	Dvector<REAL>	d_Zs;
	Dvector<REAL>	d_Xs;
	Dvector<REAL>	d_newXs;
	Dvector<REAL>	d_prevXs;
public:
	Dvector<REAL>	d_edgeLimits;
	vector<REAL>	h_edgeLimits;
public:
	Cloth() {
		_type = TYPE_MESH_OBSTACLE;
		init();
	}
	virtual ~Cloth() {}
public:
	inline virtual void setParam(void) {
		MeshObject::setParam();
		((ClothParam*)_param)->_mfs = d_mfs._list;
		((ClothParam*)_param)->_ss = d_ss._list;
		((ClothParam*)_param)->_restSolidFractions = d_restSolidFractions._list;
		((ClothParam*)_param)->_maxFluidMass = d_maxFluidMasses._list;
	}
public:
	virtual void	init(void);
	virtual void	reset(void);
public:
	void	addCloth(
		Mesh* mesh, REAL mass, REAL thickness, REAL friction,
		REAL restSolidFraction, REAL maxFluidMass,
		float4 frontColor, float4 backColor, bool isSaved = true);
	void	initConstraints(void);
public:
	void	fix(void);
	void	moveFixed(REAL3 vel);
	void	rotateFixed(REAL3 degreeL, REAL3 degreeR, REAL moveL, REAL moveR, REAL invdt);
public:
	virtual void	drawSurface(void);
	virtual void	copyToHost(void);
};
#endif