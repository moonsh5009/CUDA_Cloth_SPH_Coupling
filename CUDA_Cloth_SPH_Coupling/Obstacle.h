#ifndef __OBSTACLE_H__
#define __OBSTACLE_H__

#pragma once
#include "MeshObject.h"
#include "PrimalTree.h"

class Obstacle : public MeshObject
{
public:
	PrimalTree			*_priTree;
public:
	vector<REAL3>		h_pivots;
	vector<REAL3>		h_degrees;
public:
	Dvector<REAL3>		d_pivots;
	Dvector<REAL3>		d_degrees;
public:
	Obstacle() {
		_type = TYPE_MESH_OBSTACLE;
		init();
	}
	virtual ~Obstacle() {}
public:
	virtual void	init(void);
	virtual void	reset(void);
public:
	void	addObject(
		Mesh* mesh, REAL mass, REAL thickness, REAL friction,
		float4 frontColor, float4 backColor,
		REAL3& pivot, REAL3& rotation, bool isSaved = true);
public:
};
#endif