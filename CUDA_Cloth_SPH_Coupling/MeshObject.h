#ifndef __OBJECT_H__
#define __OBJECT_H__

#pragma once
//#include "PrimalTree.h"
#include "BVH.h"

class MeshObject
{
public:
	BVH						*_bvh;
	RTriangle				*_RTri;
public:
	Dvector<uint>			d_fs;
	Dvector<REAL>			d_ns;
	Dvector<REAL>			d_n0s;
	Dvector<REAL>			d_vs;
	Dvector<REAL>			d_ms;
	Dvector<REAL>			d_invMs;
	Dvector<uchar>			d_isFixeds;
	DPrefixArray<uint>		d_ses;
	DPrefixArray<uint>		d_bes;
	DPrefixArray<uint>		d_nbFs;
	DPrefixArray<uint>		d_nbNs;
	Dvector<REAL>			d_fNorms;
	Dvector<REAL>			d_nNorms;
	Dvector<uint>			d_nodePhases;
public:
	vector<uint>			h_fs;
	vector<REAL>			h_ns;
	vector<REAL>			h_ms;
	vector<uchar>			h_isFixeds;
	PrefixArray<uint>		h_ses;
	PrefixArray<uint>		h_bes;
	PrefixArray<uint>		h_nbFs;
	PrefixArray<uint>		h_nbNs;
	vector<REAL>			h_fNorms;
	vector<REAL>			h_nNorms;
	vector<uint>			h_nodePhases;
public:
	vector<uint>			h_fs0;
	vector<REAL>			h_ns0;
	vector<REAL>			h_ms0;
	vector<uchar>			h_isFixeds0;
	PrefixArray<uint>		h_ses0;
	PrefixArray<uint>		h_bes0;
	PrefixArray<uint>		h_nbFs0;
	PrefixArray<uint>		h_nbNs0;
	vector<uint>			h_nodePhases0;
public:
	Dvector<REAL>			d_impulses;
	Dvector<REAL>			d_colWs;
	Dvector<REAL>			d_thicknesses;
	Dvector<REAL>			d_frictions;
public:
	vector<REAL>			h_thicknesses;
	vector<REAL>			h_frictions;
	vector<float4>			h_frontColors;
	vector<float4>			h_backColors;
public:
	Dvector<REAL>			d_forces;
public:
	ObjParam				*_param;
public:
	uint					_numFaces;
	uint					_numNodes;
public:
	StreamParam				*_streams;
public:
	uchar					_type;
public:
	MeshObject() { }
	virtual ~MeshObject() {}
public:
	inline virtual void setParam(void) {
		_param->_impulses = d_impulses._list;
		_param->_colWs = d_colWs._list;
		_param->_thicknesses = d_thicknesses._list;
		_param->_frictions = d_frictions._list;

		_param->_fs = d_fs._list;
		_param->_ns = d_ns._list;
		_param->_vs = d_vs._list;
		_param->_ms = d_ms._list;
		_param->_invMs = d_invMs._list;
		_param->_isFixeds = d_isFixeds._list;
		_param->_nodePhases = d_nodePhases._list;
		_param->_forces = d_forces._list;
		_param->_numFaces = _numFaces;
		_param->_numNodes = _numNodes;
		_param->_type = _type;
	}
public:
	virtual void	init(void);
	virtual void	reset(void) = 0;
public:
	void	addMesh(
		Mesh* mesh, REAL mass, REAL thickness, REAL friction, float4 frontColor, float4 backColor);
	void	initMasses(REAL mass);
	void	initVelocities(void);
	void	initNoramls(void);
	void	initBVH(void);
public:
	void	computeNormal(void);
public:
	void	draw(void);
	void	drawWire(void);
	virtual void	drawSurface(void);
public:
	void	copyToDevice(void);
	virtual void	copyToHost(void);
	void	copyNbToDevice(void);
	void	copyNbToHost(void);
	void	copyNormToDevice(void);
	void	copyNormToHost(void);
};
#endif