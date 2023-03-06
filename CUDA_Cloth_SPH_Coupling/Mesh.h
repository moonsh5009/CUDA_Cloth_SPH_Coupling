#ifndef __MESH_H__
#define __MESH_H__

#pragma once
#include "Params.h"

class Mesh
{
public:
	vector<uint>		_fs;
	vector<REAL>		_ns;
	PrefixArray<uint>	_ses;
	PrefixArray<uint>	_bes;
	PrefixArray<uint>	_nbFs;
	PrefixArray<uint>	_nbNs;
public:
	vector<REAL>		_fnorms;
	vector<REAL>		_vnorms;
public:
	AABB				_aabb;
	uint				_numFaces;
	uint				_numVertices;
public:
	Mesh() { }
	Mesh(const char* filename, REAL3 center, REAL3 scale) {
		loadObj(filename, center, scale);
	}
	Mesh(const char* filename, REAL3 center, REAL scale = (REAL)1.0) {
		loadObj(filename, center, scale);
	}
	Mesh(const char* filename) {
		loadObj(filename);
	}
	~Mesh() {}
public:
	void	loadObj(const char* filename, REAL3 center, REAL3 scale);
	void	loadObj(const char* filename, REAL3 center, REAL scale);
	void	loadObj(const char* filename);
	void	moveCenter(REAL3 center, REAL3 scale);
	void	moveCenter(REAL3 center, REAL scale);
	void	buildAdjacency(void);
	void	computeNormal(void);
	void	rotate(REAL3 degree);
};

#endif