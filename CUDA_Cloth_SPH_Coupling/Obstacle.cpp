#include "Obstacle.h"

void Obstacle::init(void) {
	MeshObject::init();
	_priTree = new PrimalTree();

	_param = new ObjParam();
	_type = TYPE_MESH_OBSTACLE;
}
void Obstacle::reset(void) {
	h_fs = h_fs0;
	h_ns = h_ns0;
	h_ses = h_ses0;
	h_bes = h_bes0;
	h_nbFs = h_nbFs0;
	h_nbNs = h_nbFs0;
	h_ms = h_ms0;
	h_nodePhases = h_nodePhases0;

	_numFaces = h_fs.size() / 3u;
	_numNodes = h_ns.size() / 3u;
	d_ms.resize(_numNodes);
	d_invMs.resize(_numNodes);

	initVelocities();
	initNoramls();

	copyToDevice();
	copyNbToDevice();

	computeNormal();

	initBVH();

	setParam();
}
void Obstacle::addObject(
	Mesh* mesh, REAL mass, REAL thickness, REAL friction,
	float4 frontColor, float4 backColor,
	REAL3& pivot, REAL3& rotation, bool isSaved) 
{
	addMesh(mesh, mass, thickness, friction, frontColor, backColor);

	h_pivots.push_back(pivot);
	h_degrees.push_back(rotation);
	d_pivots = h_pivots;
	d_degrees = h_degrees;

	if (isSaved) {
		h_fs0 = h_fs;
		h_ns0 = h_ns;
		h_ses0 = h_ses;
		h_bes0 = h_bes;
		h_nbFs0 = h_nbFs;
		h_nbNs0 = h_nbFs;
		h_ms0 = h_ms;
		h_isFixeds0 = h_isFixeds;
		h_nodePhases0 = h_nodePhases;
	}
	initBVH();

	ObjParam p;
	p._fs = &h_fs[0];
	p._ns = &h_ns[0];
	p._numFaces = _numFaces;
	p._numNodes = _numNodes;
	_priTree->buildTree(p, *_param, h_nbFs, h_fNorms, h_nNorms, 0.1, 7u);
}