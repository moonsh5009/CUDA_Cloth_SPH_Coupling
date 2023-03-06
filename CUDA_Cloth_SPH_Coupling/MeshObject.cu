#include "MeshObject.cuh"

void MeshObject::init(void) {
	_streams = new StreamParam();
	_streams->initStream(11u);

	_bvh = new BVH();
	_RTri = new RTriangle();

	_numFaces = _numNodes = 0u;

	h_ses._index.resize(1, 0);
	h_bes._index.resize(1, 0);
	h_nbFs._index.resize(1, 0);
	h_nbNs._index.resize(1, 0);

	h_ses0 = h_ses;
	h_bes0 = h_bes;
	h_nbFs0 = h_nbFs;
	h_nbNs0 = h_nbNs;
}
void MeshObject::addMesh(
	Mesh* mesh, REAL mass, REAL thickness, REAL friction, float4 frontColor, float4 backColor) 
{
	uint newPhase = h_frontColors.size();
	h_frontColors.push_back(frontColor);
	h_backColors.push_back(backColor);
	h_frictions.push_back(friction);
	d_frictions = h_frictions;
	h_thicknesses.push_back(thickness);
	d_thicknesses = h_thicknesses;
	
	h_fs.insert(h_fs.end(), mesh->_fs.begin(), mesh->_fs.end());
	h_ns.insert(h_ns.end(), mesh->_ns.begin(), mesh->_ns.end());

	for (uint i = _numFaces * 3u; i < h_fs.size(); i++)
		h_fs[i] += _numNodes;

	PrefixArray<uint> buffer = mesh->_ses;
	for (uint i = 0u; i < buffer._array.size(); i++)
		buffer._array[i] += _numNodes;
	for (uint i = 0u; i < buffer._index.size(); i++)
		buffer._index[i] += h_ses.arraySize() >> 1u;
	h_ses._array.insert(h_ses._array.end(), buffer._array.begin(), buffer._array.end());
	h_ses._index.insert(h_ses._index.end(), buffer._index.begin() + 1, buffer._index.end());

	buffer = mesh->_bes;
	for (uint i = 0u; i < buffer._array.size(); i++)
		buffer._array[i] += _numNodes;
	for (uint i = 0u; i < buffer._index.size(); i++)
		buffer._index[i] += h_bes.arraySize() >> 1u;
	h_bes._array.insert(h_bes._array.end(), buffer._array.begin(), buffer._array.end());
	h_bes._index.insert(h_bes._index.end(), buffer._index.begin() + 1, buffer._index.end());

	buffer = mesh->_nbFs;
	for (uint i = 0u; i < buffer._array.size(); i++)
		buffer._array[i] += _numFaces;
	for (uint i = 0u; i < buffer._index.size(); i++)
		buffer._index[i] += h_nbFs.arraySize();
	h_nbFs._array.insert(h_nbFs._array.end(), buffer._array.begin(), buffer._array.end());
	h_nbFs._index.insert(h_nbFs._index.end(), buffer._index.begin() + 1, buffer._index.end());

	buffer = mesh->_nbNs;
	for (uint i = 0u; i < buffer._array.size(); i++)
		buffer._array[i] += _numNodes;
	for (uint i = 0u; i < buffer._index.size(); i++)
		buffer._index[i] += h_nbNs.arraySize();
	h_nbNs._array.insert(h_nbNs._array.end(), buffer._array.begin(), buffer._array.end());
	h_nbNs._index.insert(h_nbNs._index.end(), buffer._index.begin() + 1, buffer._index.end());

	_numFaces += mesh->_numFaces;
	_numNodes += mesh->_numVertices;
	h_nodePhases.resize(_numNodes, newPhase);

	d_impulses.resize(_numNodes * 3u);
	d_colWs.resize(_numNodes);

	initMasses(mass);
	initVelocities();
	initNoramls();

	copyToDevice();
	copyNbToDevice();
	d_fs = h_fs;
	d_nodePhases = h_nodePhases;

	computeNormal();

	setParam();
}
void MeshObject::initMasses(REAL mass) {
	h_ms.resize(_numNodes, mass);

	d_ms.resize(_numNodes);
	d_invMs.resize(_numNodes);

	h_isFixeds.resize(_numNodes, 0u);
	d_isFixeds = h_isFixeds;
}
void MeshObject::initVelocities(void) {
	d_vs.resize(_numNodes * 3u);
	d_vs.memset(0, (*_streams)[2]);
	d_forces.resize(_numNodes * 3u);
}
void MeshObject::initNoramls(void) {
	h_fNorms.resize(_numFaces * 3u);
	h_nNorms.resize(_numNodes * 3u);
	d_fNorms.resize(_numFaces * 3u);
	d_nNorms.resize(_numNodes * 3u);
}
void MeshObject::initBVH(void) {
	_bvh->build(d_fs, d_ns);
	_RTri->init(d_fs, d_nbFs);
}

void MeshObject::computeNormal(void) {
	if (!_numFaces)
		return;
	d_nNorms.memset(0);
	compNormals_kernel << <divup(_numFaces, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_fs(), d_ns(), d_fNorms(), d_nNorms(), _numFaces);
	CUDA_CHECK(cudaPeekAtLastError());
	nodeNormNormalize_kernel << <divup(_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		d_nNorms(), _numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
	copyNormToHost();
}

void MeshObject::draw(void) {
	drawSurface();
#if SMOOTHED_RENDERING == 0
	drawWire();
#endif
}
void MeshObject::drawWire(void)
{
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glColor3d(0, 0, 0);

	for (int i = 0; i < _numFaces; i++) {
		glBegin(GL_LINE_LOOP);
		for (int j = 0; j < 3; j++) {
			auto x = h_ns[h_fs[i * 3 + j] * 3 + 0];
			auto y = h_ns[h_fs[i * 3 + j] * 3 + 1];
			auto z = h_ns[h_fs[i * 3 + j] * 3 + 2];
			glVertex3f(x, y, z);
		}
		glEnd();
	}

	glEnable(GL_LIGHTING);
	glPopMatrix();
}
void MeshObject::drawSurface(void)
{
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1); // turn on two-sided lighting.
	float black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	float white[] = { 0.4f, 0.4f, 0.4f, 1.0f };
	glMaterialfv(GL_FRONT, GL_SPECULAR, white);
	glMaterialf(GL_FRONT, GL_SHININESS, 64);
	glMaterialfv(GL_BACK, GL_SPECULAR, black); // no specular highlights
	
	uint prevPhase = 0xffffffff;
	for (uint i = 0u; i < _numFaces; i++) {
		uint ino0 = h_fs[i * 3u + 0u];
		uint ino1 = h_fs[i * 3u + 1u];
		uint ino2 = h_fs[i * 3u + 2u];
		REAL3 a = make_REAL3(h_ns[ino0 * 3u + 0u], h_ns[ino0 * 3u + 1u], h_ns[ino0 * 3u + 2u]);
		REAL3 b = make_REAL3(h_ns[ino1 * 3u + 0u], h_ns[ino1 * 3u + 1u], h_ns[ino1 * 3u + 2u]);
		REAL3 c = make_REAL3(h_ns[ino2 * 3u + 0u], h_ns[ino2 * 3u + 1u], h_ns[ino2 * 3u + 2u]);

		uint phase = h_nodePhases[ino0];
		if (h_frontColors[phase].w == 0.0)
			continue;
		if (phase != prevPhase) {
			glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, &h_frontColors[phase].x);
			glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, &h_backColors[phase].x);
			prevPhase = phase;
		}

		glBegin(GL_TRIANGLES);

#if SMOOTHED_RENDERING
		glNormal3f(h_nNorms[ino0 * 3u + 0u], h_nNorms[ino0 * 3u + 1u], h_nNorms[ino0 * 3u + 2u]);
		glVertex3f(a.x, a.y, a.z);
		glNormal3f(h_nNorms[ino1 * 3u + 0u], h_nNorms[ino1 * 3u + 1u], h_nNorms[ino1 * 3u + 2u]);
		glVertex3f(b.x, b.y, b.z);
		glNormal3f(h_nNorms[ino2 * 3u + 0u], h_nNorms[ino2 * 3u + 1u], h_nNorms[ino2 * 3u + 2u]);
		glVertex3f(c.x, c.y, c.z);
#else
		glNormal3f(h_fNorms[i * 3u + 0u], h_fNorms[i * 3u + 1u], h_fNorms[i * 3u + 2u]);
		glVertex3f(a.x, a.y, a.z);
		glVertex3f(b.x, b.y, b.z);
		glVertex3f(c.x, c.y, c.z);
#endif

		glEnd();
	}
}

void MeshObject::copyToDevice(void) {
	//d_fs.copyFromHost(h_fs, (*_streams)[0]);
	d_ns.copyFromHost(h_ns, (*_streams)[1]);
}
void MeshObject::copyToHost(void) {
	//d_fs.copyToHost(h_fs, (*_streams)[0]);
	d_ns.copyToHost(h_ns, (*_streams)[1]);
}
void MeshObject::copyNbToDevice(void) {
	d_ses.copyFromHost(h_ses, &(*_streams)[2]);
	d_bes.copyFromHost(h_bes, &(*_streams)[4]);
	d_nbFs.copyFromHost(h_nbFs, &(*_streams)[6]);
	d_nbNs.copyFromHost(h_nbNs, &(*_streams)[8]);
}
void MeshObject::copyNbToHost(void) {
	d_ses.copyToHost(h_ses, &(*_streams)[2]);
	d_bes.copyToHost(h_bes, &(*_streams)[4]);
	d_nbFs.copyToHost(h_nbFs, &(*_streams)[6]);
	d_nbNs.copyToHost(h_nbNs, &(*_streams)[8]);
}
void MeshObject::copyNormToDevice(void) {
	d_fNorms.copyFromHost(h_fNorms, (*_streams)[0]);
	d_nNorms.copyFromHost(h_nNorms, (*_streams)[1]);
}
void MeshObject::copyNormToHost(void) {
	d_fNorms.copyToHost(h_fNorms, (*_streams)[0]);
	d_nNorms.copyToHost(h_nNorms, (*_streams)[1]);
}