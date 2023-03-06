#include "Cloth.h"

void Cloth::init(void) {
	 MeshObject::init();

	 _param = new ClothParam();
	 _type = TYPE_MESH_CLOTH;
}
void Cloth::reset(void) {
	h_fs = h_fs0;
	h_ns = h_ns0;
	h_ses = h_ses0;
	h_bes = h_bes0;
	h_nbFs = h_nbFs0;
	h_nbNs = h_nbNs0;
	h_ms = h_ms0;
	h_isFixeds = h_isFixeds0;
	h_nodePhases = h_nodePhases0;

	d_restSolidFractions.resize(_numNodes);
	d_maxFluidMasses.resize(_numNodes);
	d_maxFluidMasses.memset(0.0);
	d_restSolidFractions.copyToHost(h_restSolidFractions);
	d_maxFluidMasses.copyToHost(h_maxFluidMasses);

	d_mfs.resize(_numNodes);
	d_ss.resize(_numNodes);
	d_mfs.memset(0.0);
	d_mfs.copyToHost(h_mfs);
	d_ss.copyToHost(h_ss);

	_numFaces = h_fs.size() / 3u;
	_numNodes = h_ns.size() / 3u;

	initConstraints();

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
void Cloth::addCloth(
	Mesh* mesh, REAL mass, REAL thickness, REAL friction, 
	REAL restSolidFraction, REAL maxFluidMass,
	float4 frontColor, float4 backColor, bool isSaved)
{
	uint numSes0 = h_ses.arraySize();
	uint numBes0 = h_bes.arraySize();

	addMesh(mesh, mass, thickness, friction, frontColor, backColor);
#if SCENE==0
#if QUALITY==0
	h_stretchWs.push_back(1000000.0 * mass);
	h_bendingWs.push_back(100000.0 * mass);
#elif QUALITY==1
	h_stretchWs.push_back(1000000.0 * mass);
	h_bendingWs.push_back(100000.0 * mass);
#else
	h_stretchWs.push_back(1000000.0 * mass);
	h_bendingWs.push_back(100000.0 * mass);
#endif
#elif SCENE==1
#if QUALITY==0
	h_stretchWs.push_back(800000.0 * mass);
	h_bendingWs.push_back(80000.0 * mass);
#elif QUALITY==1
	h_stretchWs.push_back(2000000.0 * mass);
	h_bendingWs.push_back(200000.0 * mass);
#else
	h_stretchWs.push_back(5000000.0 * mass);
	h_bendingWs.push_back(500000.0 * mass);
#endif
#elif SCENE==2
#if QUALITY==0
	h_stretchWs.push_back(30000.0 * mass);
	h_bendingWs.push_back(3000.0 * mass);
#elif QUALITY==1
	h_stretchWs.push_back(300000.0 * mass);
	h_bendingWs.push_back(30000.0 * mass);
#else
	h_stretchWs.push_back(800000.0 * mass);
	h_bendingWs.push_back(80000.0 * mass);
#endif
#elif SCENE==3
#if QUALITY==0
	h_stretchWs.push_back(800000.0 * mass);
	h_bendingWs.push_back(80000.0 * mass);
#elif QUALITY==1
	h_stretchWs.push_back(3000000.0 * mass);
	h_bendingWs.push_back(300000.0 * mass);
#else
	h_stretchWs.push_back(5000000.0 * mass);
	h_bendingWs.push_back(500000.0 * mass);
#endif
#elif SCENE==4
#if QUALITY==0
	h_stretchWs.push_back(800000.0 * mass);
	h_bendingWs.push_back(80000.0 * mass);
#elif QUALITY==1
	h_stretchWs.push_back(1000000.0 * mass);
	h_bendingWs.push_back(100000.0 * mass);
#else
	h_stretchWs.push_back(2000000.0 * mass);
	h_bendingWs.push_back(200000.0 * mass);
#endif
#endif

	uint numSes = h_ses.arraySize();
	uint numBes = h_bes.arraySize();
	for (uint i = numSes0; i < numSes; i += 2) {
		uint ino0 = h_ses._array[i];
		uint ino1 = h_ses._array[i + 1u];
		REAL3 n0 = make_REAL3(h_ns[ino0 * 3u + 0u], h_ns[ino0 * 3u + 1u], h_ns[ino0 * 3u + 2u]);
		REAL3 n1 = make_REAL3(h_ns[ino1 * 3u + 0u], h_ns[ino1 * 3u + 1u], h_ns[ino1 * 3u + 2u]);
		h_stretchRLs.push_back(Length(n0 - n1));
		h_edgeLimits.push_back(1.8 + 0.2 * (REAL)rand() / (REAL)RAND_MAX);
	}
	for (uint i = numBes0; i < numBes; i += 2) {
		uint ino0 = h_bes._array[i];
		uint ino1 = h_bes._array[i + 1u];
		REAL3 n0 = make_REAL3(h_ns[ino0 * 3u + 0u], h_ns[ino0 * 3u + 1u], h_ns[ino0 * 3u + 2u]);
		REAL3 n1 = make_REAL3(h_ns[ino1 * 3u + 0u], h_ns[ino1 * 3u + 1u], h_ns[ino1 * 3u + 2u]);
		h_bendingRLs.push_back(Length(n0 - n1));
	}

	d_stretchWs = h_stretchWs;
	d_bendingWs = h_bendingWs;
	d_stretchRLs = h_stretchRLs;
	d_bendingRLs = h_bendingRLs;

	d_restSolidFractions.resize(_numNodes);
	d_maxFluidMasses.resize(_numNodes);
	d_maxFluidMasses.memset(0.0);
	d_restSolidFractions.copyToHost(h_restSolidFractions);
	d_maxFluidMasses.copyToHost(h_maxFluidMasses);

	d_mfs.resize(_numNodes);
	d_ss.resize(_numNodes);
	d_mfs.memset(0.0);
	d_mfs.copyToHost(h_mfs);
	d_ss.copyToHost(h_ss);

#if SCENE==0
	//fix();
#elif SCENE==1
	fix();
#elif SCENE==2
#elif SCENE==3
	fix();
#elif SCENE==4
	fix();
#endif

	if (isSaved) {
		h_fs0 = h_fs;
		h_ns0 = h_ns;
		h_ses0 = h_ses;
		h_bes0 = h_bes;
		h_nbFs0 = h_nbFs;
		h_nbNs0 = h_nbNs;
		h_ms0 = h_ms;
		h_isFixeds0 = h_isFixeds;
		h_nodePhases0 = h_nodePhases;
	}
	initBVH();
	setParam();
}
void Cloth::initConstraints(void) {
	uint numSes = h_ses.arraySize();
	uint numBes = h_bes.arraySize();
	for (uint i = 0; i < numSes; i += 2) {
		uint ino0 = h_ses._array[i];
		uint ino1 = h_ses._array[i + 1u];
		REAL3 n0 = make_REAL3(h_ns[ino0 * 3u + 0u], h_ns[ino0 * 3u + 1u], h_ns[ino0 * 3u + 2u]);
		REAL3 n1 = make_REAL3(h_ns[ino1 * 3u + 0u], h_ns[ino1 * 3u + 1u], h_ns[ino1 * 3u + 2u]);
		h_stretchRLs.push_back(Length(n0 - n1));
		h_edgeLimits.push_back(1.8 + 0.2 * (REAL)rand() / (REAL)RAND_MAX);
	}
	for (uint i = 0; i < numBes; i += 2) {
		uint ino0 = h_bes._array[i];
		uint ino1 = h_bes._array[i + 1u];
		REAL3 n0 = make_REAL3(h_ns[ino0 * 3u + 0u], h_ns[ino0 * 3u + 1u], h_ns[ino0 * 3u + 2u]);
		REAL3 n1 = make_REAL3(h_ns[ino1 * 3u + 0u], h_ns[ino1 * 3u + 1u], h_ns[ino1 * 3u + 2u]);
		h_bendingRLs.push_back(Length(n0 - n1));
	}
}
void Cloth::fix(void) {
#if SCENE==0
	uint mxmy = 0u;
	uint Mxmy = 0u;
	uint mxMy = 0u;
	uint MxMy = 0u;
	for (uint i = 1u; i < _numNodes; i++) {
		REAL3 n = make_REAL3(h_ns[i * 3u + 0u], h_ns[i * 3u + 1u], h_ns[i * 3u + 2u]);

		if (n.x - h_ns[mxmy * 3u + 0u] < 1.0e-5 && n.z - h_ns[mxmy * 3u + 2u] < 1.0e-5)
			mxmy = i;
		if (n.x - h_ns[Mxmy * 3u + 0u] > -1.0e-5 && n.z - h_ns[Mxmy * 3u + 2u] < 1.0e-5)
			Mxmy = i;
		if (n.x - h_ns[mxMy * 3u + 0u] < 1.0e-5 && n.z - h_ns[mxMy * 3u + 2u] > -1.0e-5)
			mxMy = i;
		if (n.x - h_ns[MxMy * 3u + 0u] > -1.0e-5 && n.z - h_ns[MxMy * 3u + 2u] > -1.0e-5)
			MxMy = i;
	}

	h_isFixeds[Mxmy] = 1u;
	h_isFixeds[MxMy] = 1u;
	h_isFixeds[mxmy] = 1u;
	h_isFixeds[mxMy] = 1u;
	/*uint mxMy = 0u;
	uint MxMy = 0u;
	for (uint i = 1u; i < _numNodes; i++) {
		REAL3 n = make_REAL3(h_ns[i * 3u + 0u], h_ns[i * 3u + 1u], h_ns[i * 3u + 2u]);

		if (n.x - h_ns[mxMy * 3u + 0u] < 1.0e-5 && n.y - h_ns[mxMy * 3u + 1u] > -1.0e-5)
			mxMy = i;
		if (n.x - h_ns[MxMy * 3u + 0u] > -1.0e-5 && n.y - h_ns[MxMy * 3u + 1u] > -1.0e-5)
			MxMy = i;
	}

	h_isFixeds[MxMy] = 1u;
	h_isFixeds[mxMy] = 1u;*/
#elif SCENE==1
	uint mxmy = 0u;
	uint Mxmy = 0u;
	uint mxMy = 0u;
	uint MxMy = 0u;
	for (uint i = 1u; i < _numNodes; i++) {
		REAL3 n = make_REAL3(h_ns[i * 3u + 0u], h_ns[i * 3u + 1u], h_ns[i * 3u + 2u]);

		if (n.x - h_ns[mxmy * 3u + 0u] < 1.0e-5 && n.z - h_ns[mxmy * 3u + 2u] < 1.0e-5)
			mxmy = i;
		if (n.x - h_ns[Mxmy * 3u + 0u] > -1.0e-5 && n.z - h_ns[Mxmy * 3u + 2u] < 1.0e-5)
			Mxmy = i;
		if (n.x - h_ns[mxMy * 3u + 0u] < 1.0e-5 && n.z - h_ns[mxMy * 3u + 2u] > -1.0e-5)
			mxMy = i;
		if (n.x - h_ns[MxMy * 3u + 0u] > -1.0e-5 && n.z - h_ns[MxMy * 3u + 2u] > -1.0e-5)
			MxMy = i;
	}

	h_isFixeds[Mxmy] = 1u;
	h_isFixeds[MxMy] = 1u;
	h_isFixeds[mxmy] = 1u;
	h_isFixeds[mxMy] = 1u;
#elif SCENE==2
#elif SCENE==3
	/*uint mxMy = 0u;
	uint MxMy = 0u;
	for (uint i = 1u; i < _numNodes; i++) {
		REAL3 n = make_REAL3(h_ns[i * 3u + 0u], h_ns[i * 3u + 1u], h_ns[i * 3u + 2u]);

		if (n.x - h_ns[mxMy * 3u + 0u] < 1.0e-5 && n.y - h_ns[mxMy * 3u + 1u] > -1.0e-5)
			mxMy = i;
		if (n.x - h_ns[MxMy * 3u + 0u] > -1.0e-5 && n.y - h_ns[MxMy * 3u + 1u] > -1.0e-5)
			MxMy = i;
	}

	h_isFixeds[MxMy] = 1u;
	h_isFixeds[mxMy] = 1u;*/
	uint mxmy = 0u;
	uint Mxmy = 0u;
	uint mxMy = 0u;
	uint MxMy = 0u;
	for (uint i = 1u; i < _numNodes; i++) {
		REAL3 n = make_REAL3(h_ns[i * 3u + 0u], h_ns[i * 3u + 1u], h_ns[i * 3u + 2u]);

		if (n.x - h_ns[mxmy * 3u + 0u] < 1.0e-5 && n.y - h_ns[mxmy * 3u + 1u] < 1.0e-5)
			mxmy = i;
		if (n.x - h_ns[Mxmy * 3u + 0u] > -1.0e-5 && n.y - h_ns[Mxmy * 3u + 1u] < 1.0e-5)
			Mxmy = i;
		if (n.x - h_ns[mxMy * 3u + 0u] < 1.0e-5 && n.y - h_ns[mxMy * 3u + 1u] > -1.0e-5)
			mxMy = i;
		if (n.x - h_ns[MxMy * 3u + 0u] > -1.0e-5 && n.y - h_ns[MxMy * 3u + 1u] > -1.0e-5)
			MxMy = i;
	}

	h_isFixeds[Mxmy] = 1u;
	h_isFixeds[MxMy] = 1u;
	h_isFixeds[mxmy] = 1u;
	h_isFixeds[mxMy] = 1u;
#elif SCENE==4
	/*uint mxmy = 0u;
	uint Mxmy = 0u;
	uint mxMy = 0u;
	uint MxMy = 0u;
	for (uint i = 1u; i < _numNodes; i++) {
		REAL3 n = make_REAL3(h_ns[i * 3u + 0u], h_ns[i * 3u + 1u], h_ns[i * 3u + 2u]);

		if (n.x - h_ns[mxmy * 3u + 0u] < 1.0e-5 && n.z - h_ns[mxmy * 3u + 2u] < 1.0e-5)
			mxmy = i;
		if (n.x - h_ns[Mxmy * 3u + 0u] > -1.0e-5 && n.z - h_ns[Mxmy * 3u + 2u] < 1.0e-5)
			Mxmy = i;
		if (n.x - h_ns[mxMy * 3u + 0u] < 1.0e-5 && n.z - h_ns[mxMy * 3u + 2u] > -1.0e-5)
			mxMy = i;
		if (n.x - h_ns[MxMy * 3u + 0u] > -1.0e-5 && n.z - h_ns[MxMy * 3u + 2u] > -1.0e-5)
			MxMy = i;
	}

	h_isFixeds[Mxmy] = 1u;
	h_isFixeds[MxMy] = 1u;
	h_isFixeds[mxmy] = 1u;
	h_isFixeds[mxMy] = 1u;*/
	for (uint i = 0u; i < _numNodes; i++) {
		REAL3 n = make_REAL3(h_ns[i * 3u + 0u], h_ns[i * 3u + 1u], h_ns[i * 3u + 2u]);

		if (n.x < -1.0 + 0.05 || n.x > 1.0 - 0.05)
			h_isFixeds[i] = 1u;
	}
#endif


	d_isFixeds = h_isFixeds;
}
void Cloth::moveFixed(REAL3 vel) {
	vector<REAL> h_vs;
	d_vs.copyToHost(h_vs);

	for (uint i = 0u; i < _numNodes; i++) {
		if (h_isFixeds[i]) {
			h_vs[i * 3u + 0u] = vel.x;
			h_vs[i * 3u + 1u] = vel.y;
			h_vs[i * 3u + 2u] = vel.z;
		}
	}
	d_vs = h_vs;
}
void Cloth::rotateFixed(REAL3 degreeL, REAL3 degreeR, REAL moveL, REAL moveR, REAL invdt) {
	vector<REAL> h_vs;
	d_vs.copyToHost(h_vs);

	REAL3 a, b, pa, pb, center;
	REAL3 pivot, va, vb;
	REAL cx, sx, cy, sy, cz, sz;

	REAL3 degree = degreeL * M_PI * 0.00555555555555555555555555555556;
	cx = cos(degree.x);
	sx = sin(degree.x);
	cy = cos(degree.y);
	sy = -sin(degree.y);
	cz = cos(degree.z);
	sz = sin(degree.z);

	for (uint i = 0u; i < _numNodes; i++) {
		if (h_isFixeds[i]) {
			a = make_REAL3(h_ns[i * 3u + 0u], h_ns[i * 3u + 1u], h_ns[i * 3u + 2u]);
			if (a.x < 0.0) {
				pa.x = a.x * cz * cy + a.y * (cz * sy * sx - sz * cx) + a.z * (cz * sy * cx + sz * sx);
				pa.y = a.x * sz * cy + a.y * (sz * sy * sx + cz * cx) + a.z * (sz * sy * cx - cz * sx);
				pa.z = a.x * -sy + a.y * cy * sx + a.z * cy * cx;

				pa.x += moveL;

				va = invdt * (pa - a);
				h_vs[i * 3u + 0u] = va.x;
				h_vs[i * 3u + 1u] = va.y;
				h_vs[i * 3u + 2u] = va.z;
			}
		}
	}

	degree = degreeR * M_PI * 0.00555555555555555555555555555556;
	cx = cos(degree.x);
	sx = sin(degree.x);
	cy = cos(degree.y);
	sy = -sin(degree.y);
	cz = cos(degree.z);
	sz = sin(degree.z);

	for (uint i = 0u; i < _numNodes; i++) {
		if (h_isFixeds[i]) {
			a = make_REAL3(h_ns[i * 3u + 0u], h_ns[i * 3u + 1u], h_ns[i * 3u + 2u]);
			if (a.x > 0.0) {
				pa.x = a.x * cz * cy + a.y * (cz * sy * sx - sz * cx) + a.z * (cz * sy * cx + sz * sx);
				pa.y = a.x * sz * cy + a.y * (sz * sy * sx + cz * cx) + a.z * (sz * sy * cx - cz * sx);
				pa.z = a.x * -sy + a.y * cy * sx + a.z * cy * cx;

				pa.x -= moveR;

				va = invdt * (pa - a);
				h_vs[i * 3u + 0u] = va.x;
				h_vs[i * 3u + 1u] = va.y;
				h_vs[i * 3u + 2u] = va.z;
			}
		}
	}
	d_vs = h_vs;
}

void Cloth::drawSurface(void)
{
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1); // turn on two-sided lighting.
	float black[] = { 0.0f, 0.0f, 0.0f, 1.0f };
	float white[] = { 0.4f, 0.4f, 0.4f, 1.0f };
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 64);

	uint prevPhase = 0xffffffff;
	for (uint i = 0u; i < _numFaces; i++) {
		/*if (i == _bvh->h_triInfos[_bvh->_test]._id) {
			printf("%d, %d, %lu\n", i, _bvh->h_triInfos[_bvh->_test]._id, _bvh->h_triInfos[_bvh->_test]._pos);
			uint ino0 = h_fs[i * 3u + 0u];
			uint ino1 = h_fs[i * 3u + 1u];
			uint ino2 = h_fs[i * 3u + 2u];
			REAL3 a = make_REAL3(h_ns[ino0 * 3u + 0u], h_ns[ino0 * 3u + 1u], h_ns[ino0 * 3u + 2u]);
			REAL3 b = make_REAL3(h_ns[ino1 * 3u + 0u], h_ns[ino1 * 3u + 1u], h_ns[ino1 * 3u + 2u]);
			REAL3 c = make_REAL3(h_ns[ino2 * 3u + 0u], h_ns[ino2 * 3u + 1u], h_ns[ino2 * 3u + 2u]);

			float frontColor[4] = { 1.f, 0.f, 0.f, 1.f };
			float backColor[4] = { 1.f, 0.f, 0.f, 1.f };
			glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, frontColor);
			glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, backColor);

			glBegin(GL_TRIANGLES);

			glNormal3f(h_fNorms[i * 3u + 0u], h_fNorms[i * 3u + 1u], h_fNorms[i * 3u + 2u]);
			glVertex3f(a.x, a.y, a.z);
			glVertex3f(b.x, b.y, b.z);
			glVertex3f(c.x, c.y, c.z);

			glEnd();
			
			continue;
		}*/
		uint ino0 = h_fs[i * 3u + 0u];
		uint ino1 = h_fs[i * 3u + 1u];
		uint ino2 = h_fs[i * 3u + 2u];
		REAL3 a = make_REAL3(h_ns[ino0 * 3u + 0u], h_ns[ino0 * 3u + 1u], h_ns[ino0 * 3u + 2u]);
		REAL3 b = make_REAL3(h_ns[ino1 * 3u + 0u], h_ns[ino1 * 3u + 1u], h_ns[ino1 * 3u + 2u]);
		REAL3 c = make_REAL3(h_ns[ino2 * 3u + 0u], h_ns[ino2 * 3u + 1u], h_ns[ino2 * 3u + 2u]);

		uint phase = h_nodePhases[ino0];
		if (h_frontColors[phase].w == 0.0)
			continue;

		float frontColor[4];
		float backColor[4];

		/*float s0 = powf(1.f - min((1.0 - h_restSolidFractions[ino0]) * max(min(h_ss[ino0], 1.0), 0.0), 1.f), 1.0f);
		float s1 = powf(1.f - min((1.0 - h_restSolidFractions[ino1]) * max(min(h_ss[ino1], 1.0), 0.0), 1.f), 1.0f);
		float s2 = powf(1.f - min((1.0 - h_restSolidFractions[ino2]) * max(min(h_ss[ino2], 1.0), 0.0), 1.f), 1.0f);*/

		float s0, s1, s2;
		if (h_maxFluidMasses[ino0] > 0.0)
			s0 = pow(1.f - (1.0 - h_restSolidFractions[ino0]) * max(min(h_mfs[ino0] / (h_maxFluidMasses[ino0] + FLT_EPSILON), 1.0), 0.0), 1.0);
		else s0 = 1.f;
		if (h_maxFluidMasses[ino1] > 0.0)
			s1 = pow(1.f - (1.0 - h_restSolidFractions[ino1]) * max(min(h_mfs[ino1] / (h_maxFluidMasses[ino1] + FLT_EPSILON), 1.0), 0.0), 1.0);
		else s1 = 1.f;
		if (h_maxFluidMasses[ino2] > 0.0)
			s2 = pow(1.f - (1.0 - h_restSolidFractions[ino2]) * max(min(h_mfs[ino2] / (h_maxFluidMasses[ino2] + FLT_EPSILON), 1.0), 0.0), 1.0);
		else s2 = 1.f;

		frontColor[3] = backColor[3] = 1.0;

		glBegin(GL_TRIANGLES);

#if SMOOTHED_RENDERING
		for (int j = 0; j < 3; j++) {
			frontColor[j] = (&h_frontColors[phase].x)[j] * s0;
			backColor[j] = (&h_backColors[phase].x)[j] * s0;
		}

		glMaterialfv(GL_FRONT, GL_DIFFUSE, frontColor);
		glMaterialfv(GL_BACK, GL_DIFFUSE, backColor);

		glNormal3f(h_nNorms[ino0 * 3u + 0u], h_nNorms[ino0 * 3u + 1u], h_nNorms[ino0 * 3u + 2u]);
		glVertex3f(a.x, a.y, a.z);

		for (int j = 0; j < 3; j++) {
			frontColor[j] = (&h_frontColors[phase].x)[j] * s1;
			backColor[j] = (&h_backColors[phase].x)[j] * s1;
		}

		glMaterialfv(GL_FRONT, GL_DIFFUSE, frontColor);
		glMaterialfv(GL_BACK, GL_DIFFUSE, backColor);

		glNormal3f(h_nNorms[ino1 * 3u + 0u], h_nNorms[ino1 * 3u + 1u], h_nNorms[ino1 * 3u + 2u]);
		glVertex3f(b.x, b.y, b.z);

		for (int j = 0; j < 3; j++) {
			frontColor[j] = (&h_frontColors[phase].x)[j] * s2;
			backColor[j] = (&h_backColors[phase].x)[j] * s2;
		}

		glMaterialfv(GL_FRONT, GL_DIFFUSE, frontColor);
		glMaterialfv(GL_BACK, GL_DIFFUSE, backColor);

		glNormal3f(h_nNorms[ino2 * 3u + 0u], h_nNorms[ino2 * 3u + 1u], h_nNorms[ino2 * 3u + 2u]);
		glVertex3f(c.x, c.y, c.z);
#else
		glNormal3f(h_fNorms[i * 3u + 0u], h_fNorms[i * 3u + 1u], h_fNorms[i * 3u + 2u]);

		for (int j = 0; j < 3; j++) {
			frontColor[j] = (&h_frontColors[phase].x)[j] * s0;
			backColor[j] = (&h_backColors[phase].x)[j] * s0;
		}
		glMaterialfv(GL_FRONT, GL_DIFFUSE, frontColor);
		glMaterialfv(GL_BACK, GL_DIFFUSE, backColor);

		glVertex3f(a.x, a.y, a.z);

		for (int j = 0; j < 3; j++) {
			frontColor[j] = (&h_frontColors[phase].x)[j] * s1;
			backColor[j] = (&h_backColors[phase].x)[j] * s1;
		}
		glMaterialfv(GL_FRONT, GL_DIFFUSE, frontColor);
		glMaterialfv(GL_BACK, GL_DIFFUSE, backColor);

		glVertex3f(b.x, b.y, b.z);

		for (int j = 0; j < 3; j++) {
			frontColor[j] = (&h_frontColors[phase].x)[j] * s2;
			backColor[j] = (&h_backColors[phase].x)[j] * s2;
		}
		glMaterialfv(GL_FRONT, GL_DIFFUSE, frontColor);
		glMaterialfv(GL_BACK, GL_DIFFUSE, backColor);

		glVertex3f(c.x, c.y, c.z);
#endif

		glEnd();
	}
}
void Cloth::copyToHost(void) {
	MeshObject::copyToHost();
	d_restSolidFractions.copyToHost(h_restSolidFractions);
	d_maxFluidMasses.copyToHost(h_maxFluidMasses);
	d_mfs.copyToHost(h_mfs);
	d_ss.copyToHost(h_ss);
}