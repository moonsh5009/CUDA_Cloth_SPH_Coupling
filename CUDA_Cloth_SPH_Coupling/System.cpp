#include "System.h"

void System::init(REAL3& gravity, REAL dt) {
	_gravity = gravity;
	_dt = dt;
	_invdt = 1.0 / dt;

	_boundary._min = make_REAL3(-1.5, -1.5, -1.5);
	_boundary._max = make_REAL3(1.5, 1.5, 1.5);

	_cloths = new Cloth();
	_obstacles = new Obstacle();
	
	_boundaryParticles = new BoundaryParticle();
	_sphParticles = new SPHParticle();
	_poreParticles = new PoreParticle();

	_hash = new SpatialHashing(make_uint3(64u, 64u, 64u));
	_frame = 0u;
}

void System::addCloth(
	Mesh* mesh, REAL friction,
	REAL radius, REAL restDensity, REAL restFluidDensity, REAL restSolidFraction,
	REAL viscosity, REAL surfaceTension,
	float4 frontColor, float4 backColor, bool isSaved)
{
	REAL mass = radius * radius * radius * M_PI * 4.0 / 3.0 * restDensity;
	REAL fluidMass = radius * radius * radius * M_PI * 4.0 / 3.0 * restFluidDensity * (1.0 - restSolidFraction);
	_cloths->addCloth(mesh, mass, radius, friction, restSolidFraction, fluidMass, frontColor, backColor, isSaved);
	_poreParticles->addModel(radius, restDensity, restFluidDensity, restSolidFraction, viscosity, surfaceTension, frontColor);
	_poreParticles->d_shortEs.resize(_cloths->_numFaces);
	_poreParticles->d_sampNums.resize(_cloths->_numFaces);
	_poreParticles->d_shortEs.memset(0);
	_poreParticles->d_sampNums.memset(0);
	_poreParticles->d_shortEs.copyToHost(_poreParticles->h_shortEs0);
	_poreParticles->d_sampNums.copyToHost(_poreParticles->h_sampNums0);
	_poreParticles->setParam();

	if (_hash->_radius < radius * SPH_RADIUS_RATIO)
		_hash->setRadius(radius * SPH_RADIUS_RATIO);
}
void System::addObstacle(
	Mesh* mesh, REAL mass, REAL friction,
	REAL3& pivot, REAL3& rotation,
	REAL radius, REAL viscosity, REAL surfaceTension,
	float4 frontColor, float4 backColor, bool isSaved)
{
	_obstacles->addObject(mesh, mass, radius, friction, frontColor, backColor, pivot, rotation, isSaved);
	_boundaryParticles->addModel(radius, viscosity, surfaceTension, frontColor);
	_boundaryParticles->d_shortEs.resize(_obstacles->_numFaces);
	_boundaryParticles->d_sampNums.resize(_obstacles->_numFaces);
	_boundaryParticles->d_shortEs.memset(0);
	_boundaryParticles->d_sampNums.memset(0);
	_boundaryParticles->d_shortEs.copyToHost(_boundaryParticles->h_shortEs0);
	_boundaryParticles->d_sampNums.copyToHost(_boundaryParticles->h_sampNums0);
	_boundaryParticles->setParam();

	if (_hash->_radius < radius * SPH_RADIUS_RATIO)
		_hash->setRadius(radius * SPH_RADIUS_RATIO);
}
void System::addSPHModel(
	REAL radius, REAL restDensity,
	REAL viscosity, REAL surfaceTension, float4 color)
{
	_sphParticles->addModel(radius, restDensity, viscosity, surfaceTension, color);
	_sphParticles->setParam();

	if (_hash->_radius < radius * SPH_RADIUS_RATIO)
		_hash->setRadius(radius * SPH_RADIUS_RATIO);
}

void System::spawnSPHParticle(uchar spawnType) {
	REAL radius = _sphParticles->h_radii[0];
	REAL wide = (radius + radius);
	uint spawnPhase = 0u;

	REAL3 box = _boundary._max - _boundary._min - wide;
	REAL3 pos0, pos;
	int w, h, d;
	switch (spawnType) {
	case 1:
#if SCENE==0
		/*w = (int)(box.x / wide * 0.2 * 0.5);
		d = (int)(box.z / wide * 0.2 * 0.5);
		w = min(w, d);
		pos0 = make_REAL3(
			(_boundary._min.x + _boundary._max.x) * 0.5,
			_boundary._max.y - wide - wide - wide,
			(_boundary._min.z + _boundary._max.z) * 0.5);

		for (int i = -w; i < w; i++) {
			for (int j = 0; j > -w - w; j--) {
				for (int k = -w; k < w; k++) {
					pos = make_REAL3(
						pos0.x + wide * i,
						pos0.y + wide * j,
						pos0.z + wide * k);
					_sphParticles->addParticle(pos, make_REAL3(0.0), spawnPhase);
				}
			}
		}*/
		w = (int)(box.x / wide * 0.12 * 0.3);
		d = (int)(box.z / wide * 0.12 * 0.3);
		w = min(w, d);
		pos0 = make_REAL3(
			(_boundary._min.x + _boundary._max.x) * 0.5,
			_boundary._min.y + wide + wide + wide,
			(_boundary._min.z + _boundary._max.z) * 0.5);

		for (int i = -w; i < w; i++) {
			for (int j = 0; j < w + w; j++) {
				for (int k = -w; k < w; k++) {
					pos = make_REAL3(
						pos0.x + wide * i,
						pos0.y + wide * j,
						pos0.z + wide * k);
					_sphParticles->addParticle(pos, make_REAL3(0.0), spawnPhase);
				}
			}
		}
#elif SCENE==2
		w = (int)(box.x / wide * 0.2 * 0.5);
		d = (int)(box.z / wide * 0.2 * 0.5);
		w = min(w, d);
		pos0 = make_REAL3(
			(_boundary._min.x + _boundary._max.x) * 0.5,
			_boundary._max.y - wide * 30.0,
			(_boundary._min.z + _boundary._max.z) * 0.5);

		for (int i = -w; i < w; i++) {
			for (int j = 0; j > -w - w; j--) {
				for (int k = -w; k < w; k++) {
					pos = make_REAL3(
						pos0.x + wide * i,
						pos0.y + wide * j,
						pos0.z + wide * k);
					_sphParticles->addParticle(pos, make_REAL3(0.0), spawnPhase);
				}
			}
		}
#elif SCENE==3
		w = (int)(box.x / wide * 0.2 * 0.5);
		d = (int)(box.z / wide * 0.2 * 0.5);
		w = min(w, d);
		pos0 = make_REAL3(
			(_boundary._min.x + _boundary._max.x) * 0.5,
			_boundary._max.y - 0.4,
			_boundary._max.z - wide * 2);

		for (int i = -w; i < w; i++) {
			for (int j = 0; j > -w - w; j--) {
				for (int k = 0; k < w + w; k++) {
					pos = make_REAL3(
						pos0.x + wide * i,
						pos0.y + wide * j,
						pos0.z - wide * k);
					_sphParticles->addParticle(pos, make_REAL3(0.0, 0.0, -5.5), spawnPhase);
				}
			}
		}
#else
		w = (int)(box.x / wide * 0.2 * 0.5);
		d = (int)(box.z / wide * 0.2 * 0.5);
		w = min(w, d);
		pos0 = make_REAL3(
			(_boundary._min.x + _boundary._max.x) * 0.5,
			_boundary._max.y - wide - wide - wide,
			(_boundary._min.z + _boundary._max.z) * 0.5);

		for (int i = -w; i < w; i++) {
			for (int j = 0; j > -w - w; j--) {
				for (int k = -w; k < w; k++) {
					pos = make_REAL3(
						pos0.x + wide * i,
						pos0.y + wide * j,
						pos0.z + wide * k);
					_sphParticles->addParticle(pos, make_REAL3(0.0), spawnPhase);
				}
			}
		}
#endif
		break;
	case 2:
		/*w = (int)floor(box.x / wide * 1.0);
		h = (int)floor(box.y / wide * 0.65);
		d = (int)floor(box.z / wide * 0.16); */
		w = (int)floor(box.x / wide * 0.3);
		h = (int)floor(box.y / wide * 0.5);
		d = (int)floor(box.z / wide * 0.3);

		pos0 = make_REAL3(
			_boundary._min.x + wide,
			_boundary._min.y + wide,
			_boundary._min.z + wide);
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				for (int k = 0; k < d; k++) {
					pos = make_REAL3(
						pos0.x + wide * i,
						pos0.y + wide * j,
						pos0.z + wide * k);
					_sphParticles->addParticle(pos, make_REAL3(0.0), spawnPhase);
				}
			}
		}
		break;
	case 3:
		w = (int)floor(box.x / wide * 0.05);
		h = (int)floor(box.y / wide * 0.28);
		d = (int)floor(box.z / wide);

		pos0 = make_REAL3(
			_boundary._min.x + wide,
			_boundary._min.y + wide,
			_boundary._max.z - wide);
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				for (int k = 0; k < d; k++) {
					pos = make_REAL3(
						pos0.x + wide * i,
						pos0.y + wide * j,
						pos0.z - wide * k);
					_sphParticles->addParticle(pos, make_REAL3(0.0), spawnPhase);
				}
			}
		}
		break;
	case 4:
		w = (int)floor(box.x / wide);
		h = (int)floor(box.y / wide * 0.2);
		d = (int)floor(box.z / wide);

		pos0 = make_REAL3(
			_boundary._min.x + wide,
			_boundary._min.y + wide,
			_boundary._max.z - wide);
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				for (int k = 0; k < d; k++) {
					pos = make_REAL3(
						pos0.x + wide * i,
						pos0.y + wide * j,
						pos0.z - wide * k);
					_sphParticles->addParticle(pos, make_REAL3(0.0), spawnPhase);
				}
			}
		}
		break;
	case 5:
		w = (int)ceil(box.x / wide * 0.16);
		h = (int)ceil(13.0 / wide);
		d = (int)ceil(box.z / wide);
		pos0 = make_REAL3(
			_boundary._min.x + box.x * 0.34,
			_boundary._min.y + radius,
			_boundary._min.z + radius);
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				for (int k = 0; k < d; k++) {
					pos = make_REAL3(
						pos0.x + wide * i,
						pos0.y + wide * j,
						pos0.z + wide * k);
					_sphParticles->addParticle(pos, make_REAL3(0.0), spawnPhase);
				}
			}
		}
		break;
	case 6:
		w = (int)ceil(box.x / wide) - 1;
		h = (int)ceil(box.y / wide * 0.8) - 1;
		d = (int)ceil(box.z / wide) - 1;

		pos0 = make_REAL3(
			_boundary._min.x + wide,
			_boundary._max.y - radius * box.y / wide * 0.05,
			_boundary._min.z + wide);
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {
				for (int k = 0; k < d; k++) {
					pos = make_REAL3(
						pos0.x + wide * i,
						pos0.y - wide * j,
						pos0.z + wide * k);
					_sphParticles->addParticle(pos, make_REAL3(0.0), spawnPhase);
				}
			}
		}
		break;
	}
	printf("SPH Partice °³¼ö: %d\n", _sphParticles->_numParticles);
	_sphParticles->updateParticle();
}

void System::getParticleNeighors(bool sorting) {
	_hash->insert(_sphParticles, _poreParticles, _boundaryParticles);
	_hash->getNeighbors(_sphParticles, _poreParticles, _boundaryParticles);
}
void System::compDivergenceFree(REAL dt) {
	REAL h_sumError;
	REAL* d_sumError;
	CUDA_CHECK(cudaMalloc((void**)&d_sumError, sizeof(REAL)));

	_sphParticles->d_ps.memset(0);
	Dvector<REAL> sphV0s;
	Dvector<REAL> clothV0s;
	sphV0s = _sphParticles->d_vs;
	clothV0s = _cloths->d_vs;

	uint l = 0;
	while (l < 100u) {
		ParticleSampling::lerpVelocity(_cloths, _poreParticles);

		SPHSolver::compDFStiffness(_sphParticles, _poreParticles, _boundaryParticles, dt, d_sumError);

		CUDA_CHECK(cudaMemcpy(&h_sumError, d_sumError, sizeof(REAL), cudaMemcpyDeviceToHost));
		h_sumError /= (REAL)_sphParticles->_numParticles + FLT_EPSILON;

		if (h_sumError < 1.0e-3 && l >= 1)
			break;

		_cloths->d_forces.memset(0);
		_sphParticles->d_forces.memset(0);
		_poreParticles->d_forces.memset(0);

		SPHSolver::applyDFSPH(_sphParticles, _poreParticles, _boundaryParticles);

		ParticleSampling::lerpForce(_cloths, _poreParticles);
		Simulation::applyForce(_cloths, dt);
		SPHSolver::applyForce(_sphParticles, dt);

		l++;
	}
	{
		_sphParticles->d_vs = sphV0s;
		_cloths->d_vs = clothV0s;

		_cloths->d_forces.memset(0);
		_sphParticles->d_forces.memset(0);
		_poreParticles->d_forces.memset(0);

		ParticleSampling::lerpVelocity(_cloths, _poreParticles);

		SPHSolver::applyPressureForce(_sphParticles, _poreParticles, _boundaryParticles);

		ParticleSampling::lerpForce(_cloths, _poreParticles);
		Simulation::applyForce(_cloths, dt);
		SPHSolver::applyForce(_sphParticles, dt);
	}

	printf("DF iteration %d\n", l);

	CUDA_CHECK(cudaFree(d_sumError));
}
void System::compPressureForce(REAL dt) {
	REAL h_sumError;
	REAL* d_sumError;
	CUDA_CHECK(cudaMalloc((void**)&d_sumError, sizeof(REAL)));

	uint l = 0;

	_sphParticles->d_ps.memset(0);
	Dvector<REAL> sphV0s;
	Dvector<REAL> clothV0s;
	sphV0s = _sphParticles->d_vs;
	clothV0s = _cloths->d_vs;

	while (l < 100u) {
		ParticleSampling::lerpVelocity(_cloths, _poreParticles);

		SPHSolver::compCDStiffness(_sphParticles, _poreParticles, _boundaryParticles, dt, d_sumError);

		CUDA_CHECK(cudaMemcpy(&h_sumError, d_sumError, sizeof(REAL), cudaMemcpyDeviceToHost));
		h_sumError /= (REAL)_sphParticles->_numParticles + FLT_EPSILON;

		if (h_sumError < 1.0e-4 && l >= 2)
			break;

		_cloths->d_forces.memset(0);
		_sphParticles->d_forces.memset(0);
		_poreParticles->d_forces.memset(0);

		SPHSolver::applyDFSPH(_sphParticles, _poreParticles, _boundaryParticles);

		ParticleSampling::lerpForce(_cloths, _poreParticles);

		Simulation::applyForce(_cloths, dt);
		SPHSolver::applyForce(_sphParticles, dt);

		l++;
	}
	{
		_sphParticles->d_vs = sphV0s;
		_cloths->d_vs = clothV0s;

		_cloths->d_forces.memset(0);
		_sphParticles->d_forces.memset(0);
		_poreParticles->d_forces.memset(0);

		ParticleSampling::lerpVelocity(_cloths, _poreParticles);

		SPHSolver::applyPressureForce(_sphParticles, _poreParticles, _boundaryParticles);

		ParticleSampling::lerpForce(_cloths, _poreParticles);
		Simulation::applyForce(_cloths, dt);
		SPHSolver::applyForce(_sphParticles, dt);
	}

	printf("CD iteration %d\n", l);

	CUDA_CHECK(cudaFree(d_sumError));
}

void System::compProjectiveDynamics(REAL dt) {
	REAL invdt = 1.0 / dt;
	REAL invdt2 = invdt * invdt;

	REAL underRelax = 0.9;
	REAL omg, maxError;

	uint itr = 0u;

	if (_cloths->_numNodes > 0.0) {
		_cloths->d_Bs.resize(_cloths->_numNodes);
		_cloths->d_Xs.resize(_cloths->_numNodes * 3u);
		_cloths->d_Zs.resize(_cloths->_numNodes * 3u);
		_cloths->d_newXs.resize(_cloths->_numNodes * 3u);
		_cloths->d_prevXs.resize(_cloths->_numNodes * 3u);

		Simulation::initProject(_cloths, dt, invdt2);
		while (1) {
			if (itr < 11u)			omg = 1.0;
			else if (itr == 11u)	omg = 2.0 / (2.0 - 0.9962 * 0.9962);
			else					omg = 4.0 / (4.0 - 0.9962 * 0.9962 * omg);
			itr++;

			_cloths->d_Bs.memset(0.0);
			_cloths->d_newXs = _cloths->d_Zs;
			Simulation::compErrorProject(_cloths);
			Simulation::updateXsProject(_cloths, invdt2, underRelax, omg, &maxError);
			Simulation::updateVelocity(_cloths->d_ns, _cloths->d_Xs, _cloths->d_vs, invdt);

			if (itr >= 100u)
				break;
		}
	}
}

void System::update(void) {
	REAL subDt = _dt / (REAL)_subStep;

	Simulation::initMasses(_cloths, _obstacles);

	cudaDeviceSynchronize();
	ctimer timer = CNOW;

	// Particle Sampling

	ParticleSampling::particleSampling(_obstacles, _boundaryParticles);
	ParticleSampling::particleSampling(_cloths, _poreParticles);
	ParticleSampling::compNodeWeights(_cloths, _poreParticles);

	ParticleSampling::lerpPosition(_obstacles, _boundaryParticles);
	ParticleSampling::lerpPosition(_cloths, _poreParticles);
	ParticleSampling::lerpVelocity(_obstacles, _boundaryParticles);
	ParticleSampling::lerpVelocity(_cloths, _poreParticles);
	PorousSolver::lerpPoreFactorToParticle(_cloths, _poreParticles);

	cudaDeviceSynchronize();
	printf("Particle Sampling: %f msec\n", (CNOW - timer) / 10000.0);
	timer = CNOW;

	// SPH Factor
	getParticleNeighors(false);

	cudaDeviceSynchronize();
	printf("Search Neighbors: %f msec\n", (CNOW - timer) / 10000.0);
	timer = CNOW;

	SPHSolver::compVolume(_poreParticles, _boundaryParticles);
	SPHSolver::compDensity(_sphParticles, _poreParticles, _boundaryParticles);
	SPHSolver::compDFSPHFactor(_sphParticles, _poreParticles, _boundaryParticles);
	PorousSolver::initPoreFactor(_poreParticles);
	PorousSolver::updateRelaxT(_sphParticles, subDt);

	cudaDeviceSynchronize();
	printf("SPH Factor: %f msec\n", (CNOW - timer) / 10000.0);
	timer = CNOW;

	// Compute Divergence Free
	compDivergenceFree(subDt);

	cudaDeviceSynchronize();
	printf("Compute Divergence Free: %f msec\n", (CNOW - timer) / 10000.0);
	timer = CNOW;

	// Compute Force
	{
		_cloths->d_forces.memset(0);
		_obstacles->d_forces.memset(0);
		_sphParticles->d_forces.memset(0);
		_poreParticles->d_forces.memset(0);
		_boundaryParticles->d_forces.memset(0);

		Simulation::compGravityForce(_cloths, _gravity);
		Simulation::compRotationForce(_obstacles, subDt);

		SPHSolver::compGravityForce(_sphParticles, _gravity);
		SPHSolver::compViscosityForce(_sphParticles, _poreParticles, _boundaryParticles);
		SPHSolver::compSurfaceTensionForce(_sphParticles, _poreParticles, _boundaryParticles);

		PorousSolver::compPorePressureForce(_sphParticles, _poreParticles);
		PorousSolver::compDragForce(_sphParticles, _poreParticles);
		////PorousSolver::compPoreAttractionForce(_sphParticles, _poreParticles);
		PorousSolver::compPoreAdhesionForce(_sphParticles, _poreParticles, _boundaryParticles);

		ParticleSampling::lerpForce(_cloths, _poreParticles);

		Simulation::applyForce(_cloths, subDt);
		Simulation::applyForce(_obstacles, subDt);
		SPHSolver::applyForce(_sphParticles, subDt);

		ParticleSampling::lerpVelocity(_obstacles, _boundaryParticles);
		ParticleSampling::lerpVelocity(_cloths, _poreParticles);
	}

	cudaDeviceSynchronize();
	printf("Compute Force: %f msec\n", (CNOW - timer) / 10000.0);
	timer = CNOW;

	// Compute Projective Dynamics
	compProjectiveDynamics(subDt);

	cudaDeviceSynchronize();
	printf("Compute Projective Dynamics: %f msec\n", (CNOW - timer) / 10000.0);
	timer = CNOW;

	// Compute Density Pressure
	compPressureForce(subDt);

	cudaDeviceSynchronize();
	printf("Compute Density Pressure: %f msec\n", (CNOW - timer) / 10000.0);
	timer = CNOW;

	// Compute Pore Simulation
	PorousSolver::lerpPoreFactorToObject(_cloths, _poreParticles);
	PorousSolver::compPoreVelocity(_sphParticles, _poreParticles, _boundaryParticles, _gravity);
	PorousSolver::compAbsorption(_cloths, _sphParticles, _poreParticles, subDt);
	PorousSolver::compDiffusion(_cloths, _poreParticles, subDt);
	PorousSolver::compEmission(_cloths, _sphParticles, _poreParticles, _boundaryParticles);
	PorousSolver::deleteAbsorbedParticle(_sphParticles);
	PorousSolver::generateDrippingParticle(_sphParticles, _poreParticles);

	cudaDeviceSynchronize();
	printf("Compute Pore Simulation: %f msec\n", (CNOW - timer) / 10000.0);
	timer = CNOW;

	// Compute Collision
	_cloths->d_n0s = _cloths->d_ns;
	_obstacles->d_n0s = _obstacles->d_ns;
	_sphParticles->d_x0s = _sphParticles->d_xs;
	CollisionSolver::compCollisionIteration(_ceParam, _cloths, _obstacles, _sphParticles, subDt);

	cudaDeviceSynchronize();
	printf("Compute Collision: %f msec\n", (CNOW - timer) / 10000.0);
	timer = CNOW;

	Simulation::updatePosition(_cloths->d_ns, _cloths->d_vs, subDt);
	Simulation::updatePosition(_obstacles->d_ns, _obstacles->d_vs, subDt);
	Simulation::updatePosition(_sphParticles->d_xs, _sphParticles->d_vs, subDt);
}
void System::simulation(void) {
	ctimer timer;

	printf("\n===< Frame: %d >=======================\n", _frame);
	if (_frame % 100 == 0u)
		_hash->sort(_sphParticles);

	for (uint step = 1u; step <= _subStep; step++) {
		Simulation::Damping(_cloths->d_vs, _cloths->d_isFixeds, 0.99);
		Simulation::Damping(_sphParticles->d_vs, 0.99);
		printf("===< Step %d >=======================\n", step);

		CUDA_CHECK(cudaDeviceSynchronize());
		timer = CNOW;

		update();

		CUDA_CHECK(cudaDeviceSynchronize());
		printf("Update: %f\n", (CNOW - timer) / 10000.0);
		timer = CNOW;

		_cloths->computeNormal();
		_obstacles->computeNormal();

		CUDA_CHECK(cudaDeviceSynchronize());
		printf("Compute Normals: %f\n", (CNOW - timer) / 10000.0);
		timer = CNOW;

		_cloths->copyToHost();
		_obstacles->copyToHost();
		_boundaryParticles->copyToHost();
		_sphParticles->copyToHost();
		_poreParticles->copyToHost();

		/*REAL total = 0.0;
		for (int i = 0; i < _sphParticles->_numParticles; i++) {
			total += _sphParticles->h_masses[0] * _sphParticles->h_ss[i];
		}
		vector<REAL> mfs;
		_cloths->d_mfs.copyToHost(mfs);
		for (int i = 0; i < _cloths->_numNodes; i++) {
			total += mfs[i];
			if (mfs[i] < -1.0e-4)
				printf("asdfasdf %f\n", mfs[i]);
		}
		printf("Total Mass: %f\n", total);*/

		CUDA_CHECK(cudaDeviceSynchronize());
		printf("Copy to Host: %f\n", (CNOW - timer) / 10000.0);
	}
	_frame++;
}
void System::reset(void) {
	_frame = 0u;
	_cloths->reset();
	_obstacles->reset();

	_boundaryParticles->reset();
	_poreParticles->reset();
	_sphParticles->reset();
}
void System::draw(void) {
	_cloths->draw();
	_obstacles->draw();

	//_cloths->_bvh->draw();
	//_obstacles->_bvh->draw();
	//_obstacles->_priTree->draw();

	//_boundaryParticles->draw(true);
	//_poreParticles->draw(true);
	_sphParticles->draw(false);

	//_poreParticles->drawPoreVelocity();

	//drawBoundary();
}
void System::drawBoundary(void) {
	glPushMatrix();
	glDisable(GL_LIGHTING);
	glColor3d(1, 1, 1);
	glLineWidth(3.0f);

	glBegin(GL_LINES);
	glVertex3f(_boundary._min.x, _boundary._min.y, _boundary._min.z);
	glVertex3f(_boundary._max.x, _boundary._min.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._min.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._max.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._min.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._min.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._max.y, _boundary._max.z);
	glVertex3f(_boundary._min.x, _boundary._max.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._max.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._min.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._max.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._max.y, _boundary._min.z);
	glVertex3f(_boundary._max.x, _boundary._min.y, _boundary._min.z);
	glVertex3f(_boundary._max.x, _boundary._min.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._min.y, _boundary._min.z);
	glVertex3f(_boundary._max.x, _boundary._max.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._max.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._max.y, _boundary._max.z);
	glVertex3f(_boundary._min.x, _boundary._max.y, _boundary._min.z);
	glVertex3f(_boundary._max.x, _boundary._max.y, _boundary._min.z);
	glVertex3f(_boundary._min.x, _boundary._min.y, _boundary._max.z);
	glVertex3f(_boundary._min.x, _boundary._max.y, _boundary._max.z);
	glVertex3f(_boundary._min.x, _boundary._min.y, _boundary._max.z);
	glVertex3f(_boundary._max.x, _boundary._min.y, _boundary._max.z);
	glEnd();

	glLineWidth(1.0f);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}