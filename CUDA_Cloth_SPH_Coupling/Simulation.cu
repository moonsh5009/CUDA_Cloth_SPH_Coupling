#include "Simulation.cuh"

void Simulation::initMasses(Cloth* cloth, Obstacle* obstacle) {
	Dvector<REAL> ms;
	ms = cloth->h_ms;

	initClothMasses_kernel << <divup(cloth->_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		*((ClothParam*)cloth->_param), ms(), cloth->d_isFixeds());
	CUDA_CHECK(cudaPeekAtLastError());

	ms = obstacle->h_ms;
	initMasses_kernel << <divup(obstacle->_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		*obstacle->_param, ms(), obstacle->d_isFixeds());
	CUDA_CHECK(cudaPeekAtLastError());
}
void Simulation::compGravityForce(MeshObject* obj, const REAL3& gravity) {
	compGravityForce_kernel << <divup(obj->_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		obj->d_forces(), obj->d_ms(), gravity, obj->_numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void Simulation::compRotationForce(Obstacle* obj, const REAL dt) {
	compRotationForce_kernel << <divup(obj->_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		obj->d_ns(), obj->d_vs(), obj->d_forces(), obj->d_ms(), obj->d_nodePhases(),
		obj->d_pivots(), obj->d_degrees(), 1.0 / dt, obj->_numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}

void Simulation::applyForce(MeshObject* obj, const REAL dt) {
	applyForce_kernel << <divup(obj->_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		obj->d_vs(), obj->d_forces(), obj->d_invMs(), obj->d_isFixeds(), dt, obj->_numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void Simulation::updateVelocity(Dvector<REAL>& n0s, Dvector<REAL>& n1s, Dvector<REAL>& vs, const REAL invdt) {
	uint numNodes = n0s.size() / 3u;
	updateVelocity_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		n0s(), n1s(), vs(), invdt, numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void Simulation::updatePosition(Dvector<REAL>& ns, Dvector<REAL>& vs, const REAL dt) {
	uint numNodes = ns.size() / 3u;
	updatePosition_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		ns(), vs(), dt, numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}

void Simulation::initProject(Cloth* obj, const REAL dt, const REAL invdt2) {
	initProject_kernel << <divup(obj->_numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		obj->d_ns(), obj->d_vs(), obj->d_ms(), obj->d_Zs(), obj->d_Xs(), 
		dt, 1.0 / (dt * dt), obj->_numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void Simulation::compErrorProject(Cloth* obj) {
	uint numSpring = obj->d_ses.arraySize() >> 1u;
	compErrorProject_kernel << <divup(numSpring, BLOCKSIZE), BLOCKSIZE >> > (
		obj->d_Xs(), obj->d_newXs(), obj->d_Bs(),
		obj->d_ses._array(), obj->d_stretchWs(), obj->d_stretchRLs(),
		obj->d_nodePhases(), numSpring);
	CUDA_CHECK(cudaPeekAtLastError());

	numSpring = obj->d_bes.arraySize() >> 1u;
	compErrorProject_kernel << <divup(numSpring, BLOCKSIZE), BLOCKSIZE >> > (
		obj->d_Xs(), obj->d_newXs(), obj->d_Bs(),
		obj->d_bes._array(), obj->d_bendingWs(), obj->d_bendingRLs(),
		obj->d_nodePhases(), numSpring);
	CUDA_CHECK(cudaPeekAtLastError());
}
void Simulation::updateXsProject(
	Cloth* obj, const REAL invdt2, 
	const REAL underRelax, const REAL omega, REAL* maxError) 
{
	REAL* d_maxError;
	CUDA_CHECK(cudaMalloc((void**)&d_maxError, sizeof(REAL)));
	CUDA_CHECK(cudaMemset(d_maxError, 0, sizeof(REAL)));

	updateXsProject_kernel << <divup(obj->_numNodes, BLOCKSIZE), BLOCKSIZE, BLOCKSIZE * sizeof(REAL) >> > (
		obj->d_ms(), obj->d_isFixeds(), obj->d_Bs(), obj->d_Xs(), obj->d_prevXs(), obj->d_newXs(),
		underRelax, omega, invdt2, obj->_numNodes, d_maxError);
	CUDA_CHECK(cudaPeekAtLastError());

	CUDA_CHECK(cudaMemcpy(maxError, d_maxError, sizeof(REAL), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_maxError));
}

void  Simulation::Damping(Dvector<REAL>& vs, REAL w) {
	uint numNodes = vs.size() / 3u;
	Damping_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		vs(), w, numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}
void  Simulation::Damping(Dvector<REAL>& vs, Dvector<uchar>& isFixeds, REAL w) {
	uint numNodes = vs.size() / 3u;
	Damping_kernel << <divup(numNodes, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		vs(), isFixeds(), w, numNodes);
	CUDA_CHECK(cudaPeekAtLastError());
}