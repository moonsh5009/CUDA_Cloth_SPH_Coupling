#include "PorousSolver.cuh"

void PorousSolver::lerpPoreFactorToParticle(Cloth* cloth, PoreParticle* poreParticles) {
	lerpPoreFactorToParticle_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((ClothParam*)cloth->_param), cloth->d_fNorms(), *((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void PorousSolver::lerpPoreFactorToObject(Cloth* cloth, PoreParticle* poreParticles) {
	cloth->d_restSolidFractions.memset(0);
	cloth->d_maxFluidMasses.memset(0);
	//cloth->d_mfs.memset(0.0);
	cloth->d_ss.memset(0.0);
	lerpPoreFactorToObject_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((ClothParam*)cloth->_param), *((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}

void PorousSolver::initPoreFactor(PoreParticle* poreParticles)
{
	initPoreFactor_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void PorousSolver::updateRelaxT(SPHParticle* sphParticles, REAL dt)
{
	updateRelaxT_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param), dt);
	CUDA_CHECK(cudaPeekAtLastError());
}
void PorousSolver::compPoreVelocity(
	SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles,
	const REAL3& gravity)
{
	compPoreVelocity_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param),
		gravity);
	CUDA_CHECK(cudaPeekAtLastError());
}
void PorousSolver::compAbsorption(
	Cloth* cloths, SPHParticle* sphParticles, PoreParticle* poreParticles, const REAL dt)
{
	poreParticles->d_dms.memset(0.0);
	compAbsorption_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param), *((PoreParticleParam*)poreParticles->_param), dt);
	CUDA_CHECK(cudaPeekAtLastError());
	lerpMassToObject_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((ClothParam*)cloths->_param), *((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void PorousSolver::compEmission(
	Cloth* cloths, SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles)
{
	compIsDripping_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((PoreParticleParam*)poreParticles->_param), *((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());

	Dvector<REAL> nodeWeights;
	nodeWeights.resize(cloths->_numNodes);
	nodeWeights.memset(0);
	compEmissionNodeWeight_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((ClothParam*)cloths->_param), *((PoreParticleParam*)poreParticles->_param), nodeWeights());
	CUDA_CHECK(cudaPeekAtLastError());
	lerpEmissionMassToParticle_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((ClothParam*)cloths->_param), *((PoreParticleParam*)poreParticles->_param), nodeWeights());
	CUDA_CHECK(cudaPeekAtLastError());

	poreParticles->d_dms.memset(0.0);
	compEmissionWeight_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param), *((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
	compEmissionToSPHParticle_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param), *((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
	compEmissionToPoreParticle_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param), *((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
	lerpMassToObject_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((ClothParam*)cloths->_param), *((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void PorousSolver::deleteAbsorbedParticle(SPHParticle* sphParticles) {
	if (!sphParticles->_numParticles)
		return;
	uint oldNumParticles = sphParticles->_numParticles;
	uint newNumParticles;

	Dvector<uint> ids(oldNumParticles + 1u);

	compNewNumParticles_kernel << <divup(oldNumParticles, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param), ids());
	CUDA_CHECK(cudaPeekAtLastError());
	thrust::inclusive_scan(thrust::device_ptr<uint>(ids.begin()), thrust::device_ptr<uint>(ids.end()), ids.begin());
	CUDA_CHECK(cudaMemcpy(&newNumParticles, ids() + oldNumParticles, sizeof(uint), cudaMemcpyDeviceToHost));

	if (newNumParticles != oldNumParticles) {
		printf("delete %d\n", oldNumParticles - newNumParticles);
		Dvector<REAL> newXs(newNumParticles * 3u);
		Dvector<REAL> newVs(newNumParticles * 3u);
		Dvector<REAL> newSs(newNumParticles);
		Dvector<REAL> newRelaxTs(newNumParticles);
		Dvector<uint> newPhases(newNumParticles);

		copyOldToNew_kernel << <divup(oldNumParticles, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
			*((SPHParticleParam*)sphParticles->_param), newXs(), newVs(), newSs(), newRelaxTs(), newPhases(), ids());
		CUDA_CHECK(cudaPeekAtLastError());

		newXs.copyToHost(sphParticles->h_xs);
		newVs.copyToHost(sphParticles->h_vs);
		newSs.copyToHost(sphParticles->h_ss);
		newRelaxTs.copyToHost(sphParticles->h_relaxTs);
		newPhases.copyToHost(sphParticles->h_phases);
		newXs.clear();
		newVs.clear();
		newSs.clear();
		newRelaxTs.clear();
		newPhases.clear();
		sphParticles->updateParticle();
	}
}
void PorousSolver::generateDrippingParticle(SPHParticle* sphParticles, PoreParticle* poreParticles) {
	if (!poreParticles->_numParticles)
		return;

	uint newNumParticles = 0u;
	Dvector<uint> ids(poreParticles->_numParticles + 1u);

	compDrippingNum_kernel << <divup(poreParticles->_numParticles, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> > (
		*((PoreParticleParam*)poreParticles->_param), ids());
	CUDA_CHECK(cudaPeekAtLastError());
	thrust::inclusive_scan(thrust::device_ptr<uint>(ids.begin()), thrust::device_ptr<uint>(ids.end()), ids.begin());
	CUDA_CHECK(cudaMemcpy(&newNumParticles, ids() + poreParticles->_numParticles, sizeof(uint), cudaMemcpyDeviceToHost));

	if (newNumParticles > 0u) {
		printf("generate %d\n", newNumParticles);
		uint oldNumParticles = sphParticles->_numParticles;
		newNumParticles += oldNumParticles;
		Dvector<REAL> newXs(newNumParticles * 3u);
		Dvector<REAL> newVs(newNumParticles * 3u);
		Dvector<REAL> newSs(newNumParticles);
		Dvector<REAL> newRelaxTs(newNumParticles);
		Dvector<uint> newPhases(newNumParticles);
		CUDA_CHECK(cudaMemcpy(newXs(), sphParticles->d_xs(), oldNumParticles * 3u * sizeof(REAL), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(newVs(), sphParticles->d_vs(), oldNumParticles * 3u * sizeof(REAL), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(newSs(), sphParticles->d_ss(), oldNumParticles * sizeof(REAL), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(newRelaxTs(), sphParticles->d_relaxTs(), oldNumParticles * sizeof(REAL), cudaMemcpyDeviceToDevice));
		CUDA_CHECK(cudaMemcpy(newPhases(), sphParticles->d_phases(), oldNumParticles * sizeof(uint), cudaMemcpyDeviceToDevice));

		generateDrippingParticle_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
			*((SPHParticleParam*)sphParticles->_param), *((PoreParticleParam*)poreParticles->_param), 
			newXs(), newVs(), newSs(), newRelaxTs(), newPhases(), ids(), oldNumParticles);
		CUDA_CHECK(cudaPeekAtLastError());

		newXs.copyToHost(sphParticles->h_xs);
		newVs.copyToHost(sphParticles->h_vs);
		newSs.copyToHost(sphParticles->h_ss);
		newRelaxTs.copyToHost(sphParticles->h_relaxTs);
		newPhases.copyToHost(sphParticles->h_phases);
		newXs.clear();
		newVs.clear();
		newSs.clear();
		newRelaxTs.clear();
		newPhases.clear();
		sphParticles->updateParticle();
	}
}
void PorousSolver::compDiffusion(Cloth* cloths, PoreParticle* poreParticles, REAL dt)
{
	uint itr = 50u;
	REAL subDt = dt / (REAL)itr;

	for (uint l = 1; l <= itr; l++) {
		poreParticles->d_dms.memset(0.0);
		lerpDiffusionMassToParticle_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
			*((ClothParam*)cloths->_param), *((PoreParticleParam*)poreParticles->_param));
		CUDA_CHECK(cudaPeekAtLastError());
		compDiffusion_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
			*((ClothParam*)cloths->_param), *((PoreParticleParam*)poreParticles->_param), subDt);
		CUDA_CHECK(cudaPeekAtLastError());
		lerpMassToObject_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
			*((ClothParam*)cloths->_param), *((PoreParticleParam*)poreParticles->_param));
		CUDA_CHECK(cudaPeekAtLastError());
	}
}
void PorousSolver::compPorePressureForce(SPHParticle* sphParticles, PoreParticle* poreParticles) {
	compPorePressureForce_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void PorousSolver::compDragForce(SPHParticle* sphParticles, PoreParticle* poreParticles) {
	compDragForce_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void PorousSolver::compPoreAttractionForce(SPHParticle* sphParticles, PoreParticle* poreParticles) {
	compPoreAttractionForce_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void PorousSolver::compPoreAdhesionForce(SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles) {
	compPoreAdhesionForce_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}