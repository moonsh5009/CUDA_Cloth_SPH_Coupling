#include "SPHSolver.cuh"

void SPHSolver::compVolume(
	PoreParticle* poreParticles, BoundaryParticle* boundaryParticles)
{
	compBoundaryParticleVolume_kernel << <divup(boundaryParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((BoundaryParticleParam*)boundaryParticles->_param), *((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
	compPoreParticleVolume_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((PoreParticleParam*)poreParticles->_param), *((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void SPHSolver::compDensity(
	SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles)
{
	compDensity_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param), 
		*((PoreParticleParam*)poreParticles->_param), 
		*((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void SPHSolver::compDFSPHFactor(
	SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles)
{
	compDFSPHFactor_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}

void SPHSolver::compCDStiffness(
	SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles,
	REAL dt, REAL* d_sumError)
{
	CUDA_CHECK(cudaMemset(d_sumError, 0, sizeof(REAL)));

	compCDStiffness_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param),
		dt, d_sumError);
	CUDA_CHECK(cudaPeekAtLastError());
}
void SPHSolver::compDFStiffness(
	SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles,
	REAL dt, REAL* d_sumError)
{
	CUDA_CHECK(cudaMemset(d_sumError, 0, sizeof(REAL)));

	compDFStiffness_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param),
		dt, d_sumError);
	CUDA_CHECK(cudaPeekAtLastError());
}
void SPHSolver::applyDFSPH(
	SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles)
{
	applyDFSPH_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void SPHSolver::applyPressureForce(
	SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles)
{
	applyPressureForce_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}

void SPHSolver::compGravityForce(SPHParticle* sphParticles, const REAL3& gravity)
{
	compGravityForce_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param), gravity);
	CUDA_CHECK(cudaPeekAtLastError());
}

void SPHSolver::compViscosityForce(
	SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles)
{
	compViscosityForce_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void SPHSolver::compSurfaceTensionForce(
	SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles)
{
#if 1
	compColorNormal_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
	compSurfaceTensionForce_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
#else
	compColorField_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
	compColorGradient_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
	compSurfaceTensionForce_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
#endif
}

void SPHSolver::applyForce(SPHParticle* sphParticles, const REAL dt)
{
	applyForce_kernel << <divup(sphParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param), dt);
	CUDA_CHECK(cudaPeekAtLastError());
}