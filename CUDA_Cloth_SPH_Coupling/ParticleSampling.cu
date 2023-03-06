#include "ParticleSampling.cuh"

void ParticleSampling::compParticleNum(
	MeshObject* obj, BoundaryParticle* boundaryParticles, 
	Dvector<uint>& prevInds, Dvector<uint>& currInds, Dvector<bool>& isGenerateds, bool& isApplied) 
{
	bool* d_isApplied;
	CUDA_CHECK(cudaMalloc((void**)&d_isApplied, sizeof(bool)));
	CUDA_CHECK(cudaMemset(d_isApplied, 0, sizeof(bool)));

	compSamplingNum_kernel << <divup(obj->_numFaces, BLOCKSIZE), BLOCKSIZE >> > (
		*obj->_param, obj->_RTri->_param, *((BoundaryParticleParam*)boundaryParticles->_param),
		boundaryParticles->d_shortEs(), boundaryParticles->d_sampNums(),
		prevInds(), currInds(), isGenerateds(), d_isApplied);
	CUDA_CHECK(cudaPeekAtLastError());
	CUDA_CHECK(cudaMemcpy(&isApplied, d_isApplied, sizeof(bool), cudaMemcpyDeviceToHost));

	CUDA_CHECK(cudaFree(d_isApplied));
}
void ParticleSampling::particleSampling(MeshObject* obj, BoundaryParticle* boundaryParticles) {
	/*cudaDeviceSynchronize();
	ctimer timer = CNOW;*/

	Dvector<uint> prevInds(obj->_numFaces + 1u);
	Dvector<uint> currInds(obj->_numFaces + 1u);
	Dvector<bool> isGenerateds(obj->_numFaces);
	bool isApplied;

	compParticleNum(obj, boundaryParticles, prevInds, currInds, isGenerateds, isApplied);
	if (isApplied) {
		Dvector<REAL> prevXs;
		Dvector<REAL> prevWs;
		Dvector<uint> prevInos;
		prevXs = boundaryParticles->d_xs;
		prevWs = boundaryParticles->d_ws;
		prevInos = boundaryParticles->d_inos;

		uint prevSampNum, currSampNum;
		thrust::inclusive_scan(thrust::device_ptr<uint>(prevInds.begin()), thrust::device_ptr<uint>(prevInds.end()), prevInds.begin());
		thrust::inclusive_scan(thrust::device_ptr<uint>(currInds.begin()), thrust::device_ptr<uint>(currInds.end()), currInds.begin());
		CUDA_CHECK(cudaMemcpy(&prevSampNum, prevInds() + obj->_numFaces, sizeof(uint), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(&currSampNum, currInds() + obj->_numFaces, sizeof(uint), cudaMemcpyDeviceToHost));
		boundaryParticles->resize(currSampNum);
		boundaryParticles->setParam();

		generateBoundaryParticle_kernel << <divup(obj->_numFaces, BLOCKSIZE), BLOCKSIZE >> > (
			*obj->_param, obj->_RTri->_param,
			*((BoundaryParticleParam*)boundaryParticles->_param), prevXs(), prevWs(), prevInos(),
			boundaryParticles->d_shortEs(), prevInds(), currInds(), isGenerateds());
		CUDA_CHECK(cudaPeekAtLastError());

		setBarycentric_kernel << <divup(boundaryParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
			*obj->_param, *((BoundaryParticleParam*)boundaryParticles->_param));
		CUDA_CHECK(cudaPeekAtLastError());

		boundaryParticles->d_phases.copyToHost(boundaryParticles->h_phases);
	}

	/*cudaDeviceSynchronize();
	printf("ParticleSampling::particleSampling: %f msec\n", (CNOW - timer) / 10000.0);*/
}

void ParticleSampling::compNodeWeights(Cloth* cloth, PoreParticle* poreParticles) {
	poreParticles->d_nodeWeights.resize(cloth->_numNodes);
	poreParticles->d_nodeWeights.memset(0);
	((PoreParticleParam*)poreParticles->_param)->_nodeWeights = poreParticles->d_nodeWeights._list;
	compNodeWeights_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((ClothParam*)cloth->_param), *((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void ParticleSampling::lerpPosition(MeshObject* obj, BoundaryParticle* boundaryParticles) {
	lerpPosition_kernel << <divup(boundaryParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*obj->_param, *((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void ParticleSampling::lerpVelocity(MeshObject* obj, BoundaryParticle* boundaryParticles) {
	lerpVelocity_kernel << <divup(boundaryParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*obj->_param, *((BoundaryParticleParam*)boundaryParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}
void ParticleSampling::lerpForce(Cloth* cloth, PoreParticle* poreParticles) {
	lerpForce_kernel << <divup(poreParticles->_numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*cloth->_param, *((PoreParticleParam*)poreParticles->_param));
	CUDA_CHECK(cudaPeekAtLastError());
}