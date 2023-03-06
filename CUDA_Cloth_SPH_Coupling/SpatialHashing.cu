#include "SpatialHashing.cuh"

void SpatialHashing::sort(SPHParticle* particles) {
#ifdef HASH_TIMER
	cudaDeviceSynchronize();
	ctimer timer = CNOW;
#endif

	uint numParticles = particles->_numParticles;
	if (!numParticles)
		return;

	initParticle(numParticles);

	Dvector<REAL> oldXs(numParticles * 3u);
	Dvector<REAL> oldVs(numParticles * 3u);
	Dvector<REAL> oldSs(numParticles);
	Dvector<REAL> oldRelaxTs(numParticles);
	Dvector<uint> oldPhase(numParticles);

	CUDA_CHECK(cudaMemcpy(oldXs(), particles->d_xs(), numParticles * sizeof(REAL3), cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(oldVs(), particles->d_vs(), numParticles * sizeof(REAL3), cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(oldSs(), particles->d_ss(), numParticles * sizeof(REAL), cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(oldRelaxTs(), particles->d_relaxTs(), numParticles * sizeof(REAL), cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaMemcpy(oldPhase(), particles->d_phases(), numParticles * sizeof(uint), cudaMemcpyDeviceToDevice));

	initHashZindex_kernel << <divup(numParticles, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> >
		((REAL3*)particles->d_xs(), _param);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::sort_by_key(thrust::device_ptr<uint>((uint*)_keys()),
		thrust::device_ptr<uint>(((uint*)_keys()) + _numParticles),
		thrust::device_ptr<uint2>((uint2*)_ids()));

	ParticleZSort_kernel << <divup(numParticles, BLOCKSIZE), BLOCKSIZE >> >
		(*((SPHParticleParam*)particles->_param), oldXs(), oldVs(), oldSs(), oldRelaxTs(), oldPhase(), _ids());
	CUDA_CHECK(cudaPeekAtLastError());

	particles->d_xs.copyToHost(particles->h_xs);
	particles->d_vs.copyToHost(particles->h_vs);
	particles->d_ss.copyToHost(particles->h_ss);
	particles->d_relaxTs.copyToHost(particles->h_relaxTs);
	particles->d_phases.copyToHost(particles->h_phases);

#ifdef HASH_TIMER
	cudaDeviceSynchronize();
	printf("SpatialHashing::sort: %f msec\n", (CNOW - timer) / 10000.0);
#endif
}
void SpatialHashing::insert(SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles) {
#ifdef HASH_TIMER
	cudaDeviceSynchronize();
	ctimer timer = CNOW;
#endif

	uint numParticles = sphParticles->_numParticles + poreParticles->_numParticles + boundaryParticles->_numParticles;
	if (!numParticles)
		return;

	initParticle(numParticles);
	
	initHash_kernel << <divup(numParticles, MAX_BLOCKSIZE), MAX_BLOCKSIZE >> >
		(*sphParticles->_param, *poreParticles->_param, *boundaryParticles->_param, _param);
	CUDA_CHECK(cudaPeekAtLastError());

	thrust::sort_by_key(thrust::device_ptr<uint>((uint*)_keys()),
		thrust::device_ptr<uint>(((uint*)_keys()) + _numParticles),
		thrust::device_ptr<uint2>((uint2*)_ids()));

	reorderHash_kernel << <divup(numParticles, MAX_BLOCKSIZE), MAX_BLOCKSIZE, (MAX_BLOCKSIZE + 1) * sizeof(uint) >> >
		(_param);
	CUDA_CHECK(cudaPeekAtLastError());

#ifdef HASH_TIMER
	cudaDeviceSynchronize();
	printf("SpatialHashing::insert: %f msec\n", (CNOW - timer) / 10000.0);
#endif
}
void SpatialHashing::getNeighbors(SPHParticle* sphParticles, PoreParticle* poreParticles, BoundaryParticle* boundaryParticles) {
#ifdef HASH_TIMER
	cudaDeviceSynchronize();
	ctimer timer = CNOW;
#endif
	uint numParticles = sphParticles->_numParticles + poreParticles->_numParticles + boundaryParticles->_numParticles;
	if (!numParticles)
		return;

	compNumNeighbors_kernel << <divup(numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param), 
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param), _param);
	CUDA_CHECK(cudaPeekAtLastError());

	uint numSPHNeis = 0u;
	uint numPoreNeis = 0u;
	uint numBoundaryNeis = 0u;

	if (sphParticles->_numParticles) {
		thrust::inclusive_scan(thrust::device_ptr<uint>(sphParticles->d_ineis.begin()),
			thrust::device_ptr<uint>(sphParticles->d_ineis.end()), sphParticles->d_ineis.begin());
		CUDA_CHECK(cudaMemcpy(&numSPHNeis, sphParticles->d_ineis() + sphParticles->_numParticles, sizeof(uint), cudaMemcpyDeviceToHost));
	}
	if (poreParticles->_numParticles) {
		thrust::inclusive_scan(thrust::device_ptr<uint>(poreParticles->d_ineis.begin()),
			thrust::device_ptr<uint>(poreParticles->d_ineis.end()), poreParticles->d_ineis.begin());
		CUDA_CHECK(cudaMemcpy(&numPoreNeis, poreParticles->d_ineis() + poreParticles->_numParticles, sizeof(uint), cudaMemcpyDeviceToHost));
	}
	if (boundaryParticles->_numParticles) {
		thrust::inclusive_scan(thrust::device_ptr<uint>(boundaryParticles->d_ineis.begin()),
			thrust::device_ptr<uint>(boundaryParticles->d_ineis.end()), boundaryParticles->d_ineis.begin());
		CUDA_CHECK(cudaMemcpy(&numBoundaryNeis, boundaryParticles->d_ineis() + boundaryParticles->_numParticles, sizeof(uint), cudaMemcpyDeviceToHost));
	}

	sphParticles->d_neis.resize(numSPHNeis);
	poreParticles->d_neis.resize(numPoreNeis);
	boundaryParticles->d_neis.resize(numBoundaryNeis);
	sphParticles->_param->_neis = sphParticles->d_neis._list;
	poreParticles->_param->_neis = poreParticles->d_neis._list;
	boundaryParticles->_param->_neis = boundaryParticles->d_neis._list;

	getNeighbors_kernel << <divup(numParticles, BLOCKSIZE), BLOCKSIZE >> > (
		*((SPHParticleParam*)sphParticles->_param),
		*((PoreParticleParam*)poreParticles->_param),
		*((BoundaryParticleParam*)boundaryParticles->_param), _param);
	CUDA_CHECK(cudaPeekAtLastError());

#ifdef HASH_TIMER
	cudaDeviceSynchronize();
	printf("SpatialHashing::getNeighbors: %f msec\n", (CNOW - timer) / 10000.0);
#endif
}