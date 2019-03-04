#include "Type.cu"


__global__ void reduction(const bool *vec, bool *vec2, int numels)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	extern __shared__ bool shmem[];

	shmem[threadIdx.x] = 1;

	while(i < numels) {
		shmem[threadIdx.x] &= vec[i];
		i += blockDim.x*gridDim.x;
	}

	for (int c = blockDim.x/2; c > 0; c/=2) {
		__syncthreads();
		if (threadIdx.x < c)
			shmem[threadIdx.x] &= shmem[threadIdx.x + c];
	}

	if (threadIdx.x == 0)
		vec2[blockIdx.x] = shmem[threadIdx.x];
}

__global__ void reductionQuadratoT(const T* __restrict__ vec, T *vec2, int numels)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;

	extern __shared__ T shmemT[];

	shmemT[threadIdx.x] = 0;

	T tmp = vec[index];

	while(index < numels) {
		shmemT[threadIdx.x] += tmp * tmp;
		index += blockDim.x*gridDim.x;
		
	}

	for (int c = blockDim.x/2; c > 0; c/=2) {
		__syncthreads();
		if (threadIdx.x < c)
			shmemT[threadIdx.x] += shmemT[threadIdx.x + c];
	}

	if (threadIdx.x == 0){
		vec2[blockIdx.x] = shmemT[threadIdx.x];
	}
}

__global__ void reductionDiffT(const T* __restrict__ vec, const T* __restrict__ vec1, T *vec2, int numels)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;

	extern __shared__ T shmemT[];

	shmemT[threadIdx.x] = 0;
	
	T diffVector = vec[index] - vec1[index];

	while(index < numels) {
		shmemT[threadIdx.x] += diffVector * diffVector;
		index += blockDim.x*gridDim.x;
		
	}

	for (int c = blockDim.x/2; c > 0; c/=2) {
		__syncthreads();
		if (threadIdx.x < c)
			shmemT[threadIdx.x] += shmemT[threadIdx.x + c];
	}

	if (threadIdx.x == 0){
		vec2[blockIdx.x] = shmemT[threadIdx.x];
	}
}

__global__ void reductionT(const T* __restrict__ vec, T *vec2, int numels)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;

	extern __shared__ T shmemT[];

	shmemT[threadIdx.x] = 0;

	while(index < numels) {
		shmemT[threadIdx.x] += vec[index];
		index += blockDim.x*gridDim.x;
		
	}

	for (int c = blockDim.x/2; c > 0; c/=2) {
		__syncthreads();
		if (threadIdx.x < c)
			shmemT[threadIdx.x] += shmemT[threadIdx.x + c];
	}

	if (threadIdx.x == 0){
		vec2[blockIdx.x] = shmemT[threadIdx.x];
	}
}