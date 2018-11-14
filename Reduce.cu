#include "Type.cu"


__global__ void reduction(const bool *vec, bool *vec2, int numels)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;

	extern __shared__ bool shmem[];

	shmem[threadIdx.x] = 0;

	while(i < numels) {
		shmem[threadIdx.x] += vec[i];
		i += blockDim.x*gridDim.x;
	}

	for (int c = blockDim.x/2; c > 0; c/=2) {
		__syncthreads();
		if (threadIdx.x < c)
			shmem[threadIdx.x] += shmem[threadIdx.x + c];
	}

	if (threadIdx.x == 0)
		vec2[blockIdx.x] = shmem[threadIdx.x];
}

__global__ void reductionT(const T *vec, T *vec2, int numels)
{
	int index = threadIdx.x + blockDim.x*blockIdx.x;

	extern __shared__ T shmemT[];

	shmemT[threadIdx.x] = 0;

	while(index < numels) {
		shmemT[threadIdx.x] += vec[index];
		if(threadIdx.x == 0) 
		printf("vec[index]:%g, shmem:%g\n", vec[index], shmemT[threadIdx.x]);
		index += blockDim.x*gridDim.x;
		
	}

	for (int c = blockDim.x/2; c > 0; c/=2) {
		__syncthreads();
		if (threadIdx.x < c)
			shmemT[threadIdx.x] += shmemT[threadIdx.x + c];
	}

	if (threadIdx.x == 0){
		vec2[blockIdx.x] = shmemT[threadIdx.x];
	

		printf("Vec: %g, block:%i, numels:%i\n", vec2[blockIdx.x], blockIdx.x, numels);
	}
}