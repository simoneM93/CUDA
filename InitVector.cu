#include "Type.cu"

__global__ void initVector(int dim, T* matrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim)
		return;

	matrix[index] = 0;
}

__global__ void initVectorWithIndex(int dim, T* matrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim)
		return;

	matrix[index] = index + 1;
}