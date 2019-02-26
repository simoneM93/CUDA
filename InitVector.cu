#include "Type.cu"

__global__ void initVector(int dim, T* vectorX, T* vectorB)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim)
		return;

	vectorX[index] = 0;
	vectorB[index] = index + 1;
}

__global__ void initVectorWithIndex(int dim, T* matrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim)
		return;

	matrix[index] = index + 1;
}