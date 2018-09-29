#include "Type.cu"

__global__ void copyVectorToVector(int dim, const T* __restrict__ vector1, T* vector2)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim)
		return;

	vector2[index] = vector1[index];
}