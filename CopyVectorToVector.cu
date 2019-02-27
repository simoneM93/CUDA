#include "Type.cu"

__global__ void copyVectorToVector(int dim, T*  vector1, T* vector2)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim)
		return;

	vector2[index] = vector1[index];
	vector1[index] = 0;
}