#include "Type.cu"

__global__ void diffVectorVector(int dim, T* vector1, T* vector2, T* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	result[index] = vector1[index] - vector2[index];
}