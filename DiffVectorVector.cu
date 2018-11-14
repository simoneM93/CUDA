#include "Type.cu"

__global__ void diffVectorVector(int dim, const T* __restrict__ vector1, const T* __restrict__ vector2, T* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim)
		return;

	result[index] = vector1[index] - vector2[index];
	printf("vecto1[index]:%g, vecto2[index]:%g, result[index]:%g, index:%i\n", vector1[index], vector2[index], result[index], index);
}