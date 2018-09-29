#include "Type.cu"

__global__ void moltiplicationMatrixVector(int dim, const T* __restrict__ matrix, const T* __restrict__ vector, T* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offSet = index * dim;

	for(int i = 0; i < dim; i++) 
		result[index] += matrix[i + offSet] * vector[i];
}

__global__ void moltiplicationVectorVector(int dim, const T* __restrict__ vector1, const T* __restrict__ vector2, T* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim)
		return;

	result[index] = vector1[index] * vector2[index];
}