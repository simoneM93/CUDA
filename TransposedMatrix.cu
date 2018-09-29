#include "Type.cu"

__global__ void transposedMatrix(int dim, const T* __restrict__ matrix, T* resultMatrix)
{
	int index = blockIdx.x * dim + threadIdx.x;

	for(int i = 0; i < dim; ++i)
		resultMatrix[index * dim + i] = matrix[ i * dim + index];
}