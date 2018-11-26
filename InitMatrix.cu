#include "Type.cu"

__global__ void initMatrix(int dim, T* matrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int dimMatrix = dim * dim;

	if(index > dimMatrix)
		return;

	matrix[index] = index + 1;
}

__global__ void initMatrixSDD(int dim, T* matrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim*dim) return;

	index == ((index/dim)*(dim+1)) ? matrix[index]=15.0 : matrix[index]=0.1;
}