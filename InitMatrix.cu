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

	int offSet = index * dim;

	for(int i = 0; i < dim; ++i)	
		i != index ? matrix[i + offSet] = 0.1 : matrix[i + offSet] = dim+1;
}