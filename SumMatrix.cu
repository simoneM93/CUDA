#include "Type.cu"

__global__ void sumMatrix(int dim, const T* __restrict__ matrixA, const T* __restrict__ matrixB, T* resultMatrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offSet = index * dim;
	int dimMatrix = dim * dim;

	for(int i = 0; i < dimMatrix; ++i) 
		resultMatrix[index + i] = matrixA[i + offSet] + matrixB[i + offSet];
}