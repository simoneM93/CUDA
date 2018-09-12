#include "Type.cu"

__global__ void sumMatrix(int dim, T* matrixA, T* matrixB, T* resultMatrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offSet = index * dim;
	int dimMatrix = dim * dim;

	for(int i = 0; i < dimMatrix; ++i) 
		resultMatrix[index + i] = matrixA[i + offSet] + matrixB[i + offSet];
}