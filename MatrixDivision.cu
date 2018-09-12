#include "Type.cu"

__global__ void matrixDivision(int dim, T* matrixA, T* diagonalMatrix, T*  triangularMatrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offSet = index * dim;

	for(int i = 0; i < dim; i++)
		i != index ? triangularMatrix[i + offSet] = matrixA[i + offSet] : triangularMatrix[i + offSet] = 0, diagonalMatrix[index] = matrixA[index * dim + index];	
}