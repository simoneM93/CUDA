#include "Type.cu"

__global__ void matrixDivision(int dim, const T* __restrict__ matrixA, T* diagonalMatrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim) return;

	diagonalMatrix[index] = matrixA[threadIdx.x + (dim *threadIdx.x)]; 

	/*
	for(int i = 0; i < dim; i++)
		if (i == index) 
			diagonalMatrix[index] = matrixA[index * dim + index];	
	*/
}