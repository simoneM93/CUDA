__global__ void matrixDivision(int dim, int* matrixA, int* diagonalMatrix, int*  triangularMatrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offSet = index * dim;

	for(int i = 0; i < dim; ++i)	
		if(i != index)
			triangularMatrix[i + offSet] = matrixA[i + offSet];
		else
		{
			diagonalMatrix[index] = matrixA[index * dim + index];
			triangularMatrix[i + offSet] = 0;
		}	
}