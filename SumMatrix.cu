__global__ void sumMatrix(int dim, int* matrixA, int* matrixB, int* resultMatrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offSet = index * dim;
	int dimMatrix = dim * dim;

	for(int i = 0; i < dimMatrix; ++i) 
		resultMatrix[index + i] = matrixA[i + offSet] + matrixB[i + offSet];
}