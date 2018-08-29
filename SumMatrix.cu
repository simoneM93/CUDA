__global__ void sumMatrix(int dim, int* matrixA, int* matrixB, int* resultMatrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int offSet = index * dim;

	for(int i = 0; i < dim; ++i) 
	{
		
	}
}