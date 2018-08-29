__global__ void transposedMatrix(int dim, int* matrix, int* resultMatrix)
{
	int index = blockIdx.x * dim + threadIdx.x;

	for(int i = 0; i < dim; ++i)
		resultMatrix[index * dim + i] = matrix[ i * dim + index];
}