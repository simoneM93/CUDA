__global__ void transposedMatrix(int dim, int* matrix, int* resultMatrix)
{
	int index = blockIdx.x * dim + threadIdx.x;
	int indexy = blockIdx.y * dim + threadIdx.y;
	int width = gridDim.x * dim;

	for(int i = 0; i < dim; ++i)
		resultMatrix[index * width + (indexy + i)] = matrix[(indexy + i) * width + index];
}