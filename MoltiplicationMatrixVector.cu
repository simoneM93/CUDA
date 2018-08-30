__global__ void moltiplicationMatrixVector(int dim, int* matrix, int* vector, int* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offSet = index * dim;

	for(int i = 0; i < dim; i++) 
		result[index] += matrix[i + offSet] * vector[i];
}