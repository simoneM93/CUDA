__global__ void moltiplicationMatrixVector(int dim, int* matrix, int* vector, int* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int offSet = index * dim;

	for(int i = 0; i < dim; i++) 
		result[index] += matrix[i + offSet] * vector[i];
}

__global__ void moltiplicationVectorVector(int dim, int* vector1, int* vector2, int* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim)
		return;

	result[index] = vector1[index] * vector2[index];
}