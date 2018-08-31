__global__ void sumVectorVector(int dim, int* vector1, int* vector2, int* result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	result[index] = vector1[index] + vector2[index];
}