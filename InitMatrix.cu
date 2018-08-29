__global__ void initMatrix(int dim, int* matrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int dimMatrix = dim * dim;

	if(index > dimMatrix)
		return;

	matrix[index] = index;
}