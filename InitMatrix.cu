__global__ void initMatrix(int dim, int* matrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim)
		return;

	matrix[index] = index;
}