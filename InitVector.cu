__global__ void initVector(int dim, int* matrix)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index > dim)
		return;

	matrix[index] = index + 1;
}