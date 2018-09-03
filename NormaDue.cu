__global__ void normaDue(int* vettore, int *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	result[index] += vettore[index] * vettore[index];
}