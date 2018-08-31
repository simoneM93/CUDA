__global__ void normaDue(int* vettore, int result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	result += vettore[index] * vettore[index];
}