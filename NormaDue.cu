#include "Type.cu"

__global__ void normaDue(int dim, T* vettore, T *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int i = 0;
	if (index > dim)
		return;

	result[i] += vettore[index] * vettore[index];
}