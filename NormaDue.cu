#include "Type.cu"

__global__ void normaDue(int dim, const T* __restrict__ vettore, T *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index > dim)
		return;

	result[index] = vettore[index] * vettore[index];
}