#include "Type.cu"

__global__ void normaDue(int* vettore, T *result)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	result[index] += vettore[index] * vettore[index];
}