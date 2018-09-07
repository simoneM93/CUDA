#include "AbsoluteValue.cu"

__global__ void diagonalyDominantMatrix(int dim, int* matrix, bool* flag)
{

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(index > dim - 1) 
		return;
	
	int diagonalElements = absV(matrix[index * dim + index]);

	int sum = 0;
	int offSet = index * dim;

	for(int i = 0; i < dim; ++i){
		if(i != index)
			sum += absV(matrix[i + offSet]);
	}
	
	flag[index] = sum < diagonalElements ? true : false;		
}