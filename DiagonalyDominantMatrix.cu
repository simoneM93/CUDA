#include "AbsoluteValue.cu"
#include "Type.cu"

__global__ void diagonalyDominantMatrix(int dim, const T* __restrict__ matrix, T* diagonalMatrix, bool* flag)
{

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(index > dim - 1) 
		return;

	diagonalMatrix[index] = matrix[index * dim + index];
	
	int diagonalElements = absV(matrix[index * dim + index]);

	int sum = 0;
	// int offSet = index * dim;

	for (int i = 0; i < dim; i++) 
		if(i!=index)
			sum += absV(matrix[index + (i*dim)]);

	// for(int i = 0; i < dim; i++){
	// 	if(i != index)
	// 		sum += absV(matrix[i + offSet]);
	// }
	
	flag[index] = sum < diagonalElements ? true : false;		
}