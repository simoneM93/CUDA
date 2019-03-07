#include "Type.cu"

__global__ void JacobiOperation(const int dim, const T* __restrict__ matrix, const T* __restrict__ diagonalyMatrix, const T* __restrict__ vectorX, const T* __restrict__ vectorB, T* result)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // int offSet = index * dim;

	if(index > dim)
        return;
        
    for(int i = 0; i < dim; i++) {
        //if(i != index) 
        int diagonal = (i != index);
        result[index] += (matrix[index + (i * dim)] * vectorX[i]) * diagonal;
    }

	// for(int i = 0; i < dim; i++) 
	// 	if(i != index) 
    //         result[index] += matrix[i + offSet] * vectorX[i];
    
    result[index] = (result[index] + vectorB[index]) * diagonalyMatrix[index];
}