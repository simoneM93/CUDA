#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <malloc.h>
#include "DiagonalyDominantMatrix.cu"
#include "InitMatrix.cu"

void check_cuda(cudaError_t err, const char *msg) {
 if (err != cudaSuccess) {
  fprintf(stderr, "%s -- %d: %s\n", msg,
   err, cudaGetErrorString(err));
  exit(0);
 }
}

int main (int argc, char **argv)
{
	cudaError_t error;

	int dim = 4096;
	int *hMatrix, *dMatrix;
	bool *dFlag, *hFlag;
	bool isdiagonalyDominantMatrix;

	hMatrix = (int*)malloc(dim*dim*sizeof(int));
	
	hFlag = (bool*)malloc(dim*sizeof(bool));

	error = cudaMalloc(&dMatrix, dim*dim*sizeof(int));
	check_cuda(error, "Error");
	error = cudaMalloc(&dFlag, dim*sizeof(bool));
	check_cuda(error, "error");

	initMatrix<<<dim, dim>>>(dim*dim, dMatrix);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hMatrix, dMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaThreadSynchronize();

	printf("\n\n");
	
	/*for(int i = 0; i < dim*dim; ++i) 
		printf("%d ", hMatrix[i]);*/

	printf("\n\n");

	diagonalyDominantMatrix<<<dim, dim>>>(dim, dMatrix, dFlag);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hMatrix, dMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaMemcpy(hFlag, dFlag, dim*sizeof(bool), cudaMemcpyDeviceToHost);

	for(int i = 0; i < dim; ++i)
		isdiagonalyDominantMatrix &= hFlag[i];

	if(!isdiagonalyDominantMatrix)
		printf("La matrice NON è Strettamente Diagonalmente Dominante, quindi non converge con il metodo di Jacobi!\n");
	else
		printf("La matrice è Strettamente Diagonalmente Dominante!\n");
	
	printf("\nIsDiagonaly = %d\n", isdiagonalyDominantMatrix);
}