#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <malloc.h>
#include "DiagonalyDominantMatrix.cu"
#include "InitMatrix.cu"
#include "InitVector.cu"
#include "SumMatrix.cu"
#include "TransposedMatrix.cu"
#include "MatrixDivision.cu"

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

	int dim = 4;
	int *hMatrix, *dMatrix, *hSumMatrix, *dSumMatrix, *hTraspMatrix, *dTraspMatrix, *hDiagonalMatrix, *dDiagonalMatrix, *hTriangularMatrix, *dTriangularMatrix,
	*hVector, *dVector;
	bool *dFlag, *hFlag;
	bool isdiagonalyDominantMatrix;

	hMatrix = (int*)malloc(dim*dim*sizeof(int));
	hSumMatrix = (int*)malloc(dim*dim*sizeof(int));
	hFlag = (bool*)malloc(dim*sizeof(bool));
	hTraspMatrix = (int*)malloc(dim*dim*sizeof(int));
	hDiagonalMatrix = (int*)malloc(dim*dim*sizeof(int));
	hTriangularMatrix = (int*)malloc(dim*dim*sizeof(int));

	hVector = (int*)malloc(dim*sizeof(int));

	error = cudaMalloc(&dMatrix, dim*dim*sizeof(int));
	check_cuda(error, "Error");

	error = cudaMalloc(&dDiagonalMatrix, dim*dim*sizeof(int));
	check_cuda(error, "Error");

	error = cudaMalloc(&dTriangularMatrix, dim*dim*sizeof(int));
	check_cuda(error, "Error");

	error = cudaMalloc(&dTraspMatrix, dim*dim*sizeof(int));
	check_cuda(error, "Error");
	
	error = cudaMalloc(&dFlag, dim*sizeof(bool));
	check_cuda(error, "error");
	
	error = cudaMalloc(&dSumMatrix, dim*dim*sizeof(int));
	check_cuda(error, "SumMatrix");

	error = cudaMalloc(&dVector, dim*sizeof(int));
	check_cuda(error, "Vector");


	initMatrix<<<1, dim>>>(dim, dMatrix);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hMatrix, dMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaThreadSynchronize();

	sumMatrix<<<1, dim>>>(dim, dMatrix, dMatrix, dSumMatrix);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hSumMatrix, dSumMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaThreadSynchronize();	

	diagonalyDominantMatrix<<<dim, dim>>>(dim, dMatrix, dFlag);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hMatrix, dMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaMemcpy(hFlag, dFlag, dim*sizeof(bool), cudaMemcpyDeviceToHost);

	for(int i = 0; i < dim; ++i)
		isdiagonalyDominantMatrix &= hFlag[i];

	!isdiagonalyDominantMatrix ? printf("\nLa matrice NON è Strettamente Diagonalmente Dominante, quindi non converge con il metodo di Jacobi!\n") : printf("\nLa matrice è Strettamente Diagonalmente Dominante!\n");

	transposedMatrix<<<1, dim>>>(dim, dMatrix, dTraspMatrix);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hTraspMatrix, dTraspMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaThreadSynchronize();

	matrixDivision<<<dim, 1>>>(dim, dMatrix, dDiagonalMatrix, dTriangularMatrix);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hTriangularMatrix, dTriangularMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hDiagonalMatrix, dDiagonalMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaThreadSynchronize();

	initVector<<<1, dim>>>(dim, dVector);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hVector, dVector, dim*sizeof(int), cudaMemcpyDeviceToHost);
}