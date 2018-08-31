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
#include "MoltiplicationMatrixVector.cu"
#include "SumVectorVector.cu"
#include "NormaDue.cu"
#include "Check_Cuda.cu"



int main (int argc, char **argv)
{
	cudaError_t error;
	cudaEvent_t gpu_start, gpu_stop;
    float gpu_runtime;
    float tot_gpu_runtime = 0.0f;

    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

	int dim = 1024;
	int *hMatrix, *dMatrix, *hSumMatrix, *dSumMatrix, *hTraspMatrix, *dTraspMatrix, *hDiagonalMatrix, *dDiagonalMatrix, *hTriangularMatrix, *dTriangularMatrix,
	*hVector, *dVector, *hResult, *dResult, *hVectorResult, *dVectorResult;
	bool *dFlag, *hFlag;
	bool isdiagonalyDominantMatrix;

	hMatrix = (int*)malloc(dim*dim*sizeof(int));
	hSumMatrix = (int*)malloc(dim*dim*sizeof(int));
	hFlag = (bool*)malloc(dim*sizeof(bool));
	hTraspMatrix = (int*)malloc(dim*dim*sizeof(int));
	hDiagonalMatrix = (int*)malloc(dim*sizeof(int));
	hTriangularMatrix = (int*)malloc(dim*dim*sizeof(int));
	hResult = (int*)malloc(dim*sizeof(int));
	hVectorResult = (int*)malloc(dim*sizeof(int));
	hVector = (int*)malloc(dim*sizeof(int));

	error = cudaMalloc(&dMatrix, dim*dim*sizeof(int));
	check_cuda(error, "Error");

	error = cudaMalloc(&dDiagonalMatrix, dim*sizeof(int));
	check_cuda(error, "Error");

	error = cudaMalloc(&dVectorResult, dim*sizeof(int));
	check_cuda(error, "Error");

	error = cudaMalloc(&dResult, dim*sizeof(int));
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

	cudaEventRecord(gpu_start, 0);
	initMatrix<<<dim, dim>>>(dim, dMatrix);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    printf("CUDA runtime InitMatrix: %gms\n", gpu_runtime);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hMatrix, dMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaThreadSynchronize();

	tot_gpu_runtime += gpu_runtime;

	cudaEventRecord(gpu_start, 0);
	sumMatrix<<<1, dim>>>(dim, dMatrix, dMatrix, dSumMatrix);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    printf("CUDA runtime SumMatrix: %gms\n", gpu_runtime);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hSumMatrix, dSumMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaThreadSynchronize();

	tot_gpu_runtime += gpu_runtime;	

	cudaEventRecord(gpu_start, 0);
	diagonalyDominantMatrix<<<dim, dim>>>(dim, dMatrix, dFlag);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    printf("CUDA runtime diagonalyDominantMatrix: %gms\n", gpu_runtime);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hMatrix, dMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaMemcpy(hFlag, dFlag, dim*sizeof(bool), cudaMemcpyDeviceToHost);

	tot_gpu_runtime += gpu_runtime;

	for(int i = 0; i < dim; ++i)
		isdiagonalyDominantMatrix &= hFlag[i];

	!isdiagonalyDominantMatrix ? printf("\nLa matrice NON è Strettamente Diagonalmente Dominante, quindi non converge con il metodo di Jacobi!\n") : printf("\nLa matrice è Strettamente Diagonalmente Dominante!\n");

	cudaEventRecord(gpu_start, 0);
	transposedMatrix<<<1, dim>>>(dim, dMatrix, dTraspMatrix);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    printf("CUDA runtime TrasposedMatrix: %gms\n", gpu_runtime);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hTraspMatrix, dTraspMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaThreadSynchronize();

	tot_gpu_runtime += gpu_runtime;

	cudaEventRecord(gpu_start, 0);
	matrixDivision<<<dim, 1>>>(dim, dMatrix, dDiagonalMatrix, dTriangularMatrix);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    printf("CUDA runtime MatrixDivision: %gms\n", gpu_runtime);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hTriangularMatrix, dTriangularMatrix, dim*dim*sizeof(int), cudaMemcpyDeviceToHost);
	
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hDiagonalMatrix, dDiagonalMatrix, dim*sizeof(int), cudaMemcpyDeviceToHost);
	error = cudaThreadSynchronize();

	tot_gpu_runtime += gpu_runtime;

	cudaEventRecord(gpu_start, 0);
	initVector<<<1, dim>>>(dim, dVector);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    printf("CUDA runtime InitVector: %gms\n", gpu_runtime);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hVector, dVector, dim*sizeof(int), cudaMemcpyDeviceToHost);

	tot_gpu_runtime += gpu_runtime;

	cudaEventRecord(gpu_start, 0);
	moltiplicationMatrixVector<<<dim, 1>>>(dim, dMatrix, dDiagonalMatrix, dResult);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    printf("CUDA runtime MoltiplicationMatrixVector: %gms\n", gpu_runtime);
    error = cudaThreadSynchronize();
	error = cudaMemcpy(hResult, dResult, dim*sizeof(int), cudaMemcpyDeviceToHost);

	tot_gpu_runtime += gpu_runtime;

	cudaEventRecord(gpu_start, 0);
	sumVectorVector<<<dim, 1>>>(dim, dDiagonalMatrix, dDiagonalMatrix, dVectorResult);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    printf("CUDA runtime SumVectorVector: %gms\n", gpu_runtime);
    error = cudaThreadSynchronize();
	error = cudaMemcpy(hVectorResult, dVectorResult, dim*sizeof(int), cudaMemcpyDeviceToHost);

	tot_gpu_runtime += gpu_runtime;

	printf("CUDA tot runtime: %gms\n", tot_gpu_runtime);
}