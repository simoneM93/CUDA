#include "Librerie.cu"

const int BlockSize = 8;

int main(int argc, char **argv)
{
	int dim = 4;
	int NumBlock = (dim + BlockSize - 1) / BlockSize;
	int NumThread = BlockSize;
	float gpu_runtime;
    //float tot_gpu_runtime = 0.0f;

	cudaError_t error;
	cudaEvent_t gpu_start, gpu_stop;
 
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    //int *hNorma, *dNorma;
    type *hMatrix, *dMatrix;
    type *hVector, *dVector;
    type *hVectorB, *dVectorB;
	type *hMoltiplicationResult, *dMoltiplicationResult;
    //int *hSumMatrix, *dSumMatrix;    	
	//int *hTraspMatrix, *dTraspMatrix;
    type *hSumVectorResult, *dSumVectorResult;
    type *hDiagonalMatrix, *dDiagonalMatrix;
    type *hTriangularMatrix, *dTriangularMatrix;
    type *hMoltiplicationVector, *dMoltiplicationVector;

	bool *dFlag, *hFlag;
	bool isdiagonalyDominantMatrix = true;

	hMatrix = (type*)malloc(dim*dim*sizeof(type));
	error = cudaMalloc(&dMatrix, dim*dim*sizeof(type));
	check_cuda(error, "Matrix");

	//Inizializzo la Matrice
	cudaEventRecord(gpu_start, 0);
	initMatrixSDD<<<NumBlock, NumThread>>>(dim, dMatrix);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    printf("\nCUDA runtime InitMatrix: %gms\n", gpu_runtime);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hMatrix, dMatrix, dim*dim*sizeof(type), cudaMemcpyDeviceToHost);
	error = cudaThreadSynchronize();

	hFlag = (bool*)malloc(dim*sizeof(bool));
	error = cudaMalloc(&dFlag, dim*sizeof(bool));
	check_cuda(error, "Flag");

	//Verifico che la Matrice sia Strettamente Diagonalmente Dominante (Condizione Sufficiente per la convergenza del Metodo di Jacobi)
	cudaEventRecord(gpu_start, 0);
	diagonalyDominantMatrix<<<NumBlock, NumThread>>>(dim, dMatrix, dFlag);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    printf("CUDA runtime diagonalyDominantMatrix: %gms\n", gpu_runtime);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hFlag, dFlag, dim*sizeof(bool), cudaMemcpyDeviceToHost);
	printf("\n\n----------------------------------------------------------------------------------\n\n");

	/*printf("\nMatrix: ");
	for(int i = 0; i < dim*dim; i++) {
		if(i % dim == 0) printf("\n");
		printf("%d ", hMatrix[i]);
	}*/

	for(int i = 0; i < dim; ++i)
		isdiagonalyDominantMatrix = isdiagonalyDominantMatrix && hFlag[i];

	if(!isdiagonalyDominantMatrix)
		printf("\nLa matrice NON è Strettamente Diagonalmente Dominante, quindi non converge con il Metodo di Jacobi!\n\n");
	else
	{
		printf("\nLa matrice è Strettamente Diagonalmente Dominante!\n\n");

		hDiagonalMatrix = (type*)malloc(dim*sizeof(type));
		hTriangularMatrix = (type*)malloc(dim*dim*sizeof(type));
		hVector = (type*)malloc(dim*sizeof(type));
		hVectorB = (type*)malloc(dim*sizeof(type));
		hMoltiplicationResult = (type*)malloc(dim*sizeof(type));
		hSumVectorResult = (type*)malloc(dim*sizeof(type));
		hMoltiplicationVector = (type*)malloc(dim*sizeof(type));

		error = cudaMalloc(&dDiagonalMatrix, dim*sizeof(type));
		check_cuda(error, "Diagonal");

		error = cudaMalloc(&dTriangularMatrix, dim*dim*sizeof(type));
		check_cuda(error, "Triangular");

		error = cudaMalloc(&dVector, dim*sizeof(type));
		check_cuda(error, "Vector");

		error = cudaMalloc(&dVectorB, dim*sizeof(type));
		check_cuda(error, "VectorB");

		error = cudaMalloc(&dMoltiplicationResult, dim*sizeof(type));
		check_cuda(error, "MoltiplicazionResult");

		error = cudaMalloc(&dSumVectorResult, dim*sizeof(type));
		check_cuda(error, "VectorB");

		error = cudaMalloc(&dMoltiplicationVector, dim*sizeof(type));
		check_cuda(error, "MoltiplicationVector");


		//Divido la matrice in Matrice Diagonale e Matrice Triangolare(Superiore ed Inferiore)
		cudaEventRecord(gpu_start, 0);
		matrixDivision<<<NumBlock, NumThread>>>(dim, dMatrix, dDiagonalMatrix, dTriangularMatrix);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    printf("CUDA runtime MatrixDivision: %gms\n", gpu_runtime);
		error = cudaThreadSynchronize();
		error = cudaMemcpy(hTriangularMatrix, dTriangularMatrix, dim*dim*sizeof(type), cudaMemcpyDeviceToHost);
		error = cudaMemcpy(hDiagonalMatrix, dDiagonalMatrix, dim*sizeof(type), cudaMemcpyDeviceToHost);		

		//Inizializzo Il primo Vettore X al passo 0
		cudaEventRecord(gpu_start, 0);
		initVector<<<NumBlock, NumThread>>>(dim, dVector);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    printf("CUDA runtime InitVector: %gms\n", gpu_runtime);
		error = cudaThreadSynchronize();
		error = cudaMemcpy(hVector, dVector, dim*sizeof(type), cudaMemcpyDeviceToHost);		

		//Inizializzo il Vettore B
		cudaEventRecord(gpu_start, 0);
		initVectorWithIndex<<<NumBlock, NumThread>>>(dim, dVectorB);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    printf("CUDA runtime InitVector: %gms\n", gpu_runtime);
		error = cudaThreadSynchronize();
		error = cudaMemcpy(hVectorB, dVectorB, dim*sizeof(type), cudaMemcpyDeviceToHost);	
		
		int MaxIteraton;
		scanf("%d", &MaxIteraton);

		printf("\n\nVextorB:\n");
			for(int j = 0; j < dim; j++)
				printf("%d ", hVectorB[j]);

		printf("\n\n----------------------------------------------------------------------------------\n\n");
		
		int i = 0;
		while(i < MaxIteraton)
		{
			printf("\nIterazione N°: %d\n", i);
			printf("\n\nVextorX:\n");
			for(int j = 0; j < dim; j++)
				printf("%d ", hVector[j]);

			printf("\n\n----------------------------------------------------------------------------------\n\n");
			//Moltiplico La matrice triangolare per il vettore X al passo K
			cudaEventRecord(gpu_start, 0);
			moltiplicationMatrixVector<<<NumBlock, NumThread>>>(dim, dTriangularMatrix, dVector, dMoltiplicationResult);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    printf("CUDA runtime MoltiplicationMatrixVector: %gms\n", gpu_runtime);
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hMoltiplicationResult, dMoltiplicationResult, dim*sizeof(type), cudaMemcpyDeviceToHost);

			printf("\nMoltiplicationMatrixVector:\n");
			for(int j = 0; j < dim; j++)
				printf("%d ", hMoltiplicationResult[j]);
			printf("\n\n");

			//Sommo il risultato della precedente moltiplicazione con il vettore B
			cudaEventRecord(gpu_start, 0);
			sumVectorVector<<<NumBlock, NumThread>>>(dim, dMoltiplicationResult, dVectorB, dSumVectorResult);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    printf("CUDA runtime SumVectorVector: %gms\n", gpu_runtime);
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hSumVectorResult, dSumVectorResult, dim*sizeof(type), cudaMemcpyDeviceToHost);

			printf("\nSumVectorVector:\n");
			for(int j = 0; j < dim; j++)
				printf("%d ", hSumVectorResult[j]);
			printf("\n\n");
						
			//Moltiplico il risultato della precedente somma per il la matrice Diagonale(Trattata come vettore)
			cudaEventRecord(gpu_start, 0);
			moltiplicationVectorVector<<<NumBlock, NumThread>>>(dim, dDiagonalMatrix, dSumVectorResult, dVector);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    printf("CUDA runtime MoltiplicationVectorVector: %gms\n", gpu_runtime);
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hVector, dVector, dim*sizeof(type), cudaMemcpyDeviceToHost);

			printf("\nMoltiplicationVector:\n");
			for(int j = 0; j < dim; j++)
				printf("%d ", hVector[j]);

			printf("\n\n----------------------------------------------------------------------------------\n\n");
			i++;
		}

		/*printf("\nTriangular: \n");
		for(int i = 0; i < dim*dim; i++) {
			if(i % dim == 0) printf("\n");
			printf("%d ", hTriangularMatrix[i]);
		}

		printf("\n");

		printf("\nDiagonal:");
		for(int i = 0; i < dim; i++) {
			if(i % dim == 0) printf("\n");
			printf("%d ", hDiagonalMatrix[i]);
		}

		printf("\n");

		printf("\nSumVectorResult:\n");
		for(int i = 0; i < dim; i++)
			printf("%d ", hSumVectorResult[i]);

		printf("\n");
		
		printf("\nMoltiplicationVector:\n");
		for(int i = 0; i < dim; i++)
			printf("%d ", hMoltiplicationVector[i]);


		/*printf("\nVectorB:\n");
		for(int i = 0; i < dim; i++)
			printf("%d ", hVectorB[i]);

		printf("\n");

		printf("\nVector0:\n");
		for(int i = 0; i < dim; i++) {
			printf("%d ", hVector[i]);
		}

		printf("\nMoltiplicationResult:\n");
		for(int i = 0; i < dim; i++) {
			printf("%d ", hMoltiplicationResult[i]);
		}*/
	}
}