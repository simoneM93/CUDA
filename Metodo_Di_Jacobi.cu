#include "Librerie.cu"
#include "Type.cu"

const int BlockSize = 8;

int main(int argc, char **argv)
{
	int dim = 64;
	int NumBlock = (dim + BlockSize - 1) / BlockSize;
	int NumThread = BlockSize;
	float gpu_runtime;
    //float tot_gpu_runtime = 0.0f;

	cudaError_t error;
	cudaEvent_t gpu_start, gpu_stop;
 
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    T *hMatrix, *dMatrix;
    T *hVector, *dVector;
    T *hVectorB, *dVectorB;
	T *hMoltiplicationResult, *dMoltiplicationResult;
    T *hSumVectorResult, *dSumVectorResult;
    T *hDiagonalMatrix, *dDiagonalMatrix;
    T *hTriangularMatrix, *dTriangularMatrix;
    T *hVectorResult, *dVectorResult;

	bool *dFlag, *hFlag;
	bool isdiagonalyDominantMatrix = true;

	hMatrix = (T*)malloc(dim*dim*sizeof(T));
	error = cudaMalloc(&dMatrix, dim*dim*sizeof(T));
	check_cuda(error, "Matrix");

	//Inizializzo la Matrice
	cudaEventRecord(gpu_start, 0);
	initMatrixSDD<<<NumBlock, NumThread>>>(dim, dMatrix);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    printf("\nCUDA runtime InitMatrix: %gms\n", gpu_runtime);
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hMatrix, dMatrix, dim*dim*sizeof(T), cudaMemcpyDeviceToHost);
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
		printf("%f ", hMatrix[i]);
	}*/
	
	for(int i = 0; i < dim; ++i)
		isdiagonalyDominantMatrix = isdiagonalyDominantMatrix && hFlag[i];

	if(!isdiagonalyDominantMatrix)
		printf("\nLa matrice NON è Strettamente Diagonalmente Dominante, quindi non converge con il Metodo di Jacobi!\n\n");
	else
	{
		printf("\nLa matrice è Strettamente Diagonalmente Dominante!\n\n");

		hDiagonalMatrix = (T*)malloc(dim*sizeof(T));
		hTriangularMatrix = (T*)malloc(dim*dim*sizeof(T));
		hVector = (T*)malloc(dim*sizeof(T));
		hVectorB = (T*)malloc(dim*sizeof(T));
		hMoltiplicationResult = (T*)malloc(dim*sizeof(T));
		hSumVectorResult = (T*)malloc(dim*sizeof(T));
		hVectorResult = (T*)malloc(dim*sizeof(T));

		error = cudaMalloc(&dDiagonalMatrix, dim*sizeof(T));
		check_cuda(error, "Diagonal");

		error = cudaMalloc(&dTriangularMatrix, dim*dim*sizeof(T));
		check_cuda(error, "Triangular");

		error = cudaMalloc(&dVector, dim*sizeof(T));
		check_cuda(error, "Vector");

		error = cudaMalloc(&dVectorB, dim*sizeof(T));
		check_cuda(error, "VectorB");

		error = cudaMalloc(&dMoltiplicationResult, dim*sizeof(T));
		check_cuda(error, "MoltiplicazionResult");

		error = cudaMalloc(&dSumVectorResult, dim*sizeof(T));
		check_cuda(error, "VectorB");

		error = cudaMalloc(&dVectorResult, dim*sizeof(T));
		check_cuda(error, "VectorResult");

		//Divido la matrice in Matrice Diagonale e Matrice Triangolare(Superiore ed Inferiore)
		cudaEventRecord(gpu_start, 0);
		matrixDivision<<<NumBlock, NumThread>>>(dim, dMatrix, dDiagonalMatrix, dTriangularMatrix);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    printf("CUDA runtime MatrixDivision: %gms\n", gpu_runtime);
		error = cudaThreadSynchronize();
		error = cudaMemcpy(hTriangularMatrix, dTriangularMatrix, dim*dim*sizeof(T), cudaMemcpyDeviceToHost);
		error = cudaMemcpy(hDiagonalMatrix, dDiagonalMatrix, dim*sizeof(T), cudaMemcpyDeviceToHost);		

		//Inizializzo Il primo Vettore X al passo 0
		cudaEventRecord(gpu_start, 0);
		initVector<<<NumBlock, NumThread>>>(dim, dVector);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    printf("CUDA runtime InitVectorX: %gms\n", gpu_runtime);
		error = cudaThreadSynchronize();
		error = cudaMemcpy(hVector, dVector, dim*sizeof(T), cudaMemcpyDeviceToHost);		

		//Inizializzo il Vettore B
		cudaEventRecord(gpu_start, 0);
		initVectorWithIndex<<<NumBlock, NumThread>>>(dim, dVectorB);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    printf("CUDA runtime InitVectorB: %gms\n", gpu_runtime);
		error = cudaThreadSynchronize();
		error = cudaMemcpy(hVectorB, dVectorB, dim*sizeof(T), cudaMemcpyDeviceToHost);	
		
		

		printf("\n\nVextorB:\n");
			for(int j = 0; j < dim; j++)
				printf("%f ", hVectorB[j]);

		printf("\n\n----------------------------------------------------------------------------------\n\n");
		
		printf("\nInserire Massimo Numeri Di Iterazioni Da Eseguire: ");
		int MaxIteraton=0;
		scanf("%d", &MaxIteraton);

		printf("\nInserire Valore Epsilon [Es. 10^(-12) = 0,000000000001]: ");
		float epsilon=0.0f;
		scanf("%f", &epsilon);

		int i = 0;
		while(i < MaxIteraton)
		{
			printf("\nIterazione N°: %d\n", i);
			
			printf("\n\nVextorX:\n");
			for(int j = 0; j < dim; j++)
				printf("%f \n", hVector[j]);

			//printf("\n\n----------------------------------------------------------------------------------\n\n");
			
			//Moltiplico La matrice triangolare per il vettore X al passo K
			cudaEventRecord(gpu_start, 0);
			moltiplicationMatrixVector<<<NumBlock, NumThread>>>(dim, dTriangularMatrix, dVector, dMoltiplicationResult);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    printf("CUDA runtime MoltiplicationMatrixVector: %gms\n", gpu_runtime);
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hMoltiplicationResult, dMoltiplicationResult, dim*sizeof(T), cudaMemcpyDeviceToHost);

			printf("\nMoltiplication Triangular Matrix With Vector X^%d:\n", i);
			for(int j = 0; j < dim; j++)
				printf("%f \n", hMoltiplicationResult[j]);
			printf("\n\n");

			//Sommo il risultato della precedente moltiplicazione con il vettore B
			cudaEventRecord(gpu_start, 0);
			sumVectorVector<<<NumBlock, NumThread>>>(dim, dMoltiplicationResult, dVectorB, dSumVectorResult);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    printf("CUDA runtime SumVectorVector: %gms\n", gpu_runtime);
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hSumVectorResult, dSumVectorResult, dim*sizeof(T), cudaMemcpyDeviceToHost);

			printf("\nSum Previous Vector With Vector B:\n");
			for(int j = 0; j < dim; j++)
				printf("%f \n", hSumVectorResult[j]);
			printf("\n\n");
						
			//Moltiplico il risultato della precedente somma per il la matrice Diagonale(Trattata come vettore)
			cudaEventRecord(gpu_start, 0);
			moltiplicationVectorVector<<<NumBlock, NumThread>>>(dim, dDiagonalMatrix, dSumVectorResult, dVectorResult);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    printf("CUDA runtime MoltiplicationVectorVector: %gms\n", gpu_runtime);
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hVectorResult, dVectorResult, dim*sizeof(T), cudaMemcpyDeviceToHost);

			printf("\nDiagonal Vector With Previous Vector (X^(%d+1)):\n", i);
			for(int j = 0; j < dim; j++)
				printf("%f \n", hVectorResult[j]);

			//if(/*Condizione di Arresto se si arriva alla convergenza*/)

			cudaEventRecord(gpu_start, 0);
			copyVectorToVector<<<NumBlock, NumThread>>>(dim, dVectorResult, dVector);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    printf("CUDA runtime CopyVectorToVector: %gms\n", gpu_runtime);
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hVector, dVector, dim*sizeof(T), cudaMemcpyDeviceToHost);

			printf("\nCopy VectorResult to Vector:\n", i);
			for(int j = 0; j < dim; j++)
				printf("%f \n", hVector[j]);


			printf("\n\n----------------------------------------------------------------------------------\n\n");
			i++;
		}

		/*printf("\nTriangular: \n");
		for(int i = 0; i < dim*dim; i++) {
			if(i % dim == 0) printf("\n");
			printf("%f ", hTriangularMatrix[i]);
		}

		printf("\n");

		printf("\nDiagonal:");
		for(int i = 0; i < dim; i++) {
			if(i % dim == 0) printf("\n");
			printf("%f ", hDiagonalMatrix[i]);
		}

		printf("\n");

		printf("\nSumVectorResult:\n");
		for(int i = 0; i < dim; i++)
			printf("%f ", hSumVectorResult[i]);

		printf("\n");
		
		printf("\nMoltiplicationVector:\n");
		for(int i = 0; i < dim; i++)
			printf("%f ", hMoltiplicationVector[i]);


		/*printf("\nVectorB:\n");
		for(int i = 0; i < dim; i++)
			printf("%f ", hVectorB[i]);

		printf("\n");
		
		printf("\nVector^%d:\n", i);
		for(int i = 0; i < dim; i++) {
			printf("%f \n", hVector[i]);
		}
		/*
		printf("\nMoltiplicationResult:\n");
		for(int i = 0; i < dim; i++) {
			printf("%f ", hMoltiplicationResult[i]);
		}*/

		return 0;
	}
}