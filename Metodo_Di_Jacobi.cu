#include "Librerie.cu"
#include "Type.cu"

using namespace std;

const int BlockSize = 8;
int MaxIteraton;
float epsilon;
int esponente;

int main(int argc, char **argv)
{
	int dim = 64;
	int NumBlock = (dim + BlockSize - 1) / BlockSize;
	int NumThread = BlockSize;
	float gpu_runtime;

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
    T *hDiffVectorResult, *dDiffVectorResult;
    T *hNormaResult, *dNormaResult;
    //T *hNormaResult2, *dNormaResult2;

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
    cout<<"\nCUDA runtime InitMatrix: "<<gpu_runtime<<"ms\n";
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
    cout<<"\nCUDA runtime DiagonalyDominantMatrix: "<<gpu_runtime<<"ms\n";
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hFlag, dFlag, dim*sizeof(bool), cudaMemcpyDeviceToHost);
	cout<<"\n\n----------------------------------------------------------------------------------\n\n";
	
	for(int i = 0; i < dim; ++i)
		isdiagonalyDominantMatrix = isdiagonalyDominantMatrix && hFlag[i];

	if(!isdiagonalyDominantMatrix)
		cout<<"\nLa matrice NON è Strettamente Diagonalmente Dominante, quindi non converge con il Metodo di Jacobi!\n\n";
	else
	{
		cout<<"\nLa matrice è Strettamente Diagonalmente Dominante!\n\n";

		hDiagonalMatrix = (T*)malloc(dim*sizeof(T));
		hTriangularMatrix = (T*)malloc(dim*dim*sizeof(T));
		hVector = (T*)malloc(dim*sizeof(T));
		hVectorB = (T*)malloc(dim*sizeof(T));
		hMoltiplicationResult = (T*)malloc(dim*sizeof(T));
		hSumVectorResult = (T*)malloc(dim*sizeof(T));
		hVectorResult = (T*)malloc(dim*sizeof(T));
		hDiffVectorResult = (T*)malloc(dim*sizeof(T));
		hNormaResult = (T*)malloc(1*sizeof(T));
		//hNormaResult2 = (T*)malloc(1*sizeof(T));

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

		error = cudaMalloc(&dDiffVectorResult, dim*sizeof(T));
		check_cuda(error, "DiffVectorResult");

		error = cudaMalloc(&dNormaResult, 1*sizeof(T));
		check_cuda(error, "NormaResult");

		/*error = cudaMalloc(&dNormaResult2, 1*sizeof(T));
		check_cuda(error, "NormaResult2");*/

		//Divido la matrice in Matrice Diagonale e Matrice Triangolare(Superiore ed Inferiore)
		cudaEventRecord(gpu_start, 0);
		matrixDivision<<<NumBlock, NumThread>>>(dim, dMatrix, dDiagonalMatrix, dTriangularMatrix);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    cout<<"\nCUDA runtime MatrixDivision: "<<gpu_runtime<<"ms\n";
		error = cudaThreadSynchronize();
		error = cudaMemcpy(hTriangularMatrix, dTriangularMatrix, dim*dim*sizeof(T), cudaMemcpyDeviceToHost);
		error = cudaMemcpy(hDiagonalMatrix, dDiagonalMatrix, dim*sizeof(T), cudaMemcpyDeviceToHost);		

		//Inizializzo Il primo Vettore X al passo 0
		cudaEventRecord(gpu_start, 0);
		initVector<<<NumBlock, NumThread>>>(dim, dVector);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    cout<<"\nCUDA runtime InitVectorX: "<<gpu_runtime<<"ms\n";
		error = cudaThreadSynchronize();
		error = cudaMemcpy(hVector, dVector, dim*sizeof(T), cudaMemcpyDeviceToHost);		

		//Inizializzo il Vettore B
		cudaEventRecord(gpu_start, 0);
		initVectorWithIndex<<<NumBlock, NumThread>>>(dim, dVectorB);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    cout<<"\nCUDA runtime InitVectorB: "<<gpu_runtime<<"ms\n";
		error = cudaThreadSynchronize();
		error = cudaMemcpy(hVectorB, dVectorB, dim*sizeof(T), cudaMemcpyDeviceToHost);		

		cout<<"\n\n----------------------------------------------------------------------------------\n\n";
		
		cout<<"\nInserire Massimo Numeri Di Iterazioni Da Eseguire: ";
		cin>>MaxIteraton;

		cout<<"Inserire L'Esponente Per Il Calcolo Della Epsilon [Es. -12]: ";
		cin>>esponente;

		epsilon = pow(10, esponente);

		cout<<"L'Epsilon Vale: "<< epsilon<<endl;
		system("PAUSE");

		int i = 0;
		while(i < MaxIteraton)
		{
			cout<<"\nIterazione N°: "<<i<<endl;
			
			//Moltiplico La matrice triangolare per il vettore X al passo K
			cudaEventRecord(gpu_start, 0);
			moltiplicationMatrixVector<<<NumBlock, NumThread>>>(dim, dTriangularMatrix, dVector, dMoltiplicationResult);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    cout<<"\nCUDA runtime MoltiplicationMatrixVector: "<<gpu_runtime<<"ms\n";
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hMoltiplicationResult, dMoltiplicationResult, dim*sizeof(T), cudaMemcpyDeviceToHost);

			//Sommo il risultato della precedente moltiplicazione con il vettore B
			cudaEventRecord(gpu_start, 0);
			sumVectorVector<<<NumBlock, NumThread>>>(dim, dMoltiplicationResult, dVectorB, dSumVectorResult);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    cout<<"\nCUDA runtime SumVectorVector: "<<gpu_runtime<<"ms\n";
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hSumVectorResult, dSumVectorResult, dim*sizeof(T), cudaMemcpyDeviceToHost);
						
			//Moltiplico il risultato della precedente somma per il la matrice Diagonale(Trattata come vettore)
			cudaEventRecord(gpu_start, 0);
			moltiplicationVectorVector<<<NumBlock, NumThread>>>(dim, dDiagonalMatrix, dSumVectorResult, dVectorResult);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    cout<<"\nCUDA runtime MoltiplicationVectorVector: "<<gpu_runtime<<"ms\n";
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hVectorResult, dVectorResult, dim*sizeof(T), cudaMemcpyDeviceToHost);
			
			cudaEventRecord(gpu_start, 0);
			diffVectorVector<<<NumBlock, NumThread>>>(dim, dVectorResult, dVector, dDiffVectorResult);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    cout<<"\nCUDA runtime DiffVectorVector: "<<gpu_runtime<<"ms\n";
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hDiffVectorResult, dDiffVectorResult, dim*sizeof(T), cudaMemcpyDeviceToHost);

			cudaEventRecord(gpu_start, 0);
			normaDue<<<NumBlock, NumThread>>>(dim, dDiffVectorResult, dNormaResult);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    cout<<"\nCUDA runtime NormaDue Di x^("<<i<<") - x^("<<i-1<<"): "<<gpu_runtime<<"ms\n";
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hNormaResult, dNormaResult, 1*sizeof(T), cudaMemcpyDeviceToHost);

			/*cudaEventRecord(gpu_start, 0);
			normaDue<<<NumBlock, NumThread>>>(dim, dVectorResult, dNormaResult2);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    cout<<"\nCUDA runtime NormaDue Di x^("<<i<<"):"<<gpu_runtime<<"ms\n";
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hNormaResult2, dNormaResult2, 1*sizeof(T), cudaMemcpyDeviceToHost);
			cout<<"Norma Differenza: "<<hNormaResult[0]<<endl<<"Norma vettore: "<<hNormaResult2[0]<<endl;
			
			T divisione = (hNormaResult[0]/hNormaResult2[0]);*/

			cout<<"NormaVector: "<<hNormaResult[0]<<endl;
			cout<<"L'Epsilon Vale: "<< epsilon<<endl;

			if(hNormaResult[0] < epsilon)
			{	
				cout<<"\n\n----------------------------------------------------------------------------------\n\n";				
				cout<<"Criterio Di Arresto Rispettato!";
				cout<<"\n\n----------------------------------------------------------------------------------\n\n";
				return 0;
			}

			cudaEventRecord(gpu_start, 0);
			copyVectorToVector<<<NumBlock, NumThread>>>(dim, dVectorResult, dVector);
			cudaEventRecord(gpu_stop, 0);
		    cudaEventSynchronize(gpu_stop);
		    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
		    cout<<"\nCUDA runtime CopyVectorToVector: "<<gpu_runtime<<"ms\n";
		    error = cudaThreadSynchronize();
			error = cudaMemcpy(hVector, dVector, dim*sizeof(T), cudaMemcpyDeviceToHost);

			i++;
			cout<<"\n\n----------------------------------------------------------------------------------\n\n";
		}

		return 0;
	}
}