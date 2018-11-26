#include "Librerie.cu"
#include "Type.cu"

using namespace std;

const int BlockSize = 16;
int MaxIteraton;
int esponente;
float epsilon;
float tool;

int main(int argc, char **argv)
{
	int dim = 4096;
	int NumBlock = ((dim*dim) + BlockSize - 1) / BlockSize;
	int NumThread = BlockSize;
	float gpu_runtime;

	printf("NumBlock: %i\n", NumBlock);

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
    T *hVectorResult, *dVectorResult;
    T *hDiffVectorResult, *dDiffVectorResult;
    T *hNormaResult, *dNormaResult;
    T *hNormaB, *dNormaB;
    T *dNormaReduce;
    T *dNormaReduceB;

    bool *dReduce;
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
	
	hFlag = (bool*)malloc(dim*sizeof(bool));
	error = cudaMalloc(&dFlag, dim*sizeof(bool));
	check_cuda(error, "Flag");

	error = cudaMalloc(&dReduce, dim*sizeof(bool));
	check_cuda(error, "Flag");

	//Verifico che la Matrice sia Strettamente Diagonalmente Dominante (Condizione Sufficiente per la convergenza del Metodo di Jacobi)
	NumBlock = (dim + BlockSize - 1) / BlockSize;
	cudaEventRecord(gpu_start, 0);
	diagonalyDominantMatrix<<<NumBlock, NumThread>>>(dim, dMatrix, dFlag);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    cout<<"\nCUDA runtime DiagonalyDominantMatrix: "<<gpu_runtime<<"ms\n";
	error = cudaThreadSynchronize();
	error = cudaMemcpy(hFlag, dFlag, dim*sizeof(bool), cudaMemcpyDeviceToHost);
	cout<<"\n\n----------------------------------------------------------------------------------\n\n";

	cudaEventRecord(gpu_start, 0);
	reduction<<<NumBlock, BlockSize, NumThread>>>(dFlag, dReduce, dim);
	if(NumBlock != 1)
		reduction<<<1, BlockSize, NumThread>>>(dReduce, dReduce, NumBlock);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    cout<<"\nCUDA runtime ReductionDiagonalyDominatMAtrix: "<<gpu_runtime<<"ms\n";
    error = cudaThreadSynchronize();
	error= cudaMemcpy(&isdiagonalyDominantMatrix, dReduce, sizeof(isdiagonalyDominantMatrix), cudaMemcpyDeviceToHost);

	if(!isdiagonalyDominantMatrix)
	{
		cout<<"\nLa matrice NON è Strettamente Diagonalmente Dominante, quindi non converge con il Metodo di Jacobi!\n\n";
		return 0;
	}
	
	cout<<"\nLa matrice è Strettamente Diagonalmente Dominante!\n\n";

	hDiagonalMatrix = (T*)malloc(dim*sizeof(T));
	hVector = (T*)malloc(dim*sizeof(T));
	hVectorB = (T*)malloc(dim*sizeof(T));
	hMoltiplicationResult = (T*)malloc(dim*sizeof(T));
	hSumVectorResult = (T*)malloc(dim*sizeof(T));
	hVectorResult = (T*)malloc(dim*sizeof(T));
	hDiffVectorResult = (T*)malloc(dim*sizeof(T));
	hNormaResult = (T*)malloc(dim*sizeof(T));
	hNormaB = (T*)malloc(dim*sizeof(T));

	error = cudaMalloc(&dDiagonalMatrix, dim*sizeof(T));
	check_cuda(error, "Diagonal");

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

	error = cudaMalloc(&dNormaResult, dim*sizeof(T));
	check_cuda(error, "NormaResult");

	error = cudaMalloc(&dNormaB, dim*sizeof(T));
	check_cuda(error, "NormaB");

	error = cudaMalloc(&dNormaReduce, dim*sizeof(T));
	check_cuda(error, "NormaReduce");

	error = cudaMalloc(&dNormaReduceB, dim*sizeof(T));
	check_cuda(error, "NormaReduce");

	//Divido la matrice in Matrice Diagonale e Matrice Triangolare(Superiore ed Inferiore)
	cudaEventRecord(gpu_start, 0);
	matrixDivision<<<NumBlock, NumThread>>>(dim, dMatrix, dDiagonalMatrix/*, dTriangularMatrix*/);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    cout<<"\nCUDA runtime MatrixDivision: "<<gpu_runtime<<"ms\n";
	error = cudaThreadSynchronize();
	
	//Inizializzo Il primo Vettore X al passo 0
	cudaEventRecord(gpu_start, 0);
	initVector<<<NumBlock, NumThread>>>(dim, dVector);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    cout<<"\nCUDA runtime InitVectorX: "<<gpu_runtime<<"ms\n";
	error = cudaThreadSynchronize();
	
	//Inizializzo il Vettore B
	cudaEventRecord(gpu_start, 0);
	initVectorWithIndex<<<NumBlock, NumThread>>>(dim, dVectorB);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    cout<<"\nCUDA runtime InitVectorB: "<<gpu_runtime<<"ms\n";
	error = cudaThreadSynchronize();
	
	//Calcolo la norma due del Vettore B
	cudaEventRecord(gpu_start, 0);
	normaDue<<<NumBlock, NumThread>>>(dim, dVectorB, dNormaB);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    cout<<"\nCUDA runtime NormaVectorB: "<<gpu_runtime<<"ms\n";
	error = cudaThreadSynchronize();

	T sumB = 0;

	cudaEventRecord(gpu_start, 0);
		reductionT<<<NumBlock, BlockSize, NumThread*sizeof(T)>>>(dNormaB, dNormaReduceB, dim);
		if(NumBlock != 1)
			reductionT<<<1, BlockSize, NumThread*sizeof(T)>>>(dNormaReduceB, dNormaReduceB, NumBlock);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    cout<<"\nCUDA runtime ReductionNormaVector: "<<gpu_runtime<<"ms\n";
	    error = cudaThreadSynchronize();
		error= cudaMemcpy(&sumB, dNormaReduceB, sizeof(T), cudaMemcpyDeviceToHost);	

	cout<<"SumB: "<<sumB<<"\n";

	cout<<"\n\n----------------------------------------------------------------------------------\n\n";
	
	cout<<"\nInserire Massimo Numeri Di Iterazioni Da Eseguire: ";
	cin>>MaxIteraton;
	//MaxIteraton	= 10;

	cout<<"Inserire L'Esponente Per Il Calcolo Della Epsilon [Es. 12]: ";
	cin>>esponente;

	//esponente = 3;

	int esp = esponente > 0 ? -esponente : esponente;
	cout<<"Esp: "<<esp<<"\n";

	epsilon = pow(10.0, esp);

	tool = epsilon * sumB;

	cout<<"L'Epsilon Vale: "<< epsilon<<endl;
	cout<<"La Tolleranza Vale: "<<tool<<endl;
	system("PAUSE");

	int numIteration = 0;
	while(numIteration < MaxIteraton)
	{
		cout<<"\nIterazione N°: "<<numIteration<<endl;
		
		//Moltiplico La matrice triangolare per il vettore X al passo K
		cudaEventRecord(gpu_start, 0);
		moltiplicationMatrixVector<<<NumBlock, NumThread>>>(dim, dMatrix, dVector, dMoltiplicationResult);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    cout<<"\nCUDA runtime MoltiplicationMatrixVector: "<<gpu_runtime<<"ms\n";
	    error = cudaThreadSynchronize();

		//Sommo il risultato della precedente moltiplicazione con il vettore B
		cudaEventRecord(gpu_start, 0);
		sumVectorVector<<<NumBlock, NumThread>>>(dim, dMoltiplicationResult, dVectorB, dSumVectorResult);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    cout<<"\nCUDA runtime SumVectorVector: "<<gpu_runtime<<"ms\n";
	    error = cudaThreadSynchronize();

		//Moltiplico il risultato della precedente somma per il la matrice Diagonale(Trattata come vettore)
		cudaEventRecord(gpu_start, 0);
		moltiplicationVectorVector<<<NumBlock, NumThread>>>(dim, dDiagonalMatrix, dSumVectorResult, dVectorResult);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    cout<<"\nCUDA runtime MoltiplicationVectorVector: "<<gpu_runtime<<"ms\n";
	    error = cudaThreadSynchronize();
		
		//Calcolo la differenza tra il vettore al passo k+1 e il vettore al passo k
		cudaEventRecord(gpu_start, 0);
		diffVectorVector<<<NumBlock, NumThread>>>(dim, dVectorResult, dVector, dDiffVectorResult);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    cout<<"\nCUDA runtime DiffVectorVector: "<<gpu_runtime<<"ms\n";
	    error = cudaThreadSynchronize();
		
		//Calcolo la norma due della differenza tra il vettore al passo k+1 e il vettore al passo k
		cudaEventRecord(gpu_start, 0);
		normaDue<<<NumBlock, NumThread>>>(dim, dDiffVectorResult, dNormaResult);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    cout<<"\nCUDA runtime NormaDue Di x^("<<numIteration<<") - x^("<<numIteration-1<<"): "<<gpu_runtime<<"ms\n";
	    error = cudaThreadSynchronize();
		error = cudaMemcpy(hNormaResult, dNormaResult, dim*sizeof(T), cudaMemcpyDeviceToHost);

		/*cout<<"Norma2Vector:\n";
		for(int j = 0; j < dim; j++)
			cout<<hDiffVectorResult[j]<<" ";

		cout<<"\nNorma2:\n";
		for(int j = 0; j < dim; j++)
			cout<<hNormaResult[j]<<" ";*/

		T sumIterative = 0;
		cout<<"\n";
		for(int j = 0; j < dim; j++) {
			sumIterative +=hNormaResult[j];
			//cout<<"SumIterative: "<<sumIterative<<", Iteration: "<<j<<"\n";
		}

		T sum = 0;

		cudaEventRecord(gpu_start, 0);
		reductionT<<<NumBlock, BlockSize, NumThread*sizeof(T)>>>(dNormaResult, dNormaReduce, dim);
		if(NumBlock != 1)
			reductionT<<<1, BlockSize, NumThread*sizeof(T)>>>(dNormaReduce, dNormaReduce, NumBlock);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    cout<<"\nCUDA runtime ReductionNormaVector: "<<gpu_runtime<<"ms\n";
	    error = cudaThreadSynchronize();
		error= cudaMemcpy(&sum, dNormaReduce, sizeof(T), cudaMemcpyDeviceToHost);

		cout<<"Sum: "<<sum<<"\n";
		cout<<"SumIterative: "<<sumIterative<<"\n";

		if(sum < tool )
		{	
			cout<<"\n\n----------------------------------------------------------------------------------\n";				
			cout<<"Criterio Di Arresto Rispettato!";
			cout<<"\n----------------------------------------------------------------------------------\n\n";
			return 0;
		}

		//Copio il vettore al passo k+1 nel vettore al passo k
		cudaEventRecord(gpu_start, 0);
		copyVectorToVector<<<NumBlock, NumThread>>>(dim, dVectorResult, dVector);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    cout<<"\nCUDA runtime CopyVectorToVector: "<<gpu_runtime<<"ms\n";
	    error = cudaThreadSynchronize();
	    
		numIteration++;
		cout<<"\n\n----------------------------------------------------------------------------------\n\n";
	}

	return 0;
}