#include "Librerie.cu"
#include "Type.cu"

using namespace std;

const int BlockSize = 1024;
int MaxIteraton = 10;
int esponente = 3;
int dim = 4096;
int numThread = BlockSize;
int numBlock;
float epsilon;
float tool;
float gpu_runtime;
cudaError_t error;
cudaEvent_t gpu_start, gpu_stop;


int main()
{
    numBlock = ((dim * dim) + BlockSize -1) / BlockSize;
    
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    cout<<"-----------------------------------------------------------------"<<endl;
    cout<<"NumBlock: "<<numBlock<<endl;

    int esp = esponente > 0 ? -esponente : esponente;
    epsilon = pow(10.0, esp);

    cout<<"esp: "<<esp<<endl<<"epsilon (10.0^"<<esp<<"): "<<epsilon<<endl;
    cout<<"-----------------------------------------------------------------"<<endl;

#//Dichiarazione e Allocazione variabili
    T *deviceMatrix;
    T *deviceDiagonalMatrix;
    T *deviceVectorX;
    T *deviceVectorB;
    T *deviceNormaB;
    T *deviceNormaReduceB;
    T *deviceMoltiplicationResult;
    T *deviceSumVectorResult;
    T *deviceVectorResult;
    T *deviceDiffVectorResult;
    T *deviceNormaResult;
    T *deviceNormaReduce;

    bool *deviceFlag;
    bool *deviceReduce;
    bool isdiagonalyDominantMatrix;

    error = cudaMalloc(&deviceMatrix, dim * dim * sizeof(T));
    check_cuda(error, "deviceMatrix");
    error = cudaMalloc(&deviceDiagonalMatrix, dim * sizeof(T));
    check_cuda(error, "deviceDiagonalMatrix");
    error = cudaMalloc(&deviceVectorX, dim * sizeof(T));
    check_cuda(error, "deviceVectorX");
    error = cudaMalloc(&deviceVectorB, dim * sizeof(T));
    check_cuda(error, "deviceVectorB");
    error = cudaMalloc(&deviceNormaB, dim * sizeof(T));
    check_cuda(error, "deviceNormaB");
    error = cudaMalloc(&deviceNormaReduceB, dim * sizeof(T));
    check_cuda(error, "deviceNormaReduceB");
    error = cudaMalloc(&deviceMoltiplicationResult, dim * sizeof(T));
    check_cuda(error, "deviceMoltiplicationResult");
    error = cudaMalloc(&deviceSumVectorResult, dim * sizeof(T));
    check_cuda(error, "deviceSumVectorResult");
    error = cudaMalloc(&deviceVectorResult, dim * sizeof(T));
    check_cuda(error, "deviceVectorResult");
    error = cudaMalloc(&deviceDiffVectorResult, dim * sizeof(T));
    check_cuda(error, "deviceDiffVectorResult");
    error = cudaMalloc(&deviceNormaResult, dim * sizeof(T));
    check_cuda(error, "deviceNormaResult");
    error = cudaMalloc(&deviceNormaReduce, dim * sizeof(T));
	check_cuda(error, "deviceNormaReduce");
    
#//Inizializzo la matrice iniziale
    
	check_cuda(error, "Matrix");
    cudaEventRecord(gpu_start, 0);
	initMatrixSDD<<<numBlock, numThread>>>(dim, deviceMatrix);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    cout<<"\nCUDA runtime InitMatrix: "<<gpu_runtime<<"ms\n";
    error = cudaThreadSynchronize();

	error = cudaMalloc(&deviceFlag, dim * sizeof(bool));
    check_cuda(error, "Flag");

    error = cudaMalloc(&deviceReduce, dim*sizeof(bool));
    check_cuda(error, "Flag");
#//Verifico che la Matrice sia Strettamente Diagonalmente Dominante (Condizione Sufficiente per la convergenza del Metodo di Jacobi)
    numBlock = (dim + BlockSize - 1) / BlockSize;
    cudaEventRecord(gpu_start, 0);
	diagonalyDominantMatrix<<<numBlock, numThread>>>(dim, deviceMatrix, deviceDiagonalMatrix, deviceFlag);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    cout<<"\nCUDA runtime DiagonalyDominantMatrix: "<<gpu_runtime<<"ms\n";
	error = cudaThreadSynchronize();

    cudaEventRecord(gpu_start, 0);
	reduction<<<numBlock, BlockSize, numThread>>>(deviceFlag, deviceReduce, dim);
	if(numBlock != 1)
		reduction<<<1, BlockSize, numThread>>>(deviceReduce, deviceReduce, numBlock);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    cout<<"\nCUDA runtime ReductionDiagonalyDominatMAtrix: "<<gpu_runtime<<"ms\n";
    error = cudaThreadSynchronize();
    error= cudaMemcpy(&isdiagonalyDominantMatrix, deviceReduce, sizeof(isdiagonalyDominantMatrix), cudaMemcpyDeviceToHost);
    
    if(!isdiagonalyDominantMatrix)
	{
		cout<<"\nLa matrice NON è Strettamente Diagonalmente Dominante, quindi non converge con il Metodo di Jacobi!\n\n";
		return 0;
    }

    cout<<"\n-----------------------------------------------------------------"<<endl;    
	cout<<"La matrice è Strettamente Diagonalmente Dominante!"<<endl;
    cout<<"-----------------------------------------------------------------"<<endl;
    
    numBlock = ((dim * dim) + BlockSize -1) / BlockSize;

#//Inizializzo il vettore x al passo 0 e il vettore B
    cudaEventRecord(gpu_start, 0);
	initVector<<<numBlock, numThread>>>(dim, deviceVectorX, deviceVectorB);
	cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    cout<<"\nCUDA runtime InitVectorX: "<<gpu_runtime<<"ms\n";
    error = cudaThreadSynchronize();
    
#//Calcolo la norma due del Vettore B
    T normaB = 0.0;

	cudaEventRecord(gpu_start, 0);
    reductionQuadratoT<<<numBlock, BlockSize, numThread * sizeof(T)>>>(deviceNormaB, deviceNormaReduceB, dim);
    if(numBlock != 1)
        reductionT<<<1, BlockSize, numThread * sizeof(T)>>>(deviceNormaReduceB, deviceNormaReduceB, numBlock);
    cudaEventRecord(gpu_stop, 0);
    cudaEventSynchronize(gpu_stop);
    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
    cout<<"\nCUDA runtime ReductionNormaVector: "<<gpu_runtime<<"ms\n";
    error = cudaThreadSynchronize();
    error= cudaMemcpy(&normaB, deviceNormaReduceB, sizeof(T), cudaMemcpyDeviceToHost);	
    
    cout<<"\n-----------------------------------------------------------------"<<endl;
    cout<<"NormaB: "<<normaB<<endl;
    cout<<"-----------------------------------------------------------------"<<endl;

    tool = epsilon * normaB;

    cout<<"\n-----------------------------------------------------------------"<<endl;
    cout<<"Tool: "<<tool<<endl;
    cout<<"-----------------------------------------------------------------"<<endl;

    numBlock = ((dim * dim) + BlockSize -1) / BlockSize;
#//Inizio delle iterazioni
    for(int numIteration = 0; numIteration < MaxIteraton; numIteration++)
	{
		cout<<"\nIterazione N°: "<<numIteration<<endl;
    #//Calcolo il vettore X al passo K+1
        cudaEventRecord(gpu_start, 0);
        JacobiOperation<<<numBlock, numThread>>>(dim, deviceMatrix, deviceDiagonalMatrix, deviceVectorX, deviceVectorB, deviceVectorResult);
        cudaEventRecord(gpu_stop, 0);
        cudaEventSynchronize(gpu_stop);
        cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
        cout<<"\nCUDA runtime JacobiOperation: "<<gpu_runtime<<"ms\n";
        error = cudaThreadSynchronize();
        
    #//Calcolo la norma due della differenza tra il vettore al passo k+1 e il vettore al passo k        
        T norma = 0;

        cudaEventRecord(gpu_start, 0);
		reductionDiffT<<<numBlock, BlockSize, numThread * sizeof(T)>>>(deviceVectorResult, deviceVectorX, deviceNormaReduce, dim);
		if(numBlock != 1)
			reductionT<<<1, BlockSize, numThread * sizeof(T)>>>(deviceNormaReduce, deviceNormaReduce, numBlock);
		cudaEventRecord(gpu_stop, 0);
	    cudaEventSynchronize(gpu_stop);
	    cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
	    cout<<"\nCUDA runtime ReductionNormaVector: "<<gpu_runtime<<"ms\n";
	    error = cudaThreadSynchronize();
		error= cudaMemcpy(&norma, deviceNormaReduce, sizeof(T), cudaMemcpyDeviceToHost);
        
        cout<<"\n-----------------------------------------------------------------"<<endl;
        cout<<"Norma: "<<norma<<endl;
        cout<<"-----------------------------------------------------------------"<<endl;

        numBlock = ((dim * dim) + BlockSize -1) / BlockSize;

        if(norma < tool )
		{	
			cout<<"\n----------------------------------------------------------------------------------"<<endl;				
			cout<<"Criterio Di Arresto Rispettato!"<<endl;
			cout<<"----------------------------------------------------------------------------------"<<endl;
			return 0;
        }
        
    #//Copio il vettore al passo k+1 nel vettore al passo k
        cudaEventRecord(gpu_start, 0);
        copyVectorToVector<<<numBlock, numThread>>>(dim, deviceVectorResult, deviceVectorX);
        cudaEventRecord(gpu_stop, 0);
        cudaEventSynchronize(gpu_stop);
        cudaEventElapsedTime(&gpu_runtime, gpu_start, gpu_stop);
        cout<<"\nCUDA runtime CopyVectorToVector: "<<gpu_runtime<<"ms\n";
        error = cudaThreadSynchronize();    
        
        cout<<"-----------------------------------------------------------------"<<endl;
    }

    return 0;
}