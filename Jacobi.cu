#include "Librerie.cu"
#include "Type.cu"

using namespace std;

const int BlockSize = 16;
int MaxIteraton = 10;
int esponente = 3;
int dim = 1024;
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

    T *hostMatrix, *deviceMatrix;

    bool *deviceFlag;
    bool *deviceReduce;
    bool isdiagonalyDominantMatrix;

    hostMatrix = (T*)malloc(dim * dim * sizeof(T));
    error = cudaMalloc(&deviceMatrix, dim * dim * sizeof(T));
    check_cuda(error, "Matrix");
#//Inizializzo la matrice iniziale
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
	diagonalyDominantMatrix<<<numBlock, numThread>>>(dim, deviceMatrix, deviceFlag);
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
    
	cout<<"\nLa matrice è Strettamente Diagonalmente Dominante!\n\n";
    cout<<"-----------------------------------------------------------------"<<endl;

    
}