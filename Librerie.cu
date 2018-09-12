#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <cuda_runtime_api.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_fp16.h"

//Metodi Implementati Per L'Esecuzione Del Metodo Di Jacobi
#include "DiagonalyDominantMatrix.cu" //Serve a controllare se la matrice è Strettamente Diagonalmente Dominante
#include "InitMatrix.cu" //Inizializza la matrice
#include "InitVector.cu" //Inizializza il vettore
#include "SumMatrix.cu" //Esegue la somma tra matrici
#include "TransposedMatrix.cu" //Calcola la trasposta di una matrice
#include "MatrixDivision.cu" //Divide la matrice in Matrice Diagonale e Matrice Triangolare Superiore e Inferiore
#include "MoltiplicationMatrixVector.cu" //Esegue la moltiplicazione tra matrice e vettore
#include "SumVectorVector.cu" //Esegue la somma tra due vettori
#include "DiffVectorVector.cu" //Esegue la differenza tra due vettori
#include "NormaDue.cu" //Calcola la Norma Due di un vettore
#include "Check_Cuda.cu" //Errori
#include "CopyVectorToVector.cu" //Copia un vettore in un altro vettore