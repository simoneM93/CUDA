#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <locale.h>

//Dimensione matrice e vettori (Globale)
const int dim = 4;

typedef int matrix[dim][dim];
typedef int vector[dim];


//Server a dividere la matrice iniziale A in D (Diagonale), LU(Triangolare Superiore e Inferiore)
void divisionMatrix(matrix A, matrix D, matrix LU)
{
  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      if (i == j)
      {
        D[i][j] = A[i][i];
        LU[i][j] = 0;
      }
      else
      {
        D[i][j] = 0;
        LU[i][j] = A[i][j];
      }
    }
  }
}

//Genera il vettore nullo x_0
void vectorX_0(vector x_0)
{
  for (int i = 0; i < dim; i++)
    x_0[i] = 0.00;
}

//Genera il vettore dei termini noti b
void vectorB(vector b)
{
  time_t j;
  srand(NULL);

  for (int i = 0; i < dim; i++)
    b[i] = rand() % 10;
}

void matrixA(matrix Matrix)
{
  time_t t;
  srand((unsigned)time(&t));

  for (int i = 0; i < dim; i++)
    for (int j = 0; j < dim; j++)
      Matrix[i][j] = rand() % 10+1;
}

//Stampa la matrice
void printMatrix(matrix Matrix)
{
  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
      printf(" %d ", Matrix[i][j]);

    printf("\n");
  }
  printf("\n");
}

//Stampa il vettore
void printVector(vector Vector)
{
  for (int i = 0; i < dim; i++)
  {
    printf(" %d\n", Vector[i]);
  }
  printf("\n\n");
}

//Effettua la somma di due matrici
void sumMatrix(matrix Matrix1, matrix Matrix2, matrix sumMatrix)
{
  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      sumMatrix[i][j] = Matrix1[i][j] + Matrix2[i][j];
    }
  }
}

//Effettua la moltiplicazione tra Matrice e Vettore
void multiplicationMatrixVector(matrix Matrix, vector Vector, vector Result)
{
  double sum = 0;

  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      sum += Matrix[i][j] * Vector[j];
    }
    Result[i] = sum;
  }
}

//Serve a verificare che la Matrice sia Strettamente Diagonalmente Dominante
bool diagonalyDominantMatrix(matrix Matrix)
{
  int sum = 0;

  for (int i = 0; i < dim; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      if (i != j)
        sum += (int)(fabs(Matrix[i][j]));
    }
    if (sum > fabs(Matrix[i][i]))
    {
      setlocale(LC_ALL, "");
      printf("Essendo sum = %d > Matrix[%d][%d] = %d ", sum, i, i, (int)(fabs(Matrix[i][i])));
      printf("la matrice non è Strettamente Diagonalmente Dominante (Condizione Sufficiente), quindi non può convergere utilizzando il Metodo di Jacobi!\n\n");
      return false;
    }
  }
  setlocale(LC_ALL, "");
  printf("La matrice è Strettamente Diagonalmente Dominante (Condizione Sufficiente), quindi convergere utilizzando il Metodo di Jacobi!\n\n");
  return true;
}

int main(int argc, char **argv)
{
  matrix A;
  vector B;
  vector Result;

  matrixA(A);
  printMatrix(A);

  diagonalyDominantMatrix(A);

  vectorB(B);
  printVector(B);

  multiplicationMatrixVector(A, B, Result);
  printVector(Result);

  return 0;
}