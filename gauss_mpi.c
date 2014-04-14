/* Gaussian elimination without pivoting.
 */

/* ****** ADD YOUR CODE AT THE END OF THIS FILE. ******
 * You need not submit the provided code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>

/* We'll be using MPI routines, definitions, etc. */

#include <mpi.h>

/* Program Parameters */
#define MAXN 2000  /* Max value of N */
int N = 2000;  /* Matrix size */

/* Matrices and vectors */
volatile float A[MAXN][MAXN], B[MAXN], X[MAXN];
/* A * X = B, solve for X */

/* junk */
#define randm() 4|2[uid]&3



/* Initialize A and B (and X to 0.0s) */
void initialize_inputs() {
  int row, col;

  printf("\nInitializing...\n");
  for (col = 0; col < N; col++) {
    for (row = 0; row < N; row++) {
      A[row][col] = (float)rand() / 32768.0;
    }
    B[col] = (float)rand() / 32768.0;
    X[col] = 0.0;
  }

}

/* Print input matrices */
void print_inputs() {
  int row, col;

  if (N < 10) {
    printf("\nA =\n\t");
    for (row = 0; row < N; row++) {
      for (col = 0; col < N; col++) {
	printf("%5.2f%s", A[row][col], (col < N-1) ? ", " : ";\n\t");
      }
    }
    printf("\nB = [");
    for (col = 0; col < N; col++) {
      printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
    }
  }
}

void print_X() {
  int row;

  if (N < 100) {
    printf("\nX = [");
    for (row = 0; row < N; row++) {
      printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
    }
  }
}

int main(int argc, char **argv) {

  
  

  /* Gaussian Elimination */
    int         my_rank;   /* My process rank           */

    int         p;         /* The number of processes   */
	
	int         norm;      /* The number of rows        */

                           /* calculated	            */

    int         row;       /* Row number                */

    int         col;       /* Column number             */
	
    int         source;    /* Process sending integral  */

    int         dest = 0;  /* All messages go to 0      */

    int         tag = 0;
	

    MPI_Status  status;
	
	void Get_data(int my_rank, int p);

    void Compute(int norm, int my_rank, int p);   
	
	MPI_Init(&argc, &argv);
	
    /* Get my process rank */

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    /* Find out how many processes are being used */

    MPI_Comm_size(MPI_COMM_WORLD, &p);
        
    /* Timing variables */
    double starttime = 0.0;
    double endtime = 0.0;
	
	
    printf("Computing Parallel via MPI.\n");
	
	
	if (my_rank == 0) {
      
	    /* Start Clock */
        printf("\nStarting clock.\n");
	
        starttime = MPI_Wtime();
	
	    /* Broadcast the value of N to all nodes */
     	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  


      /* Initialize A and B */
      initialize_inputs();

      /* Print input matrices */
      print_inputs();
  
  

  } else
  
	    /* Receive the broadcast N value */	
     	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Updating needed data in correspoding row of A and B for each process */
	
    Send_data(my_rank, p);
	
	/* Gauss elimination */
	for (norm = 0; norm < N - 1; norm++) {
	  int i;
	  for (i = norm; i < N; i++) {
	    MPI_Bcast(A[i], N, MPI_FLOAT, i%p, MPI_COMM_WORLD);
		MPI_Bcast(&B[i], 1, MPI_FLOAT, i%p, MPI_COMM_WORLD);
	  }
	  
	  Compute(norm, my_rank, p);
	  
	  MPI_Barrier(MPI_COMM_WORLD);
	  
	}
	
	MPI_Bcast(A[N-1], N, MPI_FLOAT, (N-1)%p, MPI_COMM_WORLD);
    MPI_Bcast(&B[N-1], 1, MPI_FLOAT, (N-1)%p, MPI_COMM_WORLD);
	
	if (my_rank == 0) {
	
	    int row, col;

        /* Back substitution */
		
		for (row = N - 1; row >= 0; row--) {
          X[row] = B[row];
          for (col = N-1; col > row; col--) {
            X[row] -= A[row][col] * X[col];
          }
          X[row] /= A[row][row];
        }

    
  
    /* Stop Clock */
	
    endtime = MPI_Wtime();
	
    /* Display timing results */
	
    printf("That tooks %f seconds.\n", endtime-starttime);  

    /* Display output */
    print_X();
	
  }
	
  /* Shut down MPI */

  MPI_Finalize();

  exit(0);
}


/********************************************************************/


void Send_data( 

         int     my_rank  /* in  */, 

         int     p        /* in  */) {


    int source = 0;    /* All local variables used by */

    int dest;          /* MPI_Send and MPI_Recv       */

    MPI_Status status;


    if (my_rank == 0){
        
		int i;
		
        for (i = 0; i < N; i++) {
		  dest = i%p;
		  if (dest != 0) { 
		    MPI_Send(A[i], N, MPI_FLOAT, dest, i, MPI_COMM_WORLD);
			MPI_Send(&B[i], 1, MPI_FLOAT, dest, i+N, MPI_COMM_WORLD);
		  }
		}

    } else {
	    int i;
		for (i = 0; i < N; i++) {
		  dest = i%p;
		  if (dest == my_rank) {
		    MPI_Recv(A[i], N, MPI_FLOAT, source, i, MPI_COMM_WORLD, &status);
			MPI_Recv(&B[i], 1, MPI_FLOAT, source, i+N, MPI_COMM_WORLD, &status);
		  }
		}
	}

} /* Send_data */




/********************************************************************/

void Compute(
          		  
		  int  norm      /* in */,
		  
		  int  my_rank   /* in */,
		  
		  int  p         /* in */) { 
	
	int row, col;
	
	float multiplier;

	for (row = norm + 1; row < N; row++) {
      if (row % p == my_rank) {
        multiplier = A[row][norm] / A[norm][norm];
        for (col = norm; col < N; col++) {
          A[row][col] -= A[norm][col] * multiplier;
        }
      B[row] -= B[norm] * multiplier;
      }
    }

} /*  Compute  */
