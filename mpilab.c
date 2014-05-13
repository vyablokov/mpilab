#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>

#define _REENTRANT
//#define PRINT

#define X 15
#define T 100
#define PI 3.14
#define a 1.0
#define dX 1
#define dT 0.25

int main(int argc, char **argv) {
	int myrank, total;
	double *Res;	//only for root
	double *res;	// Лента матрицы [mxn], вектор [n], рез-т [m]
	int n, m;
	int i, j;
	MPI_Status status1, status2;
	struct timeval start, end;

  	MPI_Init (&argc, &argv);
  	MPI_Comm_size (MPI_COMM_WORLD, &total);
  	MPI_Comm_rank (MPI_COMM_WORLD, &myrank);

	n = round(T/dT)+1;
	m = (X+1)/total;
	if (!myrank) 
	{	
		Res = (double *) malloc (sizeof(double)*n*(X+1));
		for(j = 0; j < (X + 1); j++) {
	   		//Res[0*n + j] = sin(PI * dX * j/X); //Initial string configuration
			Res[0*n + j] = 0;
		}
	};
	res=malloc(n*m*sizeof(double));
	if(!myrank) {
		gettimeofday(&start, 0);
	}

 	MPI_Scatter((void *)Res, m, MPI_DOUBLE, (void *)res, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);

 	for(j = 0; j < m; j++) {
		//res[1*m + j] = res[0*m + j] + dX * (dT * dT);	//because Z'(x, 0) = 0
		res[1*m + j] = res[0*m + j];	//because Z'(x, 0) = 0
	}

	for(i = 2; i < n; i++)
	{
		double f, prev_left, prev_right;
		if(myrank!=0)
		{
			MPI_Sendrecv(&res[(i-1)*m], 1, MPI_DOUBLE, myrank-1, 0, &prev_left, 
1, MPI_DOUBLE, myrank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status1);
		}
		if(myrank!=total-1)
		{
			MPI_Sendrecv(&res[(i-1)*m+(m-1)], 1, MPI_DOUBLE, myrank+1, 1, &prev_right, 1, 
MPI_DOUBLE, myrank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status2);
		}

		/*if(myrank!=0)
		{
			MPI_Recv(&prev_left, 1, MPI_DOUBLE, myrank-1, MPI_ANY_TAG, MPI_COMM_WORLD, &status1);
		}
		if(myrank!=total-1)
		{
			MPI_Recv(&prev_right, 1, MPI_DOUBLE, myrank+1, MPI_ANY_TAG, MPI_COMM_WORLD, &status2);
		}*/

		for(j = 0; j < m; j++)
		{
			if(fabs((double)(myrank*m + j) / (double)(total*m) - 0.25) < 1E-5) {
				//f = 0;
				f = sin(PI * i * dT) * 0.2 * (X+1) ;
				//printf("%d: %d,%d: %.2f\n", myrank, i, j, (double)(myrank*m + j) / (double)(total*m));
			}
			else {
				f = 0;
			}
			if(j == 0)
			{
				if(myrank == 0)
				{
					res[i*m+j]=res[(i-1)*m+j];
				}
				else
				{
					res[i*m + j] = a*a * ((dT*dT) / (dX*dX)) * (res[(i-1)*m + (j+1)] - 2*res[(i-1)*m + j] +
					prev_left) + 2*res[(i-1)*m + j] - res[(i-2)*m + j] + f * dT * dT;
				}
			}
			else if(j == m-1)
			{
				if(myrank == total-1)
				{
					res[i*m+j]=res[(i-1)*m+j];
				}
				else
				{
					res[i*m + j] = a*a * ((dT*dT) / (dX*dX)) * (prev_right - 2*res[(i-1)*m + j] +
					res[(i-1)*m + (j-1)]) + 2*res[(i-1)*m + j] - res[(i-2)*m + j] + f * dT * dT;
				}
			}
			else
			{
				res[i*m + j] = a*a * ((dT*dT) / (dX*dX)) * (res[(i-1)*m + (j+1)] - 2*res[(i-1)*m + j]
				+ res[(i-1)*m + (j-1)]) + 2*res[(i-1)*m + j] - res[(i-2)*m + j] + f * dT * dT;
			}
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
	
	for(i=1; i<n; i++)
	{
		 MPI_Gather((void *)(res+i*m), m, MPI_DOUBLE, (void *)(Res+i*(X+1)), m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	}
	
if(!myrank)
{
	gettimeofday(&end, 0);
	FILE *f;
	char buf[1024];

#ifdef PRINT
	for(i=0; i<n; i++)
	{
		for(j=0; j<(X+1); j++)
			printf("%.2f ", Res[i*(X+1)+j]);
		printf("\n");
	}
#endif
	printf("Calculation time: %d sec %d usec.\n", end.tv_sec - start.tv_sec, end.tv_usec - start.tv_usec);

	for(i=0; i<n; i++)
    {
    	sprintf(buf, "t=%.2f", dT*(double)i);
    	f=fopen(buf, "w");
    	for(j=0; j<X+1; j++)
    	{
    		fprintf(f, "%d %f\n", j, Res[i*(X+1)+j]);
    	}
    	fclose(f);
    }
	f=fopen("com", "w");
    for(i=0; i<n; i++)
    {
		fprintf(f, "set yrange [-1:1]\n");
		fprintf(f, "plot [0:%d]\"t=%.2f\" smooth csplines\n", X, dT*(double)i);
		fprintf(f, "pause %.2f\n", dT);
    }
}
  MPI_Finalize();
  exit(0);
  }
