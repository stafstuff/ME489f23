# include <math.h>
# include <stdlib.h>
# include <stdio.h>
# include <time.h>
# include <omp.h>
# define OUT 0
# define THREAD_NUM 8

// Include MPI header
# include "mpi.h"

// Function definitions
int main ( int argc, char *argv[] );
double boundary_condition ( double x, double time );
double initial_condition ( double x, double time );
double source ( double x, double time );
void runSolver( int n, int rank, int size );



/*-------------------------------------------------------------
  Purpose: Compute number of primes from 1 to N with naive way
 -------------------------------------------------------------*/
// This function is fully implemented for you!!!!!!
// usage: mpirun -n 4 heat1d N
// N    : Number of nodes per processor
int main ( int argc, char *argv[] ){
  int rank, size;
  double wtime;

  // Initialize MPI, get size and rank
  MPI_Init ( &argc, &argv );
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
  MPI_Comm_size ( MPI_COMM_WORLD, &size );

  // get number of nodes per processor
  int N = strtol(argv[1], NULL, 10);

  // Solve and update the solution in time
  runSolver(N, rank, size);


  // Terminate MPI.
  MPI_Finalize ( );
  // Terminate.
  return 0;
}

/*-------------------------------------------------------------
  Purpose: computes the solution of the heat equation.
 -------------------------------------------------------------*/
void runSolver( int n, int rank, int size ){
  // CFL Condition is fixed
  double cfl = 0.5; 
  // Domain boundaries are fixed
  double x_min=0.0, x_max=1.0;
  // Diffusion coefficient is fixed
  double k   = 0.002;
  // Start time and end time are fixed
  double tstart = 0.0, tend = 10.0;  
  // Storage for node coordinates, solution field and next time level values
  double *x, *q, *qn;
  // Set the x coordinates of the n nodes padded with +2 ghost nodes. 
  x  = ( double*)malloc((n+2)*sizeof(double));
  q  = ( double*)malloc((n+2)*sizeof(double));
  qn = ( double*)malloc((n+2)*sizeof(double));

  // Write solution field to text file if size==1 only
  FILE *qfile, *xfile;

  // uniform grid spacing
  double dx = ( x_max - x_min ) / ( double ) ( size * n - 1 );

  // Set time step size dt <= CFL*h^2/k
  // and then modify dt to get integer number of steps for tend
  double dt  = cfl*dx*dx/k; 
  int Nsteps = ceil(( tend - tstart )/dt);
  dt =  ( tend - tstart )/(( double )(Nsteps)); 

  int tag;
  MPI_Status status;
  double time, time_new, wtime;  

  // find the coordinates for uniform spacing 
  for ( int i = 0; i <= n + 1; i++ ){

    x[i]=rank*n*dx/size+dx*(i-1);

  }

  // Set the values of q at the initial time.
  time = tstart; q[0] = 0.0; q[n+1] = 0.0;
  for (int i = 1; i <= n; i++ ){
    q[i] = initial_condition(x[i],time);
  }
  // In single processor mode
  if (size == 1 && OUT==1){
    
    // write out the x coordinates for display.

    xfile = fopen ( "x_data.txt", "w" );

    for (int i = 1; i<(n+1); i++ ){

      fprintf ( xfile, " %f", x[i] );

    }

    fprintf(xfile,"\n");
    fclose ( xfile );

    // write out the initial solution for display.

    qfile = fopen ( "q_data_0000.txt", "w" );

    for ( int i = 1; i <= n; i++ ){

      fprintf ( qfile, " %f", q[i] );

    }

    fprintf(qfile,"\n");
    fclose ( qfile );
  }

// In multi-processor mode
  else if(OUT==1){
    
    // write out the x coordinates for display.
    {
      double *x_global=(double*)malloc((size*n)*sizeof(double));

      MPI_Allgather(&x[1],n,MPI_DOUBLE,x_global,n,MPI_DOUBLE,MPI_COMM_WORLD);

      if(rank==0){

        xfile = fopen ( "x_data.txt", "w" );

      for (int i = 0; i<=size*n-1; i++ ){

        fprintf ( xfile, " %f", x_global[i] );

      }

      fprintf(xfile,"\n");

      fclose ( xfile );free(x_global);
      
      }
    }
    // write out the q coordinates for display.
    {
      double *q_global=(double*)malloc((size*n)*sizeof(double));

      MPI_Allgather(&q[1],n,MPI_DOUBLE,q_global,n,MPI_DOUBLE,MPI_COMM_WORLD);

      

      if(rank==0){

      qfile = fopen("q_data_0000.txt","w");

      for (int i = 0; i<=size*n-1; i++ ){

        fprintf ( qfile, " %f", q_global[i] );

      }

      fprintf ( qfile, "\n" );
      fclose ( qfile );free(q_global);

      }
    }
  }
  



 // Record the starting time.
  wtime = MPI_Wtime();
     
  // Compute the values of q at the next time, based on current data.
  for ( int step = 1; step <= Nsteps; step++ ){

    time_new = time + step*dt;
    //RIGHT COMM
    if(rank!=size-1){

      MPI_Send(&q[n],1,MPI_DOUBLE,rank+1,step,MPI_COMM_WORLD);
      MPI_Recv(&q[n+1],1,MPI_DOUBLE,rank+1,step,MPI_COMM_WORLD,&status);

    }
    //LEFT COMM
    if(rank!=0){

      MPI_Send(&q[1],1,MPI_DOUBLE,rank-1,step,MPI_COMM_WORLD);
      MPI_Recv(&q[0],1,MPI_DOUBLE,rank-1,step,MPI_COMM_WORLD,&status);

    }
  
    omp_set_num_threads(THREAD_NUM);
    #pragma omp parallel for
    for ( int i = 1; i <= n; i++ ){

      double rhsqt;

      rhsqt=k/pow(dx,2)*(q[i-1]-2*q[i]+q[i+1]+source(x[i],time));

      qn[i]=q[i]+dt*rhsqt;

    }

  
    // q at the extreme left and right boundaries was incorrectly computed
    // using the differential equation.  
    // Replace that calculation by the boundary conditions.
    // global left endpoint 
    if (rank==0){

      qn[1] = boundary_condition ( x[1], time_new );

    }
    // global right endpoint 
    if (rank == size - 1 ){

      qn[n] = boundary_condition ( x[n], time_new );
      qn[n+1] = qn[n];

    }

  // Update time and field.
    time = time_new;
    // For OpenMP make this loop parallel also
    #pragma omp parallel for
    for ( int i = 1; i <= n; i++ ){
      q[i] = qn[i];
    }

  // In single processor mode, add current solution data to output file.
    if (size == 1 && OUT==1){

      char qname[BUFSIZ];
      sprintf(qname,"q_data_%04d.txt",step);

      qfile=fopen(qname,"w");

      for ( int i = 1; i <= n; i++ ){

        fprintf ( qfile, " %f", q[i] );

      }

      fprintf ( qfile, "\n" );
    }
    // In multi-processor mode, add current solution data to output file.
    {
    if (size!=1 && OUT==1){

        double *q_global=(double *)malloc((n*size)*sizeof(double));

        MPI_Allgather(&q[1],n,MPI_DOUBLE,q_global,n,MPI_DOUBLE,MPI_COMM_WORLD);

        if(rank==0){

          char qname[BUFSIZ];
          sprintf(qname,"q_data_%04d.txt",step);

          qfile=fopen(qname,"w");

          for ( int i = 0; i <= size*n-1; i++ ){

            fprintf ( qfile, " %f", q_global[i] );

          }
          fprintf(qfile,"\n");

          fclose ( qfile );free(q_global);
          
        }
    }
    }
  }

  
  // Record the final time.
  wtime = MPI_Wtime( )-wtime;

  double global_time = 0.0; 
  MPI_Reduce( &wtime, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

 if(rank==0)
   printf ( "  Wall clock elapsed seconds = %f , %d\n", global_time,rank );
  if( size == 1 && OUT==1)
  fclose ( qfile ); 

  free(q); free(qn); free(x);

  return;
}
/*-----------------------------------------------------------*/
double boundary_condition ( double x, double time ){
  double value;

  // Left condition:
  if ( x < 0.5 ){
    value = 100.0 + 10.0 * sin ( time );
  }else{
    value = 75.0;
  }
  return value;
}
/*-----------------------------------------------------------*/
double initial_condition ( double x, double time ){
  double value;
  value = 95.0;

  return value;
}
/*-----------------------------------------------------------*/
double source ( double x, double time ){
  double value;

  value = 0.0;

  return value;
}
