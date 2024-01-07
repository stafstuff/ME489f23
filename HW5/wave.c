# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <time.h>
# include <string.h>
# include "mpi.h"
#define BUFSIZE 512


// Function definitions
/******************************************************************************/
int main ( int argc, char *argv[] );
double exactSoln( double c, double x, double y, double t );
void applyBC(double *data,  double *x, double *y, double c, double time, int nx, int ny);
void solverPlot(char *fileName, double *x, double *y, int nx, int ny, double *data); 
double readInputFile(char *fileName, char* tag); 

// Solver Info
/******************************************************************************
  Purpose:
    wave2d solves the wave equation in parallel using MPI.
  Discussion:
    Discretize the equation for u(x,t):
      d^2 u/dt^2  =  c^2 * (d^2 u/dx^2 + d^2 u/dy^2)  
      for 0 < x < 1, 0 < y < 1, t>0
    with boundary conditions and Initial conditions obtained from the exact solutions:
      u(x,y, t) = sin ( 2 * pi * ( x - c * t ) )
   Usage: serial -> ./wave input.dat  parallel> mpirun -np 4 ./wave input.dat 
******************************************************************************/

int main ( int argc, char *argv[] ){
  
  // Read input file for solution parameters
  double tstart = readInputFile(argv[1], "TSART"); // Start time
  double tend   = readInputFile(argv[1], "TEND");  // End time
  double dt     = readInputFile(argv[1], "DT");    // Time step size

  // Global node number in x and y
  int NX        = (int) readInputFile(argv[1], "NX"); // Global node numbers in x direction
  int NY        = (int) readInputFile(argv[1], "NY"); // Global node numbers in y direction

  double xmax = readInputFile(argv[1], "XMAX"); // domain boundaries
  double xmin = readInputFile(argv[1], "XMIN"); // domain boundaries
  double ymax = readInputFile(argv[1], "YMAX"); // domain boundaries
  double ymin = readInputFile(argv[1], "YMIN"); // domain boundaries
  double c = readInputFile(argv[1], "CONSTANT_C");

  double *qn, *q0, *q1;               // Solution field at t+dt, t and t-dt
  static int frame=0; 

  int rank, size;
  double wtime;

  // Initialize MPI, get size and rank
  MPI_Init ( &argc, &argv );
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
  MPI_Comm_size ( MPI_COMM_WORLD, &size );
  MPI_Status status;

  wtime = MPI_Wtime();
  // DOMAIN DECOMPOSITION
  // For serial implementation nx = NX and ny = NY; 
  int nx = NX;      // local number of nodes in x direction
  // Decomposing domain in only y direction between processes
  int ny = NY/size;      // local number of nodes in y direction

  // ALLOCATE MEMORY for COORDINATES (x, y) and compute them
  double *x = ( double * ) malloc ( nx*ny * sizeof ( double ) );
  double *y = ( double * ) malloc ( nx*ny * sizeof ( double ) );
  // find uniform spacing in x and y directions
  double hx = (xmax - xmin)/(NX-1.0); 
  double hy = (ymax - ymin)/(NY-1.0); 
  // Compute coordinates of the nodes
  for(int j=0; j < ny; ++j){
    for(int i=0; i < nx;++i){
      double xn = xmin + i*hx;
      // Giving each process a unique set of y coordinates
      double yn = ymin + j*hy + rank*ny*hy;  
      
      x[i+j*nx] = xn;
      y[i+j*nx] = yn;
    }
  }

  
  // ALLOCATE MEMORY for SOLUTION and its HISTORY
  // Allocating two halo rows for MPI communication
  // Solution at time (t+dt)
  qn = ( double * ) malloc ( nx*(ny+2) * sizeof ( double ) );
  // Solution at time (t)
  q0 = ( double * ) malloc ( nx*(ny+2) * sizeof ( double ) );
  // Solution at time t-dt
  q1 = ( double * ) malloc ( nx*(ny+2) * sizeof ( double ) );
  // Allocate memory global variables
  double *q0_global=(double*) malloc(nx*ny*size*sizeof(double));
  double *q1_global=(double*) malloc(nx*ny*size*sizeof(double));
  double *y_global=(double*) malloc(nx*ny*size*sizeof(double));
  double *x_global=(double*) malloc(nx*ny*size*sizeof(double));
  // USE EXACT SOLUTION TO FILL HISTORY
  if(rank==0){
   for(int i=0; i<nx; i++){
      for(int j=1; j<ny; j++){
      const double xn = x[i+ j*nx]; 
      const double yn = y[i+ j*nx]; 
      // Exact solutions at history tstart and tstart+dt
      q0[i + (j+1)*nx] = exactSoln(c, xn, yn, tstart + dt);  
      q1[i + (j+1)*nx] = exactSoln(c, xn, yn, tstart);  
      }
    }
  }else{
      for(int i=0; i<nx; i++){
      for(int j=0; j<ny; j++){
      const double xn = x[i+ j*nx]; 
      const double yn = y[i+ j*nx]; 
      // Exact solutions at history tstart and tstart+dt
      q0[i + (j+1)*nx] = exactSoln(c, xn, yn, tstart + dt);  
      q1[i + (j+1)*nx] = exactSoln(c, xn, yn, tstart);  
      }
    }
  }



  //Collect the values for q1, y and x from all of the processes
  MPI_Allgather(&q1[0+1*nx],nx*ny,MPI_DOUBLE,q1_global,nx*ny,MPI_DOUBLE,MPI_COMM_WORLD);
  MPI_Allgather(y,nx*ny,MPI_DOUBLE,y_global,nx*ny,MPI_DOUBLE,MPI_COMM_WORLD);
  MPI_Allgather(x,nx*ny,MPI_DOUBLE,x_global,nx*ny,MPI_DOUBLE,MPI_COMM_WORLD);


  // Write the initial solution
  if(rank==0){
    char fname[BUFSIZ];
    sprintf(fname, "test_%04d.csv", frame++);
    solverPlot(fname, x_global, y_global, nx, size*ny, q1_global);
  }
  free(q1_global);
// RUN SOLVER 
  int Noutput = 10000; 
  int Nsteps=(tend - tstart)/dt;     // Assume  dt divides (tend- tstart)
  double alphax2 = pow((c*dt/hx),2); 
  double alphay2 = pow((c*dt/hy),2);
  
  // We already have 2 steps computed with exact solution
  double time = dt; 
  // for every time step
  for(int tstep = 2; tstep<=Nsteps+1; ++tstep){
    // increase  time
    time = tstart + tstep*dt; 
    
    // Apply Boundary Conditions
    applyBC(q0, x, y, c, time, nx, ny);

    //Communicate the values with MPI
    //UP COMM
    if(rank!=size-1){
      MPI_Send(&q0[0 + ny*nx],nx,MPI_DOUBLE,rank+1,tstep,MPI_COMM_WORLD);
      MPI_Recv(&q0[0 + (ny+1)*nx],nx,MPI_DOUBLE,rank+1,tstep,MPI_COMM_WORLD,&status);
    }

    //DOWN COMM
    if(rank!=0){
      MPI_Send(&q0[0 + 1*nx],nx,MPI_DOUBLE,rank-1,tstep,MPI_COMM_WORLD);
      MPI_Recv(&q0[0 + 0*nx],nx,MPI_DOUBLE,rank-1,tstep,MPI_COMM_WORLD,&status);
    }
    
    // Update solution using second order central differencing in time and space
    for(int i=1; i<nx-1; i++){ // exclude left right boundaries
      for(int j= 1; j<ny+1 ; j++){
        const int n0   = i + j*nx; 
        const int nim1 = i - 1 + j*nx; // node i-1,j
        const int nip1 = i + 1 + j*nx; // node i+1,j
        const int njm1 = i + (j-1)*nx; // node i, j-1
        const int njp1 = i + (j+1)*nx; // node i, j+1
        // update solution 
        qn[n0] = 2.0*q0[n0] - q1[n0] + alphax2*(q0[nip1]- 2.0*q0[n0] + q0[nim1])
                                     + alphay2*(q0[njp1] -2.0*q0[n0] + q0[njm1]); 
      }
    }
    // Reapply the boundary condition since it was lost in the above process.
    applyBC(qn, x, y, c, time, nx, ny);
    // Update history q1 = q0; q0 = qn, except halo rows and left-right boundaries
    for(int i=1; i<nx-1; i++){
      for(int j=1; j<ny+1; j++){
        q1[i + j*nx] = q0[i + j*nx]; 
        q0[i + j*nx] = qn[i + j*nx]; 
        
      }
    }
    // Dampout a csv file for postprocessing
    if(tstep%Noutput == 0){ 
        MPI_Allgather(&q0[0+1*nx],nx*ny,MPI_DOUBLE,q0_global,nx*ny,MPI_DOUBLE,MPI_COMM_WORLD);
        if(rank==0){ 
          char fname[BUFSIZ];
          sprintf(fname, "test_%04d.csv", frame++);
          solverPlot(fname, x_global, y_global, nx, size*ny, q0_global);
        }
      }
    }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allgather(&q0[0+1*nx],nx*ny,MPI_DOUBLE,q0_global,nx*ny,MPI_DOUBLE,MPI_COMM_WORLD);
  // Compute Linf norm of error at tend
  if(rank==0){
    double linf = 0.0; 
    for(int i=0; i<nx; i++){
      for(int j=0; j<size*ny; j++){
         double xn = x_global[i+ j*nx]; 
         double yn = y_global[i+ j*nx]; 
         // solution and the exact one
         double qn = q0_global[i+ j*nx]; 
         double qe = exactSoln(c, xn, yn, time);
         linf  = fabs(qn-qe)>linf ? fabs(qn -qe):linf; 
      }
    }
  
    // Print out the results
    printf("Infinity norm of the error: %.4e %.8e \n", linf, time);
  }
  //Record wall clock time
  wtime = MPI_Wtime( )-wtime;

  double global_time = 0.0; 
  MPI_Reduce( &wtime, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

 if(rank==0) printf ( "Wall clock elapsed seconds = %f \n", global_time);
 
  free(q0_global),free(x_global),free(y_global);
  // Terminate MPI
  MPI_Finalize();
  return 0;
}




/***************************************************************************************/
double exactSoln( double c, double x, double y, double t){
  const double pi = 3.141592653589793;
  double value = sin( 2.0*pi*( x - c*t));
  return value;
}

/***************************************************************************************/
void applyBC(double *data,  double *x, double *y, double c, double time, int nx, int ny){

  // Apply Boundary Conditions
  double xn, yn; 
  int rank, size;

  MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
  MPI_Comm_size ( MPI_COMM_WORLD, &size );
  for(int j=0; j<ny;++j){ // left right boundaries i.e. i=0 and i=nx-1
    xn = x[0 + j*nx]; 
    yn = y[0 + j*nx];    
    data[0 + (j+1)*nx] = exactSoln(c, xn, yn, time); 

    xn = x[nx-1 + j*nx]; 
    yn = y[nx-1 + j*nx];    
    data[nx-1 + (j+1)*nx] = exactSoln(c, xn, yn, time); 
  }

  // Only on rank=0 and rank=size-1
  if(rank==0){
  for(int i=0; i< nx; ++i){ // bottom boundaries if rank=0
    xn = x[i+ 0*nx]; 
    yn = y[i+ 0*nx]; 
    data[i + 1*nx] = exactSoln(c, xn, yn, time); 
  }
  }
  if (rank==size-1){
  for(int i=0; i< nx; ++i){ //top boundaries if rank=size-1
    xn = x[i+ (ny-1)*nx]; 
    yn = y[i+ (ny-1)*nx];       
    data[i + ny*nx] = exactSoln(c, xn, yn, time); 
  }
  }
}

/* ************************************************************************** */
void solverPlot(char *fileName, double *x, double *y, int nx, int ny, double *Q){
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening file\n");
        return;
    }

    fprintf(fp, "X,Y,Z,Q \n");
     for(int i=0; i<nx; i++){
      for(int j=0; j<ny; j++){
        const double xn = x[i + j*nx]; 
        const double yn = y[i + j*nx]; 
        fprintf(fp, "%.8f, %.8f,%.8f,%.8f\n", xn, yn, 0.0, Q[i + j*nx]);
      }
    }
}


/* ************************************************************************** */
double readInputFile(char *fileName, char* tag){
  FILE *fp = fopen(fileName, "r");
  if (fp == NULL) {
    printf("Error opening the input file\n");
    return -1;
  }

  int sk = 0; 
  double result; 
  char buffer[BUFSIZE];
  char fileTag[BUFSIZE]; 
  while(fgets(buffer, BUFSIZE, fp) != NULL){
    sscanf(buffer, "%s", fileTag);
    if(strstr(fileTag, tag)){
      fgets(buffer, BUFSIZE, fp);
      sscanf(buffer, "%lf", &result); 
      return result;
    }
    sk++;
  }

  if(sk==0){
    printf("could not find the tag: %s in the file %s\n", tag, fileName);
  }
}
