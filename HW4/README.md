compile the heat1d.c with:

mpicc -g -Wall -o heat1d heat1d.c -lm

compile the heat1d_omp.c with:

mpicc -fopenmp -g -Wall  -o heat1d_omp heat1d_omp.c -lm

run both with

mpirun(or mpiexec) -n [PROCESS_NUM] ./heat1d(_omp) N -lm

Change the OUT to 0 while measuring time

Change number of threads by changing THREAD_NUM value