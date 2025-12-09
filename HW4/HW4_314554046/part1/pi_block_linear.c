#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: init MPI
    
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    long long base = tosses / world_size;
    long long rem  = tosses % world_size;
    long long local_tosses = base + (world_rank < rem ? 1 : 0);
    
    srand48((long)(time(NULL) ^ (world_rank << 16) ^ getpid()));
    long long local_hit = 0;
    for (long long i = 0; i < local_tosses; ++i) {
        double x = drand48(), y = drand48();
	local_hit += (x*x + y*y <= 1.0);
    }

    if (world_rank > 0)
    {
        // TODO: handle workers
	MPI_Send(&local_hit, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0)
    {
        // TODO: main
    }

    if (world_rank == 0)
    {
        // TODO: process PI result
        
	long long total_hit = local_hit; 
        for (int src = 1; src < world_size; ++src) {
            long long tmp = 0;
            MPI_Recv(&tmp, 1, MPI_LONG_LONG, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_hit += tmp;
        }
        pi_result = 4.0 * (double)total_hit / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
