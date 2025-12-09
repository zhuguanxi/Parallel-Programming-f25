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

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // TODO: binary tree redunction
    long long base = tosses / world_size;
    long long rem  = tosses % world_size;
    long long local_tosses = base + (world_rank < rem ? 1 : 0);

    srand48((long)(time(NULL) ^ (world_rank << 16) ^ getpid()));

    long long acc = 0;
    for (long long i = 0; i < local_tosses; ++i) {
        double x = drand48(), y = drand48();
	acc += (x*x + y*y <= 1.0);
    }
    
    for (int step = 1; step < world_size; step <<= 1) {
        if ((world_rank % (step << 1)) == 0) {
            int src = world_rank + step;
            if (src < world_size) {
                long long buf = 0;
                MPI_Recv(&buf, 1, MPI_LONG_LONG, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                acc += buf;
            }
        } else {
            int dst = world_rank - step;
            MPI_Send(&acc, 1, MPI_LONG_LONG, dst, 0, MPI_COMM_WORLD);
            break; 
        }
    }

    if (world_rank == 0)
    {
        // TODO: PI result
	long long total_hit = acc;
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
