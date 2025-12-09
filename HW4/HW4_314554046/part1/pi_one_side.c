#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h> 
#include <limits.h> 

struct xorshift32_state {
    uint32_t a;
};

uint32_t xorshift32(struct xorshift32_state *state)
{
    uint32_t x = state->a;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return state->a = x;
}

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    long long slot = 0;
    void *baseptr = NULL;
    MPI_Aint winsz = 0;

    if (world_rank == 0) {
        baseptr = (void*)&slot;
        winsz = (MPI_Aint)sizeof(long long);
    }

    MPI_Win_create(baseptr, winsz, sizeof(long long),
               MPI_INFO_NULL, MPI_COMM_WORLD, &win);


    long long base = tosses / world_size;
    long long rem  = tosses % world_size;
    long long local_tosses = base + (world_rank < rem ? 1 : 0);

    struct xorshift32_state rng_state;
    long seed = (long)(time(NULL) ^ (world_rank << 16) ^ getpid());
    rng_state.a = (seed == 0) ? 1 : (uint32_t)seed;
    
    long long local_hit = 0;
    long long i = 0;
    const double INV_2P32 = 1.0 / 4294967296.0;

    for (; i + 8 <= local_tosses; i += 8) {
    int h = 0;

    double x0 = xorshift32(&rng_state) * INV_2P32;
    double y0 = xorshift32(&rng_state) * INV_2P32; h += (x0*x0 + y0*y0 <= 1.0);

    double x1 = xorshift32(&rng_state) * INV_2P32;
    double y1 = xorshift32(&rng_state) * INV_2P32; h += (x1*x1 + y1*y1 <= 1.0);

    double x2 = xorshift32(&rng_state) * INV_2P32;
    double y2 = xorshift32(&rng_state) * INV_2P32; h += (x2*x2 + y2*y2 <= 1.0);

    double x3 = xorshift32(&rng_state) * INV_2P32;
    double y3 = xorshift32(&rng_state) * INV_2P32; h += (x3*x3 + y3*y3 <= 1.0);

    double x4 = xorshift32(&rng_state) * INV_2P32;
    double y4 = xorshift32(&rng_state) * INV_2P32; h += (x4*x4 + y4*y4 <= 1.0);

    double x5 = xorshift32(&rng_state) * INV_2P32;
    double y5 = xorshift32(&rng_state) * INV_2P32; h += (x5*x5 + y5*y5 <= 1.0);

    double x6 = xorshift32(&rng_state) * INV_2P32;
    double y6 = xorshift32(&rng_state) * INV_2P32; h += (x6*x6 + y6*y6 <= 1.0);

    double x7 = xorshift32(&rng_state) * INV_2P32;
    double y7 = xorshift32(&rng_state) * INV_2P32; h += (x7*x7 + y7*y7 <= 1.0);

    local_hit += h; 
    }

    for (; i < local_tosses; ++i) {
        double x = xorshift32(&rng_state) * INV_2P32;
        double y = xorshift32(&rng_state) * INV_2P32;
        local_hit += (x*x + y*y <= 1.0);
    }    

    MPI_Comm shm_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, 
                        MPI_INFO_NULL, &shm_comm);
    int shm_rank;
    MPI_Comm_rank(shm_comm, &shm_rank);    
    long long node_hit_sum = 0;
    MPI_Reduce(&local_hit, &node_hit_sum, 1, MPI_LONG_LONG,
               MPI_SUM, 0, shm_comm);


    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

    if (shm_rank == 0)
    {
        MPI_Accumulate(&node_hit_sum, 1, MPI_LONG_LONG, 0, 0, 1,
                       MPI_LONG_LONG, MPI_SUM, win);
	MPI_Win_flush(0, win);
    }
    
    MPI_Win_unlock_all(win);
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Win_free(&win);

    if (world_rank == 0)
    {
        // TODO: handle PI result
	 long long total_hit = slot; 
        pi_result = 4.0 * (double)total_hit / (double)tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
    }

    MPI_Finalize();
    return 0;
}
