#include <mpi.h>
#include <cstring>  
#include <algorithm> 

static void compute_block(int N, int P, int idx, int &local_n, int &start)
{
    int base = N / P;
    int rem  = N % P;
    if (idx < rem)
    {
        local_n = base + 1;
        start   = idx * (base + 1);
    }
    else
    {
        local_n = base;
        start   = rem * (base + 1) + (idx - rem) * base;
    }
}

void construct_matrices(
    int n, int m, int l, const int *a_mat, const int *b_mat, int **a_mat_ptr, int **b_mat_ptr)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sizes[3];
    if (rank == 0)
    {
        sizes[0] = n;
        sizes[1] = m;
        sizes[2] = l;
    }
    MPI_Bcast(sizes, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int N = sizes[0];
    int M = sizes[1];
    int L = sizes[2];

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims); 
    int Pr = dims[0];
    int Pc = dims[1]; 

    int pr = rank / Pc;  
    int pc = rank % Pc;  
    int local_n, row_start;
    int local_l, col_start;

    compute_block(N, Pr, pr, local_n, row_start);
    compute_block(L, Pc, pc, local_l, col_start);
    *a_mat_ptr = new int[local_n * M];
    *b_mat_ptr = new int[M * local_l];

    const int TAG_A = 100;
    const int TAG_B = 200;

    if (rank == 0)
    {
        std::memcpy(*a_mat_ptr,
                    a_mat + row_start * M,
                    sizeof(int) * local_n * M);
        std::memcpy(*b_mat_ptr,
                    b_mat + col_start * M,
                    sizeof(int) * M * local_l);

        for (int proc = 1; proc < size; ++proc)
        {
            int other_pr = proc / Pc;
            int other_pc = proc % Pc;

            int on, ors, ol, ocs;
            compute_block(N, Pr, other_pr, on,  ors);
            compute_block(L, Pc, other_pc, ol, ocs);

            MPI_Send(a_mat + ors * M, on * M, MPI_INT, proc, TAG_A, MPI_COMM_WORLD);
            MPI_Send(b_mat + ocs * M, M * ol, MPI_INT, proc, TAG_B, MPI_COMM_WORLD);
        }
    }
    else
    {
        MPI_Status st;
        MPI_Recv(*a_mat_ptr, local_n * M, MPI_INT, 0, TAG_A, MPI_COMM_WORLD, &st);
        MPI_Recv(*b_mat_ptr, M * local_l, MPI_INT, 0, TAG_B, MPI_COMM_WORLD, &st);
    }
}

void matrix_multiply(
    const int n, const int m, const int l, const int *a_mat, const int *b_mat, int *out_mat)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sizes[3];
    if (rank == 0)
    {
        sizes[0] = n;
        sizes[1] = m;
        sizes[2] = l;
    }
    MPI_Bcast(sizes, 3, MPI_INT, 0, MPI_COMM_WORLD);
    int N = sizes[0];
    int M = sizes[1];
    int L = sizes[2];

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int Pr = dims[0];
    int Pc = dims[1];

    int pr = rank / Pc;
    int pc = rank % Pc;

    int local_n, row_start;
    int local_l, col_start;
    compute_block(N, Pr, pr, local_n, row_start);
    compute_block(L, Pc, pc, local_l, col_start);

    int *local_C = new int[local_n * local_l];
    std::fill(local_C, local_C + local_n * local_l, 0);

    for (int i = 0; i < local_n; ++i)
    {
        for (int j = 0; j < local_l; ++j)
        {
            int cij = 0; 
            for (int k = 0; k < M; ++k)
            {
                cij += a_mat[i * M + k] * b_mat[j * M + k];
            }
            local_C[i * local_l + j] = cij;
        }
    }

    const int TAG_C = 300;

    if (rank == 0)
    {
        for (int i = 0; i < local_n; ++i)
        {
            int global_row = row_start + i;
            int out_offset = global_row * L + col_start;
            for (int j = 0; j < local_l; ++j)
            {
                out_mat[out_offset + j] = local_C[i * local_l + j];
            }
        }
        delete[] local_C;

        for (int proc = 1; proc < size; ++proc)
        {
            int other_pr = proc / Pc;
            int other_pc = proc % Pc;

            int on, ors, ol, ocs;
            compute_block(N, Pr, other_pr, on,  ors);
            compute_block(L, Pc, other_pc, ol, ocs);

            int block_size = on * ol;
            int *buf = new int[block_size];

            MPI_Status st;
            MPI_Recv(buf, block_size, MPI_INT, proc, TAG_C, MPI_COMM_WORLD, &st);

            for (int i = 0; i < on; ++i)
            {
                int global_row = ors + i;
                int out_offset = global_row * L + ocs;
                for (int j = 0; j < ol; ++j)
                {
                    out_mat[out_offset + j] = buf[i * ol + j];
                }
            }

            delete[] buf;
        }
    }
    else
    {
        int block_size = local_n * local_l;
        MPI_Send(local_C, block_size, MPI_INT, 0, TAG_C, MPI_COMM_WORLD);
        delete[] local_C;
    }
}

void destruct_matrices(int *a_mat, int *b_mat)
{
    delete[] a_mat;
    delete[] b_mat;
}

