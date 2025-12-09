#include "page_rank.h"

#include <cmath>
#include <cstdlib>
#include <omp.h>

#include "../common/graph.h"

// page_rank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void page_rank(Graph g, double *solution, double damping, double convergence)
{              
    int N = g->num_nodes;
    if (N == 0) return;

    // allocate two buffers: old score & new score
    double *score_old = new double[N];
    double *score_new = new double[N];

    // init: uniform prob
    double init_val = 1.0 / (double)N;
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        score_old[i] = init_val;
        score_new[i] = 0.0;
    }

    // precompute out-degree for each node，免得每次重算
    int *outdeg = new int[N];
    #pragma omp parallel for
    for (int v = 0; v < N; v++) {
        int start = g->outgoing_starts[v];
        int end = (v == N - 1) ? g->num_edges : g->outgoing_starts[v + 1];
        outdeg[v] = end - start;
    }

    bool converged = false;

    // 反覆疊代
    while (!converged)
    {
        // 1. 計算 dangling_mass = sum of score_old[v] where outdeg[v] == 0
        double dangling_mass = 0.0;
        #pragma omp parallel for reduction(+ : dangling_mass)
        for (int v = 0; v < N; v++) {
            if (outdeg[v] == 0) {
                dangling_mass += score_old[v];
            }
        }

        double base = (1.0 - damping) / (double)N;
        double dangling_term = damping * dangling_mass / (double)N;

        // 2. 計算新的 PageRank 對每個節點 i
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            // sum over incoming neighbors j of i
            int in_start = g->incoming_starts[i];
            int in_end = (i == N - 1) ? g->num_edges : g->incoming_starts[i + 1];

            double incoming_sum = 0.0;
            for (int e = in_start; e < in_end; e++) {
                int j = g->incoming_edges[e]; // j -> i
                int deg_j = outdeg[j];
                if (deg_j > 0) {
                    incoming_sum += score_old[j] / (double)deg_j;
                }
                // if deg_j == 0, 那個 j 已經被放進 dangling_mass 了
            }

            // PR_new[i] = base + d * incoming_sum + d * dangling_mass/N
            score_new[i] = base + damping * incoming_sum + dangling_term;
        }

        // 3. 檢查收斂: global_diff = Σ |new - old|
        double global_diff = 0.0;
        #pragma omp parallel for reduction(+ : global_diff)
        for (int i = 0; i < N; i++) {
            global_diff += fabs(score_new[i] - score_old[i]);
        }

        if (global_diff < convergence) {
            converged = true;
        }

        // 4. swap: 把 new 複製到 old，準備下一輪
        #pragma omp parallel for
        for (int i = 0; i < N; i++) {
            score_old[i] = score_new[i];
        }
    }

    // 把結果丟回 solution
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        solution[i] = score_old[i];
    }

    delete[] score_old;
    delete[] score_new;
    delete[] outdeg;	
        /*
       For PP students: Implement the page rank algorithm here.  You
       are expected to parallelize the algorithm using openMP.  Your
       solution may need to allocate (and free) temporary arrays.

       Basic page rank pseudocode is provided below to get you started:

       // initialization: see example code above
       score_old[vi] = 1/nnodes;

       while (!converged) {

         // compute score_new[vi] for all nodes vi:
         score_new[vi] = sum over all nodes vj reachable from incoming edges
                            { score_old[vj] / number of edges leaving vj  }
         score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / nnodes;

         score_new[vi] += sum over all nodes v in graph with no outgoing edges
                            { damping * score_old[v] / nnodes }

         // compute how much per-node scores have changed
         // quit once algorithm has converged

         global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
         converged = (global_diff < convergence)
       }

     */
}
