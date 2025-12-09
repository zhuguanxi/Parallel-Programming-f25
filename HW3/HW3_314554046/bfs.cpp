#include "bfs.h"

#include <cstdlib>
#include <omp.h>
#include <vector>
#include <unordered_set>

#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif // VERBOSE
#define ALPHA 14
constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.

void top_down_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances)
{
    #pragma omp parallel for
    for (int i = 0; i < frontier->count; i++)
    {
        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

	    if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
                {
                    int index;
                    #pragma omp atomic capture
                    index = new_frontier->count++;

                    new_frontier->vertices[index] = outgoing;
                }
            }

        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    //

    const int num_nodes = graph->num_nodes;

    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
    {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    sol->distances[ROOT_NODE_ID] = 0;

    int level = 0;

    bool level_added = true;

    while (level_added)
    {
        level_added = false;

        int local_added_count = 0;

        #pragma omp parallel for reduction(|| : level_added) reduction(+ : local_added_count) schedule(dynamic, 1024)
        for (int u = 0; u < num_nodes; u++)
        {
            if (sol->distances[u] != NOT_VISITED_MARKER)
                continue;

            int start_edge = graph->incoming_starts[u];
            int end_edge = (u == num_nodes - 1) ? graph->num_edges : graph->incoming_starts[u + 1];

            for (int e = start_edge; e < end_edge; e++)
            {
                int v = graph->incoming_edges[e];

                if (sol->distances[v] == level)
                {
                    sol->distances[u] = level + 1;

                    level_added = true;
                    local_added_count++;
                    break;
                }
            }
       }
        if (!level_added)
            break;

        level++;
    }
}   
void bfs_hybrid(Graph graph, solution *sol)
{
	const int num_nodes = graph->num_nodes;

    #pragma omp parallel for
    for (int i = 0; i < num_nodes; i++)
    {
        sol->distances[i] = NOT_VISITED_MARKER;
    }
    sol->distances[ROOT_NODE_ID] = 0;

    std::vector<int> frontier_list;
    frontier_list.reserve(num_nodes);
    frontier_list.push_back(ROOT_NODE_ID);

    int level = 0;
    int unvisited_count = num_nodes - 1;

    const int HYBRID_ALPHA = 24;

    while (!frontier_list.empty())
    {
        std::vector<int> next_frontier_list;
        next_frontier_list.reserve(num_nodes);

        int frontier_size = (int)frontier_list.size();

        bool use_bottom_up = false;
        if (unvisited_count > 0)
        {
            if (frontier_size > unvisited_count / HYBRID_ALPHA)
                use_bottom_up = true;
        }

        if (!use_bottom_up)
        {
            int newly_visited_this_level = 0;

            #pragma omp parallel
            {
                std::vector<int> local_next;
                local_next.reserve(1024);

                int local_new_count = 0;

                #pragma omp for schedule(dynamic, 1024) nowait
                for (int i = 0; i < frontier_size; i++)
                {
                    int node = frontier_list[i];

                    int start_edge = graph->outgoing_starts[node];
                    int end_edge = (node == num_nodes - 1)
                                        ? graph->num_edges
                                        : graph->outgoing_starts[node + 1];

                    for (int e = start_edge; e < end_edge; e++)
                    {
                        int ngh = graph->outgoing_edges[e];

                        if (sol->distances[ngh] == NOT_VISITED_MARKER)
                        {
                            bool claimed = false;
                            #pragma omp critical
                            {
                                if (sol->distances[ngh] == NOT_VISITED_MARKER)
                                {
                                    sol->distances[ngh] = level + 1;
                                    claimed = true;
                                }
                            }
                            if (claimed)
                            {
                                local_next.push_back(ngh);
                                local_new_count++;
                            }
                        }
                    }
                }
                #pragma omp critical
                {
                    newly_visited_this_level += local_new_count;
                    next_frontier_list.insert(
                        next_frontier_list.end(),
                        local_next.begin(),
                        local_next.end()
                    );
                }
            } 

            unvisited_count -= newly_visited_this_level;
        }
        else
        {
            int newly_visited_this_level = 0;

            #pragma omp parallel
            {
                std::vector<int> local_next;
                local_next.reserve(1024);

                int local_new_count = 0;

                #pragma omp for schedule(dynamic, 1024) nowait
                for (int u = 0; u < num_nodes; u++)
                {
                    if (sol->distances[u] != NOT_VISITED_MARKER)
                        continue;

                    int start_edge = graph->incoming_starts[u];
                    int end_edge = (u == num_nodes - 1)
                                        ? graph->num_edges
                                        : graph->incoming_starts[u + 1];

                    for (int e = start_edge; e < end_edge; e++)
                    {
                        int v = graph->incoming_edges[e];

                        if (sol->distances[v] == level)
                        {
                            sol->distances[u] = level + 1;
                            local_next.push_back(u);
                            local_new_count++;
                            break;
                        }
                    }
                }

                #pragma omp critical
                {
                    newly_visited_this_level += local_new_count;
                    next_frontier_list.insert(
                        next_frontier_list.end(),
                        local_next.begin(),
                        local_next.end()
                    );
                }
            } 

            unvisited_count -= newly_visited_this_level;
        }

        if (next_frontier_list.empty())
        {
            break;
        }

        frontier_list.swap(next_frontier_list);
        level++;
    }
}
