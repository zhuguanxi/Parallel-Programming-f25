// pi.cpp — fast, pthreads, contention-free, thread-local RNG
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#ifndef CACHELINE
#define CACHELINE 64
#endif
struct rng64_t {
    uint64_t s; 
};

static inline uint64_t splitmix64(uint64_t &x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t xorshift64star(rng64_t &st) {
    uint64_t x = st.s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    st.s = x;
    return x * 2685821657736338717ULL;
}

static inline double urand01(rng64_t &st) {
    return (xorshift64star(st) >> 11) * (1.0 / 9007199254740992.0); // 2^53
}

struct alignas(CACHELINE) ThreadCtx {
    uint64_t tosses;
    uint64_t hits;
    int      tid;
    uint64_t seed_base;
    char     pad[CACHELINE - (sizeof(uint64_t)*3 + sizeof(int))];
};

static int       g_num_threads = 0;
static uint64_t  g_num_tosses  = 0;
static ThreadCtx *g_ctxs       = nullptr;
static pthread_t *g_threads    = nullptr;

static void* worker(void* arg) {
    ThreadCtx* ctx = (ThreadCtx*)arg;

    uint64_t seed = ctx->seed_base ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(ctx->tid + 1));
    rng64_t rng { .s = splitmix64(seed) | 1ULL }; 
    uint64_t local_hits = 0;
    uint64_t n = ctx->tosses;

    for (uint64_t i = 0; i < n; ++i) {
        double x = urand01(rng);
        double y = urand01(rng);
        double d2 = x*x + y*y;
        local_hits += (d2 <= 1.0);
    }

    ctx->hits = local_hits;
    return NULL;
}

int main(int argc, char** argv) {
    if (argc != 3) return 0;

    g_num_threads = atoi(argv[1]);
    if (g_num_threads <= 0) return 0;

    char* endp = nullptr;
    g_num_tosses = strtoull(argv[2], &endp, 10);
    if (endp == argv[2] || g_num_tosses == 0) return 0;

    if (posix_memalign((void**)&g_ctxs, CACHELINE, sizeof(ThreadCtx) * (size_t)g_num_threads) != 0)
        return 0;
    g_threads = (pthread_t*)malloc(sizeof(pthread_t) * (size_t)g_num_threads);
    if (!g_threads) { free(g_ctxs); return 0; }

    uint64_t base = g_num_tosses / (uint64_t)g_num_threads;
    uint64_t rem  = g_num_tosses % (uint64_t)g_num_threads;

    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t seed_base = ((uint64_t)ts.tv_nsec << 32) ^ (uint64_t)ts.tv_sec ^ (uint64_t)getpid();

    for (int t = 0; t < g_num_threads; ++t) {
        g_ctxs[t].tid       = t;
        g_ctxs[t].tosses    = base + (t < (int)rem ? 1 : 0);
        g_ctxs[t].seed_base = seed_base;
        g_ctxs[t].hits      = 0;

        int rc = pthread_create(&g_threads[t], NULL, worker, &g_ctxs[t]);
        if (rc) {
            for (int j = 0; j < t; ++j) pthread_join(g_threads[j], NULL);
            free(g_threads);
            free(g_ctxs);
            return 0;
        }
    }

    uint64_t total_hits = 0;
    for (int t = 0; t < g_num_threads; ++t) {
        pthread_join(g_threads[t], NULL);
        total_hits += g_ctxs[t].hits;
    }

    free(g_threads);
    free(g_ctxs);
    double pi = 4.0 * (double)total_hits / (double)g_num_tosses;
    printf("%.7f\n", pi);
    return 0;
}

