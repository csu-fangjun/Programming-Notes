#include <unistd.h>  // for usleep

#include "benchmark/benchmark.h"

namespace {

int test() {
  // usleep(10 * 1000);  // sleep 10ms
}

}  // namespace

static void BM_hello(benchmark::State& state) {
  for (auto _ : state) {
    test();
  }
}

BENCHMARK(BM_hello);

BENCHMARK_MAIN();

#if 0
-----------------------------------------------------
Benchmark           Time             CPU   Iterations
-----------------------------------------------------
BM_hello     10101604 ns        15494 ns         1000
#endif
