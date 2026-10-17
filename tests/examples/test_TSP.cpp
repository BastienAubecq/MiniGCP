#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <vector>
#include "utils.hpp"

#define NO_MAIN
#include "TSP.cpp"


static void requireFeasible(TSPInstance const& instance, int32_t objective, std::span<int32_t const> solution) {
    REQUIRE(solution.size() == instance.n);
    vector<bool> visited(instance.n, false);
    uint32_t curr = 0;
    int32_t cost = 0;
    for (uint32_t i = 0; i < instance.n; i++) {
        uint32_t next = solution[curr];
        REQUIRE(next >= 0);
        REQUIRE(next < instance.n);
        REQUIRE_FALSE(visited[next]);
        visited[next] = true;
        cost += instance.distance(curr, next);
        curr = next;
    }
    REQUIRE(objective == cost);
}


TEST_CASE("TSP feasible", "[TSP][example]") {
    for (uint32_t n = 10; n <= 100; n += 5) {
        TSPInstance instance(n, 100);
        auto solver = getSolver(instance, testParams);
        solver.solve([](SearchStatistics stats) { return stats.nSolutions > 0 && stats.nNodes >= 10000; });

        REQUIRE(solver.getStatistics().nSolutions > 0);
        requireFeasible(instance, solver.getStatistics().objective, solver.getSolution());
    }
}

TEST_CASE("TSP statistics", "[TSP][example]") {
    TSPInstance instance("../../data/TSP/tsp_15.txt");
    auto solver = getSolver(instance, testParams);
    solver.solve(noLimit);
    SearchStatistics stats = solver.getStatistics();

    REQUIRE(stats.nSolutions == 64);
    REQUIRE(stats.nNodes == 7033);
    REQUIRE(stats.nFailures == 3453);
    REQUIRE(stats.objective == 291);
    REQUIRE(stats.completed);
}
