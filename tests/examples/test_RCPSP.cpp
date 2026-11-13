#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <vector>
#include "utils.hpp"

#define NO_MAIN
#include "RCPSP.cpp"


static void requireFeasible(RCPSPInstance const& instance, int32_t objective, std::span<int32_t const> solution) {
    REQUIRE(solution.size() == instance.n);
    REQUIRE(solution[0] == 0);
    REQUIRE(solution[instance.n - 1] == objective);

    // Check precedence constraints: s[u] + d[u] <= s[v]
    for (size_t k = 0; k < instance.pred.size(); k++) {
        uint32_t u = instance.pred[k];
        uint32_t v = instance.succ[k];
        REQUIRE(solution[u] >= 0);
        REQUIRE(solution[u] + instance.predDur[k] <= solution[v]);
    }

    // Check cumulative resource capacity at every time point t in [0, objective[
    for (uint32_t r = 0; r < instance.nResources; r++) {
        for (int32_t t = 0; t < objective; t++) {
            int32_t load = 0;
            for (uint32_t i = 0; i < instance.n; i++) {
                if (instance.durations[i] > 0 && solution[i] <= t && t < solution[i] + instance.durations[i]) {
                    load += instance.demands[r][i];
                }
            }
            REQUIRE(load <= instance.capacities[r]);
        }
    }
}


TEST_CASE("RCPSP feasible random instances", "[RCPSP][example]") {
    for (uint32_t nAct = 6; nAct <= 14; nAct += 2) {
        RCPSPInstance instance(nAct, 2, nAct);
        auto solver = getRCPSPSolver(instance, testParams);
        solver.solve([](SearchStatistics stats) { return stats.nSolutions > 0 && stats.nNodes >= 2000; });

        REQUIRE(solver.getStatistics().nSolutions > 0);
        requireFeasible(instance, solver.getStatistics().objective, solver.getSolution());
    }
}

TEST_CASE("RCPSP benchmark rcpsp_10 optimal", "[RCPSP][example]") {
    RCPSPInstance instance("../../data/RCPSP/rcpsp_10.rcp");
    auto solver = getRCPSPSolver(instance, testParams);
    solver.solve(noLimit);
    SearchStatistics stats = solver.getStatistics();

    REQUIRE(stats.completed);
    REQUIRE(stats.nSolutions == 2);
    REQUIRE(stats.nNodes == 41);
    REQUIRE(stats.nFailures == 19);
    requireFeasible(instance, stats.objective, solver.getSolution());
    REQUIRE(stats.objective == 19);
}

TEST_CASE("RCPSP PSPLib j30_1_1 optimal", "[RCPSP][example]") {
    RCPSPInstance instance("../../data/RCPSP/j30_1_1.rcp");
    auto solver = getRCPSPSolver(instance, testParams);
    solver.solve(noLimit);
    SearchStatistics stats = solver.getStatistics();

    REQUIRE(stats.completed);
    REQUIRE(stats.nSolutions == 8);
    REQUIRE(stats.nNodes == 347);
    REQUIRE(stats.nFailures == 166);
    requireFeasible(instance, stats.objective, solver.getSolution());
    REQUIRE(stats.objective == 43);
}
