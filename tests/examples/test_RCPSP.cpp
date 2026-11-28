#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <vector>
#include "utils.hpp"

#define NO_MAIN
#include "RCPSP.cpp"


static void requireFeasible(RCPSP const& rcpsp, int32_t objective, std::span<int32_t const> solution) {
    REQUIRE(solution.size() == rcpsp.nActivities);
    REQUIRE(solution[0] == 0);
    int32_t m = 0;
    for (uint32_t i = 0; i < rcpsp.nActivities; i++) {
        REQUIRE(solution[i] >= 0);
        m = max(m, solution[i]);
    }
    REQUIRE(m == objective);

    // Check precedence constraints: s[u] + d[u] <= s[v]
    for (size_t k = 0; k < rcpsp.pred.size(); k++) {
        uint32_t u = rcpsp.pred[k];
        uint32_t v = rcpsp.succ[k];
        REQUIRE(solution[u] + rcpsp.durations[u] <= solution[v]);
    }

    // Check cumulative resource capacity at every time point t in [0, objective[
    for (uint32_t r = 0; r < rcpsp.nResources; r++) {
        for (int32_t t = 0; t < objective; t++) {
            int32_t load = 0;
            for (uint32_t i = 0; i < rcpsp.nActivities; i++) {
                if (rcpsp.durations[i] > 0 && solution[i] <= t && t < solution[i] + rcpsp.durations[i]) {
                    load += rcpsp.consumption(i, r);
                }
            }
            REQUIRE(load <= rcpsp.capacities[r]);
        }
    }
}


TEST_CASE("RCPSP feasible random instances", "[RCPSP][example]") {
    for (uint32_t nAct = 10; nAct <= 50; nAct += 10) {
        RCPSP rcpsp(nAct, 4, nAct);
        auto solver = rcpsp.solver(testParams);
        solver.solve([](SearchStatistics stats) { return stats.nSolutions > 0 && stats.nNodes >= 1000; });

        REQUIRE(solver.getStatistics().nSolutions > 0);
        requireFeasible(rcpsp, solver.getStatistics().objective, solver.getSolution());
    }
}

TEST_CASE("RCPSP benchmark rcpsp_10 optimal", "[RCPSP][example]") {
    RCPSP rcpsp("../../data/RCPSP/rcpsp_10.rcp");
    auto solver = rcpsp.solver(testParams);
    solver.solve(noLimit);
    SearchStatistics stats = solver.getStatistics();

    REQUIRE(stats.nSolutions == 2);
    REQUIRE(stats.nNodes == 43);
    REQUIRE(stats.nFailures == 20);
    REQUIRE(stats.objective == 19);
    REQUIRE(stats.completed);
    requireFeasible(rcpsp, stats.objective, solver.getSolution());
}

TEST_CASE("RCPSP PSPLib j30_1_1 optimal", "[RCPSP][example]") {
    RCPSP rcpsp("../../data/RCPSP/j30_1_1.rcp");
    auto solver = rcpsp.solver(testParams);
    solver.solve(noLimit);
    SearchStatistics stats = solver.getStatistics();

    REQUIRE(stats.nSolutions == 7);
    REQUIRE(stats.nNodes == 291);
    REQUIRE(stats.nFailures == 139);
    REQUIRE(stats.objective == 43);
    REQUIRE(stats.completed);
    requireFeasible(rcpsp, stats.objective, solver.getSolution());
}


TEST_CASE("RCPSP_binary feasible random instances", "[RCPSP_binary][example]") {
    for (uint32_t nAct = 10; nAct <= 50; nAct += 10) {
        RCPSP rcpsp(nAct, 4, nAct);
        auto solver = rcpsp.binarySolver(testParams);
        solver.solve([](SearchStatistics stats) { return stats.nSolutions > 0 && stats.nNodes >= 1000; });

        REQUIRE(solver.getStatistics().nSolutions > 0);
        requireFeasible(rcpsp, solver.getStatistics().objective, solver.getSolution());
    }
}

TEST_CASE("RCPSP_binary benchmark rcpsp_10 optimal", "[RCPSP_binary][example]") {
    RCPSP rcpsp("../../data/RCPSP/rcpsp_10.rcp");
    auto solver = rcpsp.binarySolver(testParams);
    solver.solve(noLimit);
    SearchStatistics stats = solver.getStatistics();

    REQUIRE(stats.nSolutions == 2);
    REQUIRE(stats.nNodes == 43);
    REQUIRE(stats.nFailures == 20);
    REQUIRE(stats.objective == 19);
    REQUIRE(stats.completed);
    requireFeasible(rcpsp, stats.objective, solver.getSolution());
}

TEST_CASE("RCPSP_binary PSPLib j30_1_1 optimal", "[RCPSP_binary][example]") {
    RCPSP rcpsp("../../data/RCPSP/j30_1_1.rcp");
    auto solver = rcpsp.binarySolver(testParams);
    solver.solve(noLimit);
    SearchStatistics stats = solver.getStatistics();

    REQUIRE(stats.nSolutions == 7);
    REQUIRE(stats.nNodes == 291);
    REQUIRE(stats.nFailures == 139);
    REQUIRE(stats.objective == 43);
    REQUIRE(stats.completed);
    requireFeasible(rcpsp, stats.objective, solver.getSolution());
}