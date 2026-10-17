#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <cstdlib>
#include "utils.hpp"

#define NO_MAIN
#include "NQueens.cpp"


static void requireFeasible(uint32_t n, span<int32_t const> solution) {
    REQUIRE(solution.size() == n);
    for (uint32_t i = 0; i < n; i++) {
        REQUIRE(solution[i] >= 0);
        REQUIRE(solution[i] < n);
        for (uint32_t j = i + 1; j < n; j++) {
            REQUIRE(solution[i] != solution[j]);
            REQUIRE(std::abs(solution[i] - solution[j]) != static_cast<int32_t>(j - i));
        }
    }
}


TEST_CASE("NQueens feasible", "[NQueens][example]") {
    for (uint32_t n = 10; n <= 100; n += 5) {
        auto solver = getSolver(n, testParams);
        solver.solve(firstSolution);

        REQUIRE(solver.getStatistics().nSolutions == 1);
        requireFeasible(n, solver.getSolution());
    }
}

TEST_CASE("NQueens statistics", "[NQueens][example]") {
    auto solver = getSolver(12, testParams);
    solver.solve(noLimit);
    SearchStatistics stats = solver.getStatistics();

    REQUIRE(stats.nSolutions == 14200);
    REQUIRE(stats.nNodes == 232163);
    REQUIRE(stats.nFailures == 101882);
    REQUIRE(stats.completed);
}
