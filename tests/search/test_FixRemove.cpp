#include <catch2/catch_test_macros.hpp>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/search/FixRemove.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "utils.hpp"

using namespace cp;

TEST_CASE("FixRemove fix branch", "[FixRemove]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(5);
    FixRemove decision(idx, 3);

    runOnDevice(allocator, 1, [=](Context ctx) {
        decision.applyLeft(ctx);
    });

    REQUIRE(allocator[idx].size() == 1);
    REQUIRE(allocator[idx].min() == 3);
}

TEST_CASE("FixRemove remove branch", "[FixRemove]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(5);
    FixRemove decision(idx, 3);

    runOnDevice(allocator, 1, [=](Context ctx) {
        decision.applyRight(ctx);
    });

    REQUIRE(allocator[idx].size() == 4);
    REQUIRE_FALSE(allocator[idx].contains(3));
}