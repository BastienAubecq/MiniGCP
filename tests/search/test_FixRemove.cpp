#include <catch2/catch_test_macros.hpp>
#include "MiniGCP/search/FixRemove.hpp"
#include "MiniGCP/state/IntBitset.hpp"
#include "utils.hpp"

using namespace cp;

TEST_CASE("FixRemove fix branch", "[FixRemove]") {
    Allocator allocator;
    IntBitset x(allocator, 5);
    FixRemove decision(x, 3);

    runOnDevice(allocator, 1, [=](Context ctx) {
        decision.applyLeft(ctx);
    });

    REQUIRE(x[allocator].size() == 1);
    REQUIRE(x[allocator].min() == 3);
}

TEST_CASE("FixRemove remove branch", "[FixRemove]") {
    Allocator allocator;
    IntBitset x(allocator, 5);
    FixRemove decision(x, 3);

    runOnDevice(allocator, 1, [=](Context ctx) {
        decision.applyRight(ctx);
    });

    REQUIRE(x[allocator].size() == 4);
    REQUIRE_FALSE(x[allocator].contains(3));
}