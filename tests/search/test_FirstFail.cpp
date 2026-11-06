#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/search/FirstFail.hpp"
#include "MiniGCP/state/Array.hpp"
#include "MiniGCP/state/IntBitset.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;

TEST_CASE("FirstFail unfixed variable", "[FirstFail]") {
    Allocator allocator;
    IntBitset fixed(allocator, 1);
    IntBitset selected(allocator, 2);
    vector<IntBitset> variables{fixed, selected};
    FirstFail<Array<IntBitset>> strategy(Array<IntBitset>(allocator, variables));

    bool d = runOnDevice<char>(allocator, 1, [=](Context ctx) {
        auto decision = strategy.split(ctx);
        if (!decision) return false;
        decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(selected[allocator].value() == 0);
}

TEST_CASE("FirstFail smallest domain", "[FirstFail]") {
    Allocator allocator;
    IntBitset large(allocator, 10);
    IntBitset small(allocator, 3);
    vector<IntBitset> variables{large, small};
    FirstFail strategy(Array<IntBitset>(allocator, variables));

    bool d = runOnDevice<char>(allocator, 1, [=](Context ctx) {
        auto decision = strategy.split(ctx);
        if (!decision) return false;
        decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(small[allocator].value() == 0);
}

TEST_CASE("FirstFail all fixed", "[FirstFail]") {
    Allocator allocator;
    IntBitset first(allocator, 1);
    IntBitset second(allocator, 1);
    vector<IntBitset> variables{first, second};
    FirstFail strategy(Array<IntBitset>(allocator, variables));

    bool d = runOnDevice<char>(allocator, 1, [=](Context ctx) {
        return strategy.split(ctx).has_value();
    })[0];

    REQUIRE_FALSE(d);
}

TEST_CASE("FirstFail concurrent few variables", "[FirstFail][concurrent]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 10);
    variables[5][allocator].removeAbove(5);
    FirstFail strategy(variables);

    bool d = runOnDevice<char>(allocator, 16, [=](Context ctx) {
        auto decision = strategy.split(ctx);
        if (!decision) return false;
        if (ctx.item.get_local_id(0) == 0) decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(variables[5][allocator].value() == 0);
}

TEST_CASE("FirstFail concurrent many variables", "[FirstFail][concurrent]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 100);
    variables[5][allocator].removeAbove(5);
    FirstFail strategy(variables);

    bool d = runOnDevice<char>(allocator, 16, [=](Context ctx) {
        auto decision = strategy.split(ctx);
        if (!decision) return false;
        decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(variables[5][allocator].value() == 0);
}