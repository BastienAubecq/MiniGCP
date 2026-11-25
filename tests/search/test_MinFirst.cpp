#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/search/MinFirst.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "MiniGCP/state/Ptr.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;

TEST_CASE("MinFirst unfixed variable", "[MinFirst]") {
    Allocator allocator;
    IntInterval fixed(allocator, 1);
    IntInterval selected(allocator, 2);
    vector<IntInterval> variables{fixed, selected};
    MinFirst<Ptr<IntInterval>> strategy(Ptr<IntInterval>(allocator, variables), 2);

    bool d = runOnDevice<char>(allocator, 1, [=](Context ctx) {
        auto decision = strategy.split(ctx);
        if (!decision) return false;
        decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(selected[allocator].value() == 0);
}

TEST_CASE("MinFirst smallest value", "[MinFirst]") {
    Allocator allocator;
    IntInterval large(allocator, 10, 20);
    IntInterval small(allocator, 3, 25);
    vector<IntInterval> variables{large, small};
    MinFirst strategy(Ptr<IntInterval>(allocator, variables), 2);

    bool d = runOnDevice<char>(allocator, 1, [=](Context ctx) {
        auto decision = strategy.split(ctx);
        if (!decision) return false;
        decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(small[allocator].value() == 3);
}

TEST_CASE("MinFirst all fixed", "[MinFirst]") {
    Allocator allocator;
    IntInterval first(allocator, 1);
    IntInterval second(allocator, 1);
    vector<IntInterval> variables{first, second};
    MinFirst strategy(Ptr<IntInterval>(allocator, variables), 2);

    bool d = runOnDevice<char>(allocator, 1, [=](Context ctx) {
        return strategy.split(ctx).has_value();
    })[0];

    REQUIRE_FALSE(d);
}

TEST_CASE("MinFirst concurrent few variables", "[MinFirst][concurrent]") {
    Allocator allocator;
    IntIntervalPtr variables(allocator, 10, 10);
    for (int i = 0; i < 10; i++) {
        if (i != 5) variables[i][allocator].removeBelow(5);
    }
    MinFirst strategy(variables, 10);

    bool d = runOnDevice<char>(allocator, 16, [=](Context ctx) {
        auto decision = strategy.split(ctx);
        if (!decision) return false;
        if (ctx.item.get_local_id(0) == 0) decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(variables[5][allocator].value() == 0);
}

TEST_CASE("MinFirst concurrent many variables", "[MinFirst][concurrent]") {
    Allocator allocator;
    IntIntervalPtr variables(allocator, 10, 100);
    for (int i = 0; i < 100; i++) {
        if (i != 5) variables[i][allocator].removeBelow(5);
    }
    MinFirst strategy(variables, 100);

    bool d = runOnDevice<char>(allocator, 16, [=](Context ctx) {
        auto decision = strategy.split(ctx);
        if (!decision) return false;
        decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(variables[5][allocator].value() == 0);
}