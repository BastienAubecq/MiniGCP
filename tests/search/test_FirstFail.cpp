#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/search/FirstFail.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;

TEST_CASE("FirstFail unfixed variable", "[FirstFail]") {
    Allocator allocator;
    IntVarIdx fixed = allocator.variable(1);
    IntVarIdx selected = allocator.variable(2);
    vector<IntVarIdx> variables{fixed, selected};
    FirstFail strategy(allocator.array<Scope::global, IntVarIdx>(variables));

    bool d = runOnDevice<char>(allocator, 1, [=](Context ctx) {
        optional<FixRemove> decision = strategy.split(ctx);
        if (!decision) return false;
        decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(allocator[selected].fixed());
}

TEST_CASE("FirstFail smallest domain", "[FirstFail]") {
    Allocator allocator;
    IntVarIdx large = allocator.variable(10);
    IntVarIdx small = allocator.variable(3);
    vector<IntVarIdx> variables{large, small};
    FirstFail strategy(allocator.array<Scope::global, IntVarIdx>(variables));

    bool d = runOnDevice<char>(allocator, 1, [=](Context ctx) {
        optional<FixRemove> decision = strategy.split(ctx);
        if (!decision) return false;
        decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(allocator[small].fixed());
}

TEST_CASE("FirstFail all fixed", "[FirstFail]") {
    Allocator allocator;
    IntVarIdx first = allocator.variable(1);
    IntVarIdx second = allocator.variable(1);
    vector<IntVarIdx> variables{first, second};
    FirstFail strategy(allocator.array<Scope::global, IntVarIdx>(variables));

    bool d = runOnDevice<char>(allocator, 1, [=](Context ctx) {
        return strategy.split(ctx).has_value();
    })[0];

    REQUIRE_FALSE(d);
}

TEST_CASE("FirstFail concurrent few variables", "[FirstFail][concurrent]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[5]].removeBelow(5);
    FirstFail strategy(allocator.array<Scope::global, IntVarIdx>(variables));

    bool d = runOnDevice<char>(allocator, 16, [=](Context ctx) {
        optional<FixRemove> decision = strategy.split(ctx);
        if (!decision) return false;
        if (ctx.item.get_local_id(0) == 0) decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(allocator[variables[5]].fixed());
}

TEST_CASE("FirstFail concurrent many variables", "[FirstFail][concurrent]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 100; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[5]].removeBelow(5);
    FirstFail strategy(allocator.array<Scope::global, IntVarIdx>(variables));

    bool d = runOnDevice<char>(allocator, 16, [=](Context ctx) {
        optional<FixRemove> decision = strategy.split(ctx);
        if (!decision) return false;
        decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(allocator[variables[5]].fixed());
}