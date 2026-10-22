#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/search/FirstFail.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;

TEST_CASE("FirstFail unfixed variable", "[FirstFail]") {
    Allocator allocator;
    IntVarBitsetIdx fixed = allocator.intVarBitset(1);
    IntVarBitsetIdx selected = allocator.intVarBitset(2);
    vector<IntVarBitsetIdx> variables{fixed, selected};
    FirstFail strategy(allocator.array<Scope::global, IntVarBitsetIdx>(variables));

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
    IntVarBitsetIdx large = allocator.intVarBitset(10);
    IntVarBitsetIdx small = allocator.intVarBitset(3);
    vector<IntVarBitsetIdx> variables{large, small};
    FirstFail strategy(allocator.array<Scope::global, IntVarBitsetIdx>(variables));

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
    IntVarBitsetIdx first = allocator.intVarBitset(1);
    IntVarBitsetIdx second = allocator.intVarBitset(1);
    vector<IntVarBitsetIdx> variables{first, second};
    FirstFail strategy(allocator.array<Scope::global, IntVarBitsetIdx>(variables));

    bool d = runOnDevice<char>(allocator, 1, [=](Context ctx) {
        return strategy.split(ctx).has_value();
    })[0];

    REQUIRE_FALSE(d);
}

TEST_CASE("FirstFail concurrent few variables", "[FirstFail][concurrent]") {
    Allocator allocator;
    vector<IntVarBitsetIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.intVarBitset(10));
    }
    allocator[variables[5]].removeBelow(5);
    FirstFail strategy(allocator.array<Scope::global, IntVarBitsetIdx>(variables));

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
    vector<IntVarBitsetIdx> variables;
    for (int i = 0; i < 100; i++) {
        variables.push_back(allocator.intVarBitset(10));
    }
    allocator[variables[5]].removeBelow(5);
    FirstFail strategy(allocator.array<Scope::global, IntVarBitsetIdx>(variables));

    bool d = runOnDevice<char>(allocator, 16, [=](Context ctx) {
        optional<FixRemove> decision = strategy.split(ctx);
        if (!decision) return false;
        decision->applyLeft(ctx);
        return true;
    })[0];

    REQUIRE(d);
    REQUIRE(allocator[variables[5]].fixed());
}