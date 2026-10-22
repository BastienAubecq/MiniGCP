#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/constraints/Circuit.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;


TEST_CASE("CircuitSeq one cycle", "[CircuitSeq]") {
    Allocator allocator;
    vector<IntVarBitsetIdx> x;
    for (int i = 0; i < 10; i++) {
        x.push_back(allocator.intVarBitset(10));
    }
    allocator[x[3]].fix(5);
    allocator[x[5]].fix(7);
    allocator[x[7]].fix(2);
    allocator[x[2]].fix(8);
    CircuitSeq constraint(allocator, allocator.array<Scope::global, IntVarBitsetIdx>(x));

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(allocator[x[8]].contains(3));
    REQUIRE(allocator[x[0]].size() == 9);
    REQUIRE(allocator[x[1]].size() == 9);
    REQUIRE(allocator[x[4]].size() == 9);
    REQUIRE(allocator[x[6]].size() == 9);
    REQUIRE(allocator[x[9]].size() == 9);
}

TEST_CASE("CircuitSeq few cycles", "[CircuitSeq]") {
    Allocator allocator;
    vector<IntVarBitsetIdx> x;
    for (int i = 0; i < 10; i++) {
        x.push_back(allocator.intVarBitset(10));
    }
    allocator[x[3]].fix(5);
    allocator[x[0]].fix(1);
    allocator[x[1]].fix(9);
    allocator[x[8]].fix(2);
    allocator[x[2]].fix(4);
    allocator[x[4]].fix(7);
    CircuitSeq constraint(allocator, allocator.array<Scope::global, IntVarBitsetIdx>(x));

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(allocator[x[5]].contains(3));
    REQUIRE_FALSE(allocator[x[9]].contains(0));
    REQUIRE_FALSE(allocator[x[7]].contains(8));
    REQUIRE(allocator[x[6]].size() == 9);
}

TEST_CASE("CircuitSeq full cycle", "[CircuitSeq]") {
    Allocator allocator;
    vector<IntVarBitsetIdx> x;
    for (int i = 0; i < 9; i++) {
        x.push_back(allocator.intVarBitset(i, i + 1));
    }
    x.push_back(allocator.intVarBitset(10));
    CircuitSeq constraint(allocator, allocator.array<Scope::global, IntVarBitsetIdx>(x));

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}


TEST_CASE("Circuit one cycle", "[Circuit]") {
    Allocator allocator;
    vector<IntVarBitsetIdx> x;
    for (int i = 0; i < 10; i++) {
        x.push_back(allocator.intVarBitset(10));
    }
    allocator[x[3]].fix(5);
    allocator[x[5]].fix(7);
    allocator[x[7]].fix(2);
    allocator[x[2]].fix(8);
    Circuit constraint(allocator, allocator.array<Scope::global, IntVarBitsetIdx>(x));

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(allocator[x[8]].contains(3));
    REQUIRE(allocator[x[0]].size() == 9);
    REQUIRE(allocator[x[1]].size() == 9);
    REQUIRE(allocator[x[4]].size() == 9);
    REQUIRE(allocator[x[6]].size() == 9);
    REQUIRE(allocator[x[9]].size() == 9);
}

TEST_CASE("Circuit few cycles", "[Circuit]") {
    Allocator allocator;
    vector<IntVarBitsetIdx> x;
    for (int i = 0; i < 10; i++) {
        x.push_back(allocator.intVarBitset(10));
    }
    allocator[x[3]].fix(5);
    allocator[x[0]].fix(1);
    allocator[x[1]].fix(9);
    allocator[x[8]].fix(2);
    allocator[x[2]].fix(4);
    allocator[x[4]].fix(7);
    Circuit constraint(allocator, allocator.array<Scope::global, IntVarBitsetIdx>(x));

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(allocator[x[5]].contains(3));
    REQUIRE_FALSE(allocator[x[9]].contains(0));
    REQUIRE_FALSE(allocator[x[7]].contains(8));
    REQUIRE(allocator[x[6]].size() == 9);
}

TEST_CASE("Circuit full cycle", "[Circuit]") {
    Allocator allocator;
    vector<IntVarBitsetIdx> x;
    for (int i = 0; i < 9; i++) {
        x.push_back(allocator.intVarBitset(i, i + 1));
    }
    x.push_back(allocator.intVarBitset(10));
    Circuit constraint(allocator, allocator.array<Scope::global, IntVarBitsetIdx>(x));

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}


TEST_CASE("Circuit concurrent few variables", "[Circuit][concurrent]") {
    Allocator allocator;
    vector<IntVarBitsetIdx> x;
    for (int i = 0; i < 9; i++) {
        if (i % 2 == 0) x.push_back(allocator.intVarBitset(10));
        else x.push_back(allocator.intVarBitset(i + 2, i + 2));
    }
    x.push_back(allocator.intVarBitset(10));
    Circuit constraint(allocator, allocator.array<Scope::global, IntVarBitsetIdx>(x));

    DomainChange c = runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 9; i++) {
        if (i % 2 == 0) REQUIRE(allocator[x[i]].size() == 9);
    }
    REQUIRE_FALSE(allocator[x[9]].contains(1));
}

TEST_CASE("Circuit concurrent many variables", "[Circuit][concurrent]") {
    Allocator allocator;
    vector<IntVarBitsetIdx> x;
    for (int i = 0; i < 99; i++) {
        if (i % 2 == 0) x.push_back(allocator.intVarBitset(100));
        else x.push_back(allocator.intVarBitset(i + 2, i + 2));
    }
    x.push_back(allocator.intVarBitset(100));
    Circuit constraint(allocator, allocator.array<Scope::global, IntVarBitsetIdx>(x));

    DomainChange c = runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 99; i++) {
        if (i % 2 == 0) REQUIRE(allocator[x[i]].size() == 99);
    }
    REQUIRE_FALSE(allocator[x[99]].contains(1));
}