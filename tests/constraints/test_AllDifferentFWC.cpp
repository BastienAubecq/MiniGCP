#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/constraints/AllDifferentFWC.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;


TEST_CASE("sequential one fixed", "[AllDifferentFWCSeq][device]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[5]].fix(5);
    AllDifferentFWCSeq constraint(allocator, variables);

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0] == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 5) REQUIRE_FALSE(allocator[variables[i]].contains(5));
    }
}

TEST_CASE("sequential few fixed", "[AllDifferentFWCSeq][device]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[3]].fix(2);
    allocator[variables[5]].fix(5);
    allocator[variables[7]].fix(8);
    AllDifferentFWCSeq constraint(allocator, variables);

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0] == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 3) REQUIRE_FALSE(allocator[variables[i]].contains(2));
        if (i != 5) REQUIRE_FALSE(allocator[variables[i]].contains(5));
        if (i != 7) REQUIRE_FALSE(allocator[variables[i]].contains(8));
    }
}

TEST_CASE("sequential all fixed", "[AllDifferentFWCSeq][device]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(i, i));
    }
    AllDifferentFWCSeq constraint(allocator, variables);

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0] == DomainChange::none);
}

TEST_CASE("sequential inconsistency", "[AllDifferentFWCSeq][device]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[3]].fix(5);
    allocator[variables[7]].fix(5);
    AllDifferentFWCSeq constraint(allocator, variables);

    REQUIRE((runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0] & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("parallel one fixed", "[AllDifferentFWC][device]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[5]].fix(5);
    AllDifferentFWC constraint(allocator, variables);

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0] == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 5) REQUIRE_FALSE(allocator[variables[i]].contains(5));
    }
}

TEST_CASE("parallel few fixed", "[AllDifferentFWC][device]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[3]].fix(2);
    allocator[variables[5]].fix(5);
    allocator[variables[7]].fix(8);
    AllDifferentFWC constraint(allocator, variables);

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0] == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 3) REQUIRE_FALSE(allocator[variables[i]].contains(2));
        if (i != 5) REQUIRE_FALSE(allocator[variables[i]].contains(5));
        if (i != 7) REQUIRE_FALSE(allocator[variables[i]].contains(8));
    }
}

TEST_CASE("parallel all fixed", "[AllDifferentFWC][device]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(i, i));
    }
    AllDifferentFWC constraint(allocator, variables);

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0] == DomainChange::none);
}

TEST_CASE("parallel inconsistency", "[AllDifferentFWC][device]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[3]].fix(5);
    allocator[variables[7]].fix(5);
    AllDifferentFWC constraint(allocator, variables);

    REQUIRE((runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0] & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("concurrent few variables", "[AllDifferentFWC][device][concurrent]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[3]].fix(2);
    allocator[variables[5]].fix(5);
    allocator[variables[7]].fix(8);
    AllDifferentFWCSeq constraint(allocator, variables);

    REQUIRE(combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    })) == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 3) REQUIRE_FALSE(allocator[variables[i]].contains(2));
        if (i != 5) REQUIRE_FALSE(allocator[variables[i]].contains(5));
        if (i != 7) REQUIRE_FALSE(allocator[variables[i]].contains(8));
    }
}

TEST_CASE("concurrent many variables", "[AllDifferentFWC][device][concurrent]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 100; i++) {
        variables.push_back(allocator.variable(100));
        if (i % 10 == 0) allocator[variables[i]].fix(i);
    }
    AllDifferentFWCSeq constraint(allocator, variables);

    REQUIRE(combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    })) == DomainChange::changed);
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 10; j++) {
            if (i % 10 != 0 || i / 10 != j) REQUIRE_FALSE(allocator[variables[i]].contains(j * 10));
        }
    }
}