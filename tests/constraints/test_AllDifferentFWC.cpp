#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/constraints/AllDifferentFWC.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;


TEST_CASE("AllDifferentFWCSeq one fixed", "[AllDifferentFWCSeq]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[5]].fix(5);
    AllDifferentFWCSeq constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 5) REQUIRE_FALSE(allocator[variables[i]].contains(5));
    }
}

TEST_CASE("AllDifferentFWCSeq few fixed", "[AllDifferentFWCSeq]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[3]].fix(2);
    allocator[variables[5]].fix(5);
    allocator[variables[7]].fix(8);
    AllDifferentFWCSeq constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 3) REQUIRE_FALSE(allocator[variables[i]].contains(2));
        if (i != 5) REQUIRE_FALSE(allocator[variables[i]].contains(5));
        if (i != 7) REQUIRE_FALSE(allocator[variables[i]].contains(8));
    }
}

TEST_CASE("AllDifferentFWCSeq none fixed", "[AllDifferentFWCSeq]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    AllDifferentFWCSeq constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("AllDifferentFWCSeq all fixed", "[AllDifferentFWCSeq]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(i, i));
    }
    AllDifferentFWCSeq constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("AllDifferentFWCSeq inconsistency", "[AllDifferentFWCSeq]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[3]].fix(5);
    allocator[variables[7]].fix(5);
    AllDifferentFWCSeq constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("AllDifferentFWC one fixed", "[AllDifferentFWC]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[5]].fix(5);
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 5) REQUIRE_FALSE(allocator[variables[i]].contains(5));
    }
}

TEST_CASE("AllDifferentFWC few fixed", "[AllDifferentFWC]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[3]].fix(2);
    allocator[variables[5]].fix(5);
    allocator[variables[7]].fix(8);
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 3) REQUIRE_FALSE(allocator[variables[i]].contains(2));
        if (i != 5) REQUIRE_FALSE(allocator[variables[i]].contains(5));
        if (i != 7) REQUIRE_FALSE(allocator[variables[i]].contains(8));
    }
}

TEST_CASE("AllDifferentFWC none fixed", "[AllDifferentFWC]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("AllDifferentFWC all fixed", "[AllDifferentFWC]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(i, i));
    }
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("AllDifferentFWC inconsistency", "[AllDifferentFWC]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[3]].fix(5);
    allocator[variables[7]].fix(5);
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("AllDifferentFWC concurrent few variables", "[AllDifferentFWC][concurrent]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 10; i++) {
        variables.push_back(allocator.variable(10));
    }
    allocator[variables[3]].fix(2);
    allocator[variables[5]].fix(5);
    allocator[variables[7]].fix(8);
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 3) REQUIRE_FALSE(allocator[variables[i]].contains(2));
        if (i != 5) REQUIRE_FALSE(allocator[variables[i]].contains(5));
        if (i != 7) REQUIRE_FALSE(allocator[variables[i]].contains(8));
    }
}

TEST_CASE("AllDifferentFWC concurrent many variables", "[AllDifferentFWC][concurrent]") {
    Allocator allocator;
    vector<IntVarIdx> variables;
    for (int i = 0; i < 100; i++) {
        variables.push_back(allocator.variable(100));
        if (i % 10 == 0) allocator[variables[i]].fix(i);
    }
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 10; j++) {
            if (i % 10 != 0 || i / 10 != j) REQUIRE_FALSE(allocator[variables[i]].contains(j * 10));
        }
    }
}