#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/constraints/AllDifferentFWC.hpp"
#include "MiniGCP/state/IntBitset.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;


TEST_CASE("AllDifferentFWCSeq one fixed", "[AllDifferentFWCSeq]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 10);
    variables[5][allocator].fix(5);
    AllDifferentFWCSeq constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 5) REQUIRE_FALSE(variables[i][allocator].contains(5));
    }
}

TEST_CASE("AllDifferentFWCSeq few fixed", "[AllDifferentFWCSeq]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 10);
    variables[3][allocator].fix(2);
    variables[5][allocator].fix(5);
    variables[7][allocator].fix(8);
    AllDifferentFWCSeq constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 3) REQUIRE_FALSE(variables[i][allocator].contains(2));
        if (i != 5) REQUIRE_FALSE(variables[i][allocator].contains(5));
        if (i != 7) REQUIRE_FALSE(variables[i][allocator].contains(8));
    }
}

TEST_CASE("AllDifferentFWCSeq none fixed", "[AllDifferentFWCSeq]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 10);
    AllDifferentFWCSeq constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("AllDifferentFWCSeq all fixed", "[AllDifferentFWCSeq]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 10);
    for (int i = 0; i < 10; i++) {
        variables[i][allocator].fix(i);
    }
    AllDifferentFWCSeq constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("AllDifferentFWCSeq inconsistency", "[AllDifferentFWCSeq]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 10);
    variables[3][allocator].fix(5);
    variables[7][allocator].fix(5);
    AllDifferentFWCSeq constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("AllDifferentFWC one fixed", "[AllDifferentFWC]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 10);
    variables[5][allocator].fix(5);
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 5) REQUIRE_FALSE(variables[i][allocator].contains(5));
    }
}

TEST_CASE("AllDifferentFWC few fixed", "[AllDifferentFWC]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 10);
    variables[3][allocator].fix(2);
    variables[5][allocator].fix(5);
    variables[7][allocator].fix(8);
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 3) REQUIRE_FALSE(variables[i][allocator].contains(2));
        if (i != 5) REQUIRE_FALSE(variables[i][allocator].contains(5));
        if (i != 7) REQUIRE_FALSE(variables[i][allocator].contains(8));
    }
}

TEST_CASE("AllDifferentFWC none fixed", "[AllDifferentFWC]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 10);
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("AllDifferentFWC all fixed", "[AllDifferentFWC]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 10);
    for (int i = 0; i < 10; i++) {
        variables[i][allocator].fix(i);
    }
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("AllDifferentFWC inconsistency", "[AllDifferentFWC]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 10);
    variables[3][allocator].fix(5);
    variables[7][allocator].fix(5);
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("AllDifferentFWC concurrent few variables", "[AllDifferentFWC][concurrent]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 10, 10);
    variables[3][allocator].fix(2);
    variables[5][allocator].fix(5);
    variables[7][allocator].fix(8);
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        if (i != 3) REQUIRE_FALSE(variables[i][allocator].contains(2));
        if (i != 5) REQUIRE_FALSE(variables[i][allocator].contains(5));
        if (i != 7) REQUIRE_FALSE(variables[i][allocator].contains(8));
    }
}

TEST_CASE("AllDifferentFWC concurrent many variables", "[AllDifferentFWC][concurrent]") {
    Allocator allocator;
    IntBitsetArray variables(allocator, 100, 100);
    for (int i = 0; i < 100; i += 10) {
        variables[i][allocator].fix(i);
    }
    AllDifferentFWC constraint(allocator, variables);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 100; i++) {
        for (int j = 0; j < 10; j++) {
            if (i % 10 != 0 || i / 10 != j) REQUIRE_FALSE(variables[i][allocator].contains(j * 10));
        }
    }
}