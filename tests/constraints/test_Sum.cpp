#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/constraints/Sum.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;


TEST_CASE("SumSeq update y", "[SumSeq]") {
    Allocator allocator;
    vector<IntVarIdx> x {
        allocator.variable(-1, 2),
        allocator.variable(-1, 2),
        allocator.variable(-1, 2)
    };
    IntVarIdx y = allocator.variable(-10, 10);
    SumSeq constraint(allocator, x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(allocator[x[0]].size() == 4);
    REQUIRE(allocator[x[1]].size() == 4);
    REQUIRE(allocator[x[2]].size() == 4);
    REQUIRE(allocator[y].min() == -3);
    REQUIRE(allocator[y].max() == 6);
    REQUIRE(allocator[y].size() == 10);
}

TEST_CASE("SumSeq update one x_i", "[SumSeq]") {
    Allocator allocator;
    vector<IntVarIdx> x {
        allocator.variable(-1, 2),
        allocator.variable(-10, 10),
        allocator.variable(-1, 2)
    };
    IntVarIdx y = allocator.variable(-1, 2);
    SumSeq constraint(allocator, x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(allocator[x[0]].size() == 4);
    REQUIRE(allocator[x[2]].size() == 4);
    REQUIRE(allocator[y].size() == 4);
    REQUIRE(allocator[x[1]].min() == -5);
    REQUIRE(allocator[x[1]].max() == 4);
    REQUIRE(allocator[x[1]].size() == 10);
}

TEST_CASE("SumSeq no change", "[SumSeq]") {
    Allocator allocator;
    vector<IntVarIdx> x {
        allocator.variable(-1, 2),
        allocator.variable(-1, 2),
        allocator.variable(-1, 2)
    };
    IntVarIdx y = allocator.variable(-3, 6);
    SumSeq constraint(allocator, x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("SumSeq inconsistency", "[SumSeq]") {
    Allocator allocator;
    vector<IntVarIdx> x {
        allocator.variable(-1, 2),
        allocator.variable(-1, 2),
        allocator.variable(-1, 2)
    };
    IntVarIdx y = allocator.variable(7, 10);
    SumSeq constraint(allocator, x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("SumSeq update all x_i", "[SumSeq]") {
    Allocator allocator;
    vector<IntVarIdx> x;
    for (int i = 0; i < 10; i++) {
        x.push_back(allocator.variable(-20, 1));
    }
    IntVarIdx y = allocator.variable(-1, 2);
    SumSeq constraint(allocator, x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        REQUIRE(allocator[x[i]].min() == -10);
        REQUIRE(allocator[x[i]].max() == 1);
        REQUIRE(allocator[x[i]].size() == 12);
    }
    REQUIRE(allocator[y].size() == 4);
}


TEST_CASE("Sum update y", "[Sum]") {
    Allocator allocator;
    vector<IntVarIdx> x {
        allocator.variable(-1, 2),
        allocator.variable(-1, 2),
        allocator.variable(-1, 2)
    };
    IntVarIdx y = allocator.variable(-10, 10);
    Sum constraint(allocator, x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(allocator[x[0]].size() == 4);
    REQUIRE(allocator[x[1]].size() == 4);
    REQUIRE(allocator[x[2]].size() == 4);
    REQUIRE(allocator[y].min() == -3);
    REQUIRE(allocator[y].max() == 6);
    REQUIRE(allocator[y].size() == 10);
}

TEST_CASE("Sum update one x_i", "[Sum]") {
    Allocator allocator;
    vector<IntVarIdx> x {
        allocator.variable(-1, 2),
        allocator.variable(-10, 10),
        allocator.variable(-1, 2)
    };
    IntVarIdx y = allocator.variable(-1, 2);
    Sum constraint(allocator, x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(allocator[x[0]].size() == 4);
    REQUIRE(allocator[x[2]].size() == 4);
    REQUIRE(allocator[y].size() == 4);
    REQUIRE(allocator[x[1]].min() == -5);
    REQUIRE(allocator[x[1]].max() == 4);
    REQUIRE(allocator[x[1]].size() == 10);
}

TEST_CASE("Sum no change", "[Sum]") {
    Allocator allocator;
    vector<IntVarIdx> x {
        allocator.variable(-1, 2),
        allocator.variable(-1, 2),
        allocator.variable(-1, 2)
    };
    IntVarIdx y = allocator.variable(-3, 6);
    Sum constraint(allocator, x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("Sum inconsistency", "[Sum]") {
    Allocator allocator;
    vector<IntVarIdx> x {
        allocator.variable(-1, 2),
        allocator.variable(-1, 2),
        allocator.variable(-1, 2)
    };
    IntVarIdx y = allocator.variable(7, 10);
    Sum constraint(allocator, x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("Sum update all x_i", "[Sum]") {
    Allocator allocator;
    vector<IntVarIdx> x;
    for (int i = 0; i < 10; i++) {
        x.push_back(allocator.variable(-20, 1));
    }
    IntVarIdx y = allocator.variable(-1, 2);
    Sum constraint(allocator, x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        REQUIRE(allocator[x[i]].min() == -10);
        REQUIRE(allocator[x[i]].max() == 1);
        REQUIRE(allocator[x[i]].size() == 12);
    }
    REQUIRE(allocator[y].size() == 4);
}


TEST_CASE("Sum concurrent few variables", "[Sum][concurrent]") {
    Allocator allocator;
    vector<IntVarIdx> x;
    for (int i = 0; i < 10; i++) {
        x.push_back(allocator.variable(-20, 1));
    }
    IntVarIdx y = allocator.variable(-1, 2);
    Sum constraint(allocator, x, y);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        REQUIRE(allocator[x[i]].min() == -10);
        REQUIRE(allocator[x[i]].max() == 1);
        REQUIRE(allocator[x[i]].size() == 12);
    }
    REQUIRE(allocator[y].size() == 4);
}

TEST_CASE("Sum concurrent many variables", "[Sum][concurrent]") {
    Allocator allocator;
    vector<IntVarIdx> x;
    for (int i = 0; i < 100; i++) {
        x.push_back(allocator.variable(-200, 1));
    }
    IntVarIdx y = allocator.variable(-1, 2);
    Sum constraint(allocator, x, y);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 100; i++) {
        REQUIRE(allocator[x[i]].min() == -100);
        REQUIRE(allocator[x[i]].max() == 1);
        REQUIRE(allocator[x[i]].size() == 102);
    }
    REQUIRE(allocator[y].size() == 4);
}