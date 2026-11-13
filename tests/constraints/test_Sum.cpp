#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/constraints/Sum.hpp"
#include "MiniGCP/state/Bool.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "MiniGCP/state/Mul.hpp"
#include "MiniGCP/state/Ptr.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;


TEST_CASE("SumSeq update y", "[SumSeq]") {
    Allocator allocator;
    IntIntervalPtr x(allocator, -10, 10, 3);
    IntInterval y(allocator, -10, 10);
    Sum constraint(allocator, x, 3, y);
    for (uint32_t i = 0; i < 3; i++) {
        x[i][allocator].removeBelow(-1);
        x[i][allocator].removeAbove(2);
    }

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator, 0][allocator].size() == 4);
    REQUIRE(x[allocator, 1][allocator].size() == 4);
    REQUIRE(x[allocator, 2][allocator].size() == 4);
    REQUIRE(y[allocator].min() == -3);
    REQUIRE(y[allocator].max() == 6);
    REQUIRE(y[allocator].size() == 10);
}

TEST_CASE("SumSeq update one x_i", "[SumSeq]") {
    Allocator allocator;
    Ptr<IntInterval> x(allocator, vector {
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -10, 10),
        IntInterval(allocator, -1, 2)
    });
    IntInterval y(allocator, -1, 2);
    SumSeq constraint(allocator, x, 3, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator, 0][allocator].size() == 4);
    REQUIRE(x[allocator, 2][allocator].size() == 4);
    REQUIRE(y[allocator].size() == 4);
    REQUIRE(x[allocator, 1][allocator].min() == -5);
    REQUIRE(x[allocator, 1][allocator].max() == 4);
    REQUIRE(x[allocator, 1][allocator].size() == 10);
}

TEST_CASE("SumSeq no change", "[SumSeq]") {
    Allocator allocator;
    Ptr<IntInterval> x(allocator, vector {
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2)
    });
    IntInterval y(allocator, -3, 6);
    SumSeq constraint(allocator, x, 3, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("SumSeq inconsistency", "[SumSeq]") {
    Allocator allocator;
    Ptr<IntInterval> x(allocator, vector {
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2)
    });
    IntInterval y(allocator, 7, 10);
    SumSeq constraint(allocator, x, 3, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("SumSeq update all x_i", "[SumSeq]") {
    Allocator allocator;
    IntIntervalPtr x(allocator, -20, 1, 10);
    IntInterval y(allocator, -1, 2);
    SumSeq constraint(allocator, x, 10, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        REQUIRE(x[allocator, i][allocator].min() == -10);
        REQUIRE(x[allocator, i][allocator].max() == 1);
        REQUIRE(x[allocator, i][allocator].size() == 12);
    }
    REQUIRE(y[allocator].size() == 4);
}


TEST_CASE("Sum update y", "[Sum]") {
    Allocator allocator;
    IntIntervalPtr x(allocator, -10, 10, 3);
    IntInterval y(allocator, -10, 10);
    Sum constraint(allocator, x, 3, y);
    for (uint32_t i = 0; i < 3; i++) {
        x[i][allocator].removeBelow(-1);
        x[i][allocator].removeAbove(2);
    }

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator, 0][allocator].size() == 4);
    REQUIRE(x[allocator, 1][allocator].size() == 4);
    REQUIRE(x[allocator, 2][allocator].size() == 4);
    REQUIRE(y[allocator].min() == -3);
    REQUIRE(y[allocator].max() == 6);
    REQUIRE(y[allocator].size() == 10);
}

TEST_CASE("Sum update one x_i", "[Sum]") {
    Allocator allocator;
    Ptr<IntInterval> x(allocator, vector {
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -10, 10),
        IntInterval(allocator, -1, 2)
    });
    IntInterval y(allocator, -1, 2);
    Sum constraint(allocator, x, 3, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator, 0][allocator].size() == 4);
    REQUIRE(x[allocator, 2][allocator].size() == 4);
    REQUIRE(y[allocator].size() == 4);
    REQUIRE(x[allocator, 1][allocator].min() == -5);
    REQUIRE(x[allocator, 1][allocator].max() == 4);
    REQUIRE(x[allocator, 1][allocator].size() == 10);
}

TEST_CASE("Sum no change", "[Sum]") {
    Allocator allocator;
    Ptr<IntInterval> x(allocator, vector {
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2)
    });
    IntInterval y(allocator, -3, 6);
    Sum constraint(allocator, x, 3, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("Sum inconsistency", "[Sum]") {
    Allocator allocator;
    Ptr<IntInterval> x(allocator, vector {
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2)
    });
    IntInterval y(allocator, 7, 10);
    Sum constraint(allocator, x, 3, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("Sum update all x_i", "[Sum]") {
    Allocator allocator;
    IntIntervalPtr x(allocator, -20, 1, 10);
    IntInterval y(allocator, -1, 2);
    Sum constraint(allocator, x, 10, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        REQUIRE(x[allocator, i][allocator].min() == -10);
        REQUIRE(x[allocator, i][allocator].max() == 1);
        REQUIRE(x[allocator, i][allocator].size() == 12);
    }
    REQUIRE(y[allocator].size() == 4);
}


TEST_CASE("Sum concurrent few variables", "[Sum][concurrent]") {
    Allocator allocator;
    IntIntervalPtr x(allocator, -20, 1, 10);
    IntInterval y(allocator, -1, 2);
    Sum constraint(allocator, x, 10, y);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        REQUIRE(x[allocator, i][allocator].min() == -10);
        REQUIRE(x[allocator, i][allocator].max() == 1);
        REQUIRE(x[allocator, i][allocator].size() == 12);
    }
    REQUIRE(y[allocator].size() == 4);
}

TEST_CASE("Sum concurrent many variables", "[Sum][concurrent]") {
    Allocator allocator;
    IntIntervalPtr x(allocator, -200, 1, 100);
    IntInterval y(allocator, -1, 2);
    Sum constraint(allocator, x, 100, y);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 100; i++) {
        REQUIRE(x[allocator, i][allocator].min() == -100);
        REQUIRE(x[allocator, i][allocator].max() == 1);
        REQUIRE(x[allocator, i][allocator].size() == 102);
    }
    REQUIRE(y[allocator].size() == 4);
}