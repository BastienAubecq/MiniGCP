#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/constraints/Maximum.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "MiniGCP/state/Ptr.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;


TEST_CASE("MaximumSeq update y", "[MaximumSeq]") {
    Allocator allocator;
    IntIntervalPtr x(allocator, -10, 10, 3);
    IntInterval y(allocator, -10, 10);
    MaximumSeq constraint(allocator, x, 3, y);
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
    REQUIRE(y[allocator].min() == -1);
    REQUIRE(y[allocator].max() == 2);
    REQUIRE(y[allocator].size() == 4);
}

TEST_CASE("MaximumSeq update all x_i", "[MaximumSeq]") {
    Allocator allocator;
    IntIntervalPtr x(allocator, -10, 10, 10);
    IntInterval y(allocator, -1, 2);
    MaximumSeq constraint(allocator, x, 10, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        REQUIRE(x[allocator, i][allocator].min() == -10);
        REQUIRE(x[allocator, i][allocator].max() == 2);
        REQUIRE(x[allocator, i][allocator].size() == 13);
    }
    REQUIRE(y[allocator].size() == 4);
}

TEST_CASE("MaximumSeq update only valid x_i", "[MaximumSeq]") {
    Allocator allocator;
    Ptr<IntInterval> x(allocator, vector {
        IntInterval(allocator, -10, -2),
        IntInterval(allocator, -10, 10),
        IntInterval(allocator, -10, -2)
    });
    IntInterval y(allocator, -1, 2);
    MaximumSeq constraint(allocator, x, 3, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator, 0][allocator].size() == 9);
    REQUIRE(x[allocator, 2][allocator].size() == 9);
    REQUIRE(y[allocator].size() == 4);
    REQUIRE(x[allocator, 1][allocator].min() == -1);
    REQUIRE(x[allocator, 1][allocator].max() == 2);
    REQUIRE(x[allocator, 1][allocator].size() == 4);
}

TEST_CASE("MaximumSeq no change", "[MaximumSeq]") {
    Allocator allocator;
    Ptr<IntInterval> x(allocator, vector {
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2)
    });
    IntInterval y(allocator, -1, 2);
    MaximumSeq constraint(allocator, x, 3, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}


TEST_CASE("MaximumSeq inconsistency", "[MaximumSeq]") {
    Allocator allocator;
    Ptr<IntInterval> x(allocator, vector {
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2)
    });
    IntInterval y(allocator, 3, 6);
    MaximumSeq constraint(allocator, x, 3, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("Maximum update y", "[Maximum]") {
    Allocator allocator;
    IntIntervalPtr x(allocator, -10, 10, 3);
    IntInterval y(allocator, -10, 10);
    Maximum constraint(allocator, x, 3, y);
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
    REQUIRE(y[allocator].min() == -1);
    REQUIRE(y[allocator].max() == 2);
    REQUIRE(y[allocator].size() == 4);
}

TEST_CASE("Maximum update all x_i", "[Maximum]") {
    Allocator allocator;
    IntIntervalPtr x(allocator, -10, 10, 10);
    IntInterval y(allocator, -1, 2);
    Maximum constraint(allocator, x, 10, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        REQUIRE(x[allocator, i][allocator].min() == -10);
        REQUIRE(x[allocator, i][allocator].max() == 2);
        REQUIRE(x[allocator, i][allocator].size() == 13);
    }
    REQUIRE(y[allocator].size() == 4);
}

TEST_CASE("Maximum update only valid x_i", "[Maximum]") {
    Allocator allocator;
    Ptr<IntInterval> x(allocator, vector {
        IntInterval(allocator, -10, -2),
        IntInterval(allocator, -10, 10),
        IntInterval(allocator, -10, -2)
    });
    IntInterval y(allocator, -1, 2);
    Maximum constraint(allocator, x, 3, y);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator, 0][allocator].size() == 9);
    REQUIRE(x[allocator, 2][allocator].size() == 9);
    REQUIRE(y[allocator].size() == 4);
    REQUIRE(x[allocator, 1][allocator].min() == -1);
    REQUIRE(x[allocator, 1][allocator].max() == 2);
    REQUIRE(x[allocator, 1][allocator].size() == 4);
}

TEST_CASE("Maximum no change", "[Maximum]") {
    Allocator allocator;
    Ptr<IntInterval> x(allocator, vector {
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2)
    });
    IntInterval y(allocator, -1, 2);
    Maximum constraint(allocator, x, 3, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("Maximum inconsistency", "[Maximum]") {
    Allocator allocator;
    Ptr<IntInterval> x(allocator, vector {
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2),
        IntInterval(allocator, -1, 2)
    });
    IntInterval y(allocator, 3, 6);
    Maximum constraint(allocator, x, 3, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("Maximum concurrent few variables", "[Maximum][concurrent]") {
    Allocator allocator;
    IntIntervalPtr x(allocator, -10, 10, 10);
    IntInterval y(allocator, -1, 2);
    Maximum constraint(allocator, x, 10, y);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 10; i++) {
        REQUIRE(x[allocator, i][allocator].min() == -10);
        REQUIRE(x[allocator, i][allocator].max() == 2);
        REQUIRE(x[allocator, i][allocator].size() == 13);
    }
    REQUIRE(y[allocator].size() == 4);
}

TEST_CASE("Maximum concurrent many variables", "[Maximum][concurrent]") {
    Allocator allocator;
    IntIntervalPtr x(allocator, -10, 10, 100);
    IntInterval y(allocator, -1, 2);
    Maximum constraint(allocator, x, 100, y);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 100; i++) {
        REQUIRE(x[allocator, i][allocator].min() == -10);
        REQUIRE(x[allocator, i][allocator].max() == 2);
        REQUIRE(x[allocator, i][allocator].size() == 13);
    }
    REQUIRE(y[allocator].size() == 4);
}