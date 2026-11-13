#include <catch2/catch_test_macros.hpp>
#include "MiniGCP/constraints/IsLess.hpp"
#include "MiniGCP/state/Bool.hpp"
#include "MiniGCP/state/Const.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "utils.hpp"

using namespace cp;

TEST_CASE("IsLess true update x", "[IsLess]") {
    Allocator allocator;
    IntInterval x(allocator, -10, 10);
    IntInterval y(allocator, -5, 5);
    Const<Lit<1>> r({});
    IsLess constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator].min() == -10);
    REQUIRE(x[allocator].max() == 4);
    REQUIRE(y[allocator].size() == 11);
}

TEST_CASE("IsLess true update y", "[IsLess]") {
    Allocator allocator;
    IntInterval x(allocator, -5, 5);
    IntInterval y(allocator, -10, 10);
    Const<Lit<1>> r({});
    IsLess constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator].size() == 11);
    REQUIRE(y[allocator].min() == -4);
    REQUIRE(y[allocator].max() == 10);
}

TEST_CASE("IsLess false update x", "[IsLess]") {
    Allocator allocator;
    IntInterval x(allocator, -10, 10);
    IntInterval y(allocator, -5, 5);
    Const<Lit<0>> r({});
    IsLess constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator].min() == -5);
    REQUIRE(x[allocator].max() == 10);
    REQUIRE(y[allocator].size() == 11);
}

TEST_CASE("IsLess false update y", "[IsLess]") {
    Allocator allocator;
    IntInterval x(allocator, -5, 5);
    IntInterval y(allocator, -10, 10);
    Const<Lit<0>> r({});
    IsLess constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator].size() == 11);
    REQUIRE(y[allocator].min() == -10);
    REQUIRE(y[allocator].max() == 5);
}

TEST_CASE("IsLess set true", "[IsLess]") {
    Allocator allocator;
    IntInterval x(allocator, 5);
    IntInterval y(allocator, 5, 10);
    Bool r(allocator);
    IsLess constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(r[allocator].value() == true);
}

TEST_CASE("IsLess set false", "[IsLess]") {
    Allocator allocator;
    IntInterval x(allocator, 5, 10);
    IntInterval y(allocator, 6);
    Bool r(allocator);
    IsLess constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(r[allocator].value() == false);
}

TEST_CASE("IsLess no change", "[IsLess]") {
    Allocator allocator;
    IntInterval x(allocator, 10);
    IntInterval y(allocator, 10);
    Bool r(allocator);
    IsLess constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("IsEqual inconsistency", "[IsEqual]") {
    Allocator allocator;
    IntInterval x(allocator, 5, 10);
    IntInterval y(allocator, 6);
    Const<Lit<1>> r({});
    IsLess constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}