#include <catch2/catch_test_macros.hpp>
#include "MiniGCP/constraints/IsEqual.hpp"
#include "MiniGCP/state/Bool.hpp"
#include "MiniGCP/state/Const.hpp"
#include "MiniGCP/state/IntBitset.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "utils.hpp"

using namespace cp;

TEST_CASE("IsEqual true update x", "[IsEqual]") {
    Allocator allocator;
    IntInterval x(allocator, -10, 10);
    IntInterval y(allocator, -5, 5);
    Const<Lit<1>> r({});
    IsEqual constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator].min() == -5);
    REQUIRE(x[allocator].max() == 5);
    REQUIRE(y[allocator].size() == 11);
}

TEST_CASE("IsEqual true update y", "[IsEqual]") {
    Allocator allocator;
    IntInterval x(allocator, -5, 5);
    IntInterval y(allocator, -10, 10);
    Const<Lit<1>> r({});
    IsEqual constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator].size() == 11);
    REQUIRE(y[allocator].min() == -5);
    REQUIRE(y[allocator].max() == 5);
}

TEST_CASE("IsEqual false update x", "[IsEqual]") {
    Allocator allocator;
    IntBitset x(allocator, 10);
    Const<Lit<5>> y({});
    Const<Lit<0>> r({});
    IsEqual constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(x[allocator].contains(5));
    REQUIRE(x[allocator].size() == 9);
}

TEST_CASE("IsEqual false update y", "[IsEqual]") {
    Allocator allocator;
    Const<Lit<5>> x({});
    IntBitset y(allocator, 10);
    Const<Lit<0>> r({});
    IsEqual constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(y[allocator].contains(5));
    REQUIRE(y[allocator].size() == 9);
}

TEST_CASE("IsEqual set true", "[IsEqual]") {
    Allocator allocator;
    Const<Lit<5>> x({});
    Const<Lit<5>> y({});
    Bool r(allocator);
    IsEqual constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(r[allocator].value() == true);
}

TEST_CASE("IsEqual set false", "[IsEqual]") {
    Allocator allocator;
    IntInterval x(allocator, 5);
    IntInterval y(allocator, 5, 10);
    Bool r(allocator);
    IsEqual constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(r[allocator].value() == false);
}

TEST_CASE("IsEqual no change", "[IsEqual]") {
    Allocator allocator;
    IntInterval x(allocator, 5);
    IntInterval y(allocator, 4, 10);
    Bool r(allocator);
    IsEqual constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("IsEqual inconsistency", "[IsEqual]") {
    Allocator allocator;
    IntInterval x(allocator, 5);
    IntInterval y(allocator, 5, 10);
    Const<Lit<1>> r({});
    IsEqual constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}