#include <catch2/catch_test_macros.hpp>
#include "MiniGCP/constraints/IsOr.hpp"
#include "MiniGCP/state/Bool.hpp"
#include "MiniGCP/state/Const.hpp"
#include "utils.hpp"

using namespace cp;

TEST_CASE("IsOr true update x", "[IsOr]") {
    Allocator allocator;
    Bool x(allocator);
    Const<Lit<0>> y({});
    Const<Lit<1>> r({});
    IsOr constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator].value() == true);
}

TEST_CASE("IsOr true update y", "[IsOr]") {
    Allocator allocator;
    Const<Lit<0>> x({});
    Bool y(allocator);
    Const<Lit<1>> r({});
    IsOr constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(y[allocator].value() == true);
}

TEST_CASE("IsOr false update both", "[IsOr]") {
    Allocator allocator;
    Bool x(allocator);
    Bool y(allocator);
    Const<Lit<0>> r({});
    IsOr constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(y[allocator].value() == false);
    REQUIRE(y[allocator].value() == false);
}

TEST_CASE("IsOr set true", "[IsOr]") {
    Allocator allocator;
    Const<Lit<1>> x({});
    Bool y(allocator);
    Bool r(allocator);
    IsOr constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(r[allocator].value() == true);
}

TEST_CASE("IsOr set false", "[IsOr]") {
    Allocator allocator;
    Const<Lit<0>> x({});
    Const<Lit<0>> y({});
    Bool r(allocator);
    IsOr constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(r[allocator].value() == false);
}

TEST_CASE("IsOr no change", "[IsOr]") {
    Allocator allocator;
    Bool x(allocator);
    Bool y(allocator);
    Bool r(allocator);
    IsOr constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("IsOr inconsistency", "[IsOr]") {
    Allocator allocator;
    Const<Lit<0>> x({});
    Const<Lit<0>> y({});
    Const<Lit<1>> r({});
    IsOr constraint(x, y, r);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}