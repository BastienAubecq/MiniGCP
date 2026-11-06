#include <catch2/catch_test_macros.hpp>
#include "MiniGCP/constraints/NotEqual.hpp"
#include "MiniGCP/state/IntBitset.hpp"
#include "utils.hpp"

using namespace cp;

TEST_CASE("NotEqual neither fixed", "[NotEqual]") {
    Allocator allocator;
    IntBitset x(allocator, 3);
    IntBitset y(allocator, 3);
    NotEqual constraint(x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(x[allocator].size() == 3);
    REQUIRE(y[allocator].size() == 3);
}

TEST_CASE("NotEqual first fixed", "[NotEqual]") {
    Allocator allocator;
    IntBitset x(allocator, 2);
    IntBitset y(allocator, 1);
    NotEqual constraint(x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(x[allocator].size() == 1);
    REQUIRE(x[allocator].min() == 1);
}

TEST_CASE("NotEqual second fixed", "[NotEqual]") {
    Allocator allocator;
    IntBitset x(allocator, 1);
    IntBitset y(allocator, 2);
    NotEqual constraint(x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(y[allocator].size() == 1);
    REQUIRE(y[allocator].min() == 1);
}

TEST_CASE("NotEqual both fixed", "[NotEqual]") {
    Allocator allocator;
    IntBitset x(allocator, 2);
    IntBitset y(allocator, 2);
    x[allocator].fix(0);
    y[allocator].fix(1);
    NotEqual constraint(x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("NotEqual inconsistency", "[NotEqual]") {
    Allocator allocator;
    IntBitset x(allocator, 1);
    IntBitset y(allocator, 1);
    NotEqual constraint(x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}