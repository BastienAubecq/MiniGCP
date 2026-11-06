#include <catch2/catch_test_macros.hpp>
#include <array>
#include "MiniGCP/constraints/Element1D.hpp"
#include "MiniGCP/state/IntBitset.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;


TEST_CASE("Element1D update z", "[Element1D]") {
    Allocator allocator;
    Array<int32_t> t(allocator, array { 4, 3, 1, 2 });
    IntBitset y(allocator, array { 1, 3 });
    IntInterval z(allocator, -10, 10);
    Element1D constraint(allocator, t, y, z);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(z[allocator].min() == 2);
    REQUIRE(z[allocator].max() == 3);
    REQUIRE(y[allocator].size() == 2);
}

TEST_CASE("Element1D update y", "[Element1D]") {
    Allocator allocator;
    Array<int32_t> t(allocator, array { 4, 3, 1, 2 });
    IntBitset y(allocator, 4);
    IntInterval z(allocator, 2, 3);
    Element1D constraint(allocator, t, y, z);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(y[allocator].contains(0));
    REQUIRE(y[allocator].contains(1));
    REQUIRE_FALSE(y[allocator].contains(2));
    REQUIRE(y[allocator].contains(3));
}

TEST_CASE("Element1D update y and z", "[Element1D]") {
    Allocator allocator;
    Array<int32_t> t(allocator, array { 4, 3, 1, 2 });
    IntBitset y(allocator, array { 0, 2, 3 });
    IntInterval z(allocator, 2, 3);
    Element1D constraint(allocator, t, y, z);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(y[allocator].contains(0));
    REQUIRE_FALSE(y[allocator].contains(2));
    REQUIRE(y[allocator].contains(3));
    REQUIRE(z[allocator].min() == 2);
    REQUIRE(z[allocator].max() == 2);
}

TEST_CASE("Element1D no change", "[Element1D]") {
    Allocator allocator;
    Array<int32_t> t(allocator, array { 4, 3, 1, 2 });
    IntBitset y(allocator, array { 1, 3 });
    IntInterval z(allocator, 2, 3);
    Element1D constraint(allocator, t, y, z);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("Element1D inconsistency", "[Element1D]") {
    Allocator allocator;
    Array<int32_t> t(allocator, array { 4, 3, 1, 2 });
    IntBitset y(allocator, array { 1, 3 });
    IntInterval z(allocator, 4, 5);
    Element1D constraint(allocator, t, y, z);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}