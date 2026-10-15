#include <catch2/catch_test_macros.hpp>
#include <array>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/constraints/Element1D.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;


TEST_CASE("Element1D update z", "[Element1D][device]") {
    Allocator allocator;
    auto t = allocator.array<Scope::global, int32_t>(array { 4, 3, 1, 2 });
    IntVarIdx y = allocator.variable(array { 1, 3 });
    IntVarIdx z = allocator.variable(-10, 10);
    Element1D constraint(allocator, t, y, z);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(allocator[z].min() == 2);
    REQUIRE(allocator[z].max() == 3);
    REQUIRE(allocator[y].size() == 2);
}

TEST_CASE("Element1D update y", "[Element1D][device]") {
    Allocator allocator;
    auto t = allocator.array<Scope::global, int32_t>(array { 4, 3, 1, 2 });
    IntVarIdx y = allocator.variable(4);
    IntVarIdx z = allocator.variable(2, 3);
    Element1D constraint(allocator, t, y, z);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(allocator[y].contains(0));
    REQUIRE(allocator[y].contains(1));
    REQUIRE_FALSE(allocator[y].contains(2));
    REQUIRE(allocator[y].contains(3));
}

TEST_CASE("Element1D update y and z", "[Element1D][device]") {
    Allocator allocator;
    array<int32_t, 4> t { 4, 3, 1, 2 };
    IntVarIdx y = allocator.variable(array { 0, 2, 3 });
    IntVarIdx z = allocator.variable(2, 3);
    Element1D constraint(allocator, allocator.array<Scope::global, int32_t>(t), y, z);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(allocator[y].contains(0));
    REQUIRE_FALSE(allocator[y].contains(2));
    REQUIRE(allocator[y].contains(3));
    REQUIRE(allocator[z].contains(2));
    REQUIRE_FALSE(allocator[z].contains(3));
}

TEST_CASE("Element1D no change", "[Element1D][device]") {
    Allocator allocator;
    auto t = allocator.array<Scope::global, int32_t>(array { 4, 3, 1, 2 });
    IntVarIdx y = allocator.variable(array { 1, 3 });
    IntVarIdx z = allocator.variable(2, 3);
    Element1D constraint(allocator, t, y, z);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("Element1D inconsistency", "[Element1D][device]") {
    Allocator allocator;
    auto t = allocator.array<Scope::global, int32_t>(array { 4, 3, 1, 2 });
    IntVarIdx y = allocator.variable(array { 1, 3 });
    IntVarIdx z = allocator.variable(4, 5);
    Element1D constraint(allocator, t, y, z);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}
