#include <catch2/catch_test_macros.hpp>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/constraints/NotEqual.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "utils.hpp"

using namespace cp;

TEST_CASE("NotEqual neither fixed", "[NotEqual]") {
    Allocator allocator;
    IntVarBitsetIdx x = allocator.intVarBitset(3);
    IntVarBitsetIdx y = allocator.intVarBitset(3);
    NotEqual constraint(x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(allocator[x].size() == 3);
    REQUIRE(allocator[y].size() == 3);
}

TEST_CASE("NotEqual first fixed", "[NotEqual]") {
    Allocator allocator;
    IntVarBitsetIdx x = allocator.intVarBitset(2);
    IntVarBitsetIdx y = allocator.intVarBitset(1);
    NotEqual constraint(x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(allocator[x].size() == 1);
    REQUIRE(allocator[x].min() == 1);
}

TEST_CASE("NotEqual second fixed", "[NotEqual]") {
    Allocator allocator;
    IntVarBitsetIdx x = allocator.intVarBitset(1);
    IntVarBitsetIdx y = allocator.intVarBitset(2);
    NotEqual constraint(x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(allocator[y].size() == 1);
    REQUIRE(allocator[y].min() == 1);
}

TEST_CASE("NotEqual both fixed", "[NotEqual]") {
    Allocator allocator;
    IntVarBitsetIdx x = allocator.intVarBitset(2);
    IntVarBitsetIdx y = allocator.intVarBitset(2);
    allocator[x].fix(0);
    allocator[y].fix(1);
    NotEqual constraint(x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}

TEST_CASE("NotEqual inconsistency", "[NotEqual]") {
    Allocator allocator;
    IntVarBitsetIdx x = allocator.intVarBitset(1);
    IntVarBitsetIdx y = allocator.intVarBitset(1);
    NotEqual constraint(x, y);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}