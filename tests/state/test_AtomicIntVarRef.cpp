#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <span>
#include "sycl/sycl.hpp"
#include "MiniGCP/Model.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/IntVarIdx.hpp"
#include "MiniGCP/state/IntVarRef.hpp"
#include "MiniGCP/state/AtomicIntVarRef.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "utils.hpp"

using namespace std;
using namespace cp;


TEST_CASE("remove simple", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(5);
    })[0] == DomainChange::changed);
    REQUIRE(v.size() == 9);
    REQUIRE_FALSE(v.contains(5));
    for (int32_t i = 0; i < 5; i++) REQUIRE(v.contains(i));
    for (int32_t i = 6; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("remove outside bounds", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(-1) | v.remove(10);
    })[0] == DomainChange::none);
    REQUIRE(v.size() == 10);
}

TEST_CASE("remove already removed", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];
    v.remove(5);

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(5);
    })[0] == DomainChange::none);
    REQUIRE(v.size() == 9);
}

TEST_CASE("remove min", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(0);
    })[0] == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 9);
}

TEST_CASE("remove max", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(9);
    })[0] == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 8);
    REQUIRE(v.size() == 9);
}

TEST_CASE("remove last", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(1);

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(0);
    })[0] == DomainChange::inconsistency);
}


TEST_CASE("fix simple", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.fix(5);
    })[0] == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.fixed());
    REQUIRE(v.contains(5));
    for (int32_t i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int32_t i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("fix already fixed", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(1);
    IntVarRef v = allocator[idx];

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.fix(0);
    })[0] == DomainChange::none);
    REQUIRE(v.fixed());
}

TEST_CASE("fix outside bounds", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.fix(15);
    })[0] == DomainChange::inconsistency);
}

TEST_CASE("fix removed", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];
    v.remove(5);

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.fix(5);
    })[0] == DomainChange::inconsistency);
}


TEST_CASE("removeBelow simple", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeBelow(5);
    })[0] == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 5; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("removeBelow below min", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeBelow(0) | v.removeBelow(-5);
    })[0] == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("removeBelow above max", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeBelow(15);
    })[0] == DomainChange::inconsistency);
}


TEST_CASE("removeAbove simple", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeAbove(5);
    })[0] == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
    for (int i = 0; i <= 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("removeAbove above max", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeAbove(9) | v.removeAbove(15);
    })[0] == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("removeAbove below min", "[AtomicIntVarRef][device]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);

    REQUIRE(runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeAbove(-5);
    })[0] == DomainChange::inconsistency);
}


TEST_CASE("concurrent remove", "[AtomicIntVarRef][device][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(80);
    IntVarRef v = allocator[idx];

    REQUIRE(combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(ctx.item.get_global_id(0));
    })) == DomainChange::changed);
    REQUIRE(v.min() == 64);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 16);
    for (int i = 0; i < 64; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 64; i < 80; i++) REQUIRE(v.contains(i));
}

TEST_CASE("concurrent remove all", "[AtomicIntVarRef][device][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(64);

    REQUIRE((combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(ctx.item.get_global_id(0));
    })) & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("concurrent fix same", "[AtomicIntVarRef][device][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(80);
    IntVarRef v = allocator[idx];

    REQUIRE(combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.fix(5);
    })) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.fixed());
}

TEST_CASE("concurrent fix different", "[AtomicIntVarRef][device][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(80);

    REQUIRE((combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.fix(ctx.item.get_global_id(0));
    })) & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("concurrent removeBelow", "[AtomicIntVarRef][device][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(80);
    IntVarRef v = allocator[idx];

    REQUIRE(combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeBelow(ctx.item.get_global_id(0) + 5);
    })) == DomainChange::changed);
    REQUIRE(v.min() == 68);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 12);
}

TEST_CASE("concurrent removeBelow all", "[AtomicIntVarRef][device][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(63);

    REQUIRE((combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeBelow(ctx.item.get_global_id(0) + 5);
    })) & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("concurrent removeAbove", "[AtomicIntVarRef][device][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(80);
    IntVarRef v = allocator[idx];

    REQUIRE(combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeAbove(ctx.item.get_global_id(0) + 5);
    })) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
}

TEST_CASE("concurrent removeAbove all", "[AtomicIntVarRef][device][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(6, 80);

    REQUIRE((combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeAbove(ctx.item.get_global_id(0) + 5);
    })) & DomainChange::inconsistency) == DomainChange::inconsistency);
}