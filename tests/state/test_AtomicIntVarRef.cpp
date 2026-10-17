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


TEST_CASE("AtomicIntVarRef remove simple", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.size() == 9);
    REQUIRE_FALSE(v.contains(5));
    for (int32_t i = 0; i < 5; i++) REQUIRE(v.contains(i));
    for (int32_t i = 6; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("AtomicIntVarRef remove outside bounds", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(-1) | v.remove(10);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntVarRef remove already removed", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];
    v.remove(5);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(5);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.size() == 9);
}

TEST_CASE("AtomicIntVarRef remove min", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(0);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 9);
}

TEST_CASE("AtomicIntVarRef remove max", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(9);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 8);
    REQUIRE(v.size() == 9);
}

TEST_CASE("AtomicIntVarRef remove last", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(1);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(0);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarRef fix simple", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.fix(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.fixed());
    REQUIRE(v.contains(5));
    for (int32_t i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int32_t i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("AtomicIntVarRef fix already fixed", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(1);
    IntVarRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.fix(0);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.fixed());
}

TEST_CASE("AtomicIntVarRef fix outside bounds", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.fix(15);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntVarRef fix removed", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];
    v.remove(5);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.fix(5);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarRef removeBelow simple", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeBelow(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 5; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("AtomicIntVarRef removeBelow below min", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeBelow(0) | v.removeBelow(-5);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntVarRef removeBelow above max", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeBelow(15);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarRef removeAbove simple", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeAbove(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
    for (int i = 0; i <= 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("AtomicIntVarRef removeAbove above max", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);
    IntVarRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeAbove(9) | v.removeAbove(15);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntVarRef removeAbove below min", "[AtomicIntVarRef]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeAbove(-5);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarRef concurrent remove", "[AtomicIntVarRef][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(80);
    IntVarRef v = allocator[idx];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(ctx.item.get_global_id(0));
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 64);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 16);
    for (int i = 0; i < 64; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 64; i < 80; i++) REQUIRE(v.contains(i));
}

TEST_CASE("AtomicIntVarRef concurrent remove all", "[AtomicIntVarRef][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(64);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.remove(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntVarRef concurrent fix same", "[AtomicIntVarRef][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(80);
    IntVarRef v = allocator[idx];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.fix(5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.fixed());
}

TEST_CASE("AtomicIntVarRef concurrent fix different", "[AtomicIntVarRef][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(80);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.fix(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntVarRef concurrent removeBelow", "[AtomicIntVarRef][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(80);
    IntVarRef v = allocator[idx];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 68);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 12);
}

TEST_CASE("AtomicIntVarRef concurrent removeBelow all", "[AtomicIntVarRef][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(63);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntVarRef concurrent removeAbove", "[AtomicIntVarRef][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(80);
    IntVarRef v = allocator[idx];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeAbove(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
}

TEST_CASE("AtomicIntVarRef concurrent removeAbove all", "[AtomicIntVarRef][concurrent]") {
    Allocator allocator;
    IntVarIdx idx = allocator.variable(6, 80);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarRef v = ctx[idx];
        return v.removeAbove(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}