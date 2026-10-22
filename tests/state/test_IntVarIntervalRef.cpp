#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <span>
#include "sycl/sycl.hpp"
#include "MiniGCP/Model.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/IntVarIntervalIdx.hpp"
#include "MiniGCP/state/IntVarIntervalRef.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "utils.hpp"

using namespace std;
using namespace cp;


TEST_CASE("IntVarIntervalRef allocation size", "[IntVarIntervalRef][host]") {
    Allocator allocator;
    IntVarIntervalRef v = allocator[allocator.intVarInterval(10)];

    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
    REQUIRE_FALSE(v.fixed());
}

TEST_CASE("IntVarIntervalRef allocation min max", "[IntVarIntervalRef][host]") {
    Allocator allocator;
    IntVarIntervalRef v = allocator[allocator.intVarInterval(5, 10)];

    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 10);
    REQUIRE(v.size() == 6);
}


TEST_CASE("IntVarIntervalRef fix simple", "[IntVarIntervalRef][host]") {
    Allocator allocator;
    IntVarIntervalRef v = allocator[allocator.intVarInterval(10)];

    REQUIRE(v.fix(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.fixed());
}

TEST_CASE("IntVarIntervalRef fix already fixed", "[IntVarIntervalRef][host]") {
    Allocator allocator;
    IntVarIntervalRef v = allocator[allocator.intVarInterval(1)];

    REQUIRE(v.fix(0) == DomainChange::none);
    REQUIRE(v.fixed());
}

TEST_CASE("IntVarIntervalRef fix outside bounds", "[IntVarIntervalRef][host]") {
    Allocator allocator;
    IntVarIntervalRef v = allocator[allocator.intVarInterval(10)];

    REQUIRE(v.fix(15) == DomainChange::inconsistency);
}


TEST_CASE("IntVarIntervalRef removeBelow simple", "[IntVarIntervalRef][host]") {
    Allocator allocator;
    IntVarIntervalRef v = allocator[allocator.intVarInterval(10)];

    REQUIRE(v.removeBelow(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
}

TEST_CASE("IntVarIntervalRef removeBelow below min", "[IntVarIntervalRef][host]") {
    Allocator allocator;
    IntVarIntervalRef v = allocator[allocator.intVarInterval(10)];

    REQUIRE(v.removeBelow(0) == DomainChange::none);
    REQUIRE(v.removeBelow(-5) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("IntVarIntervalRef removeBelow above max", "[IntVarIntervalRef][host]") {
    Allocator allocator;
    IntVarIntervalRef v = allocator[allocator.intVarInterval(10)];

    REQUIRE(v.removeBelow(15) == DomainChange::inconsistency);
}


TEST_CASE("IntVarIntervalRef removeAbove simple", "[IntVarIntervalRef][host]") {
    Allocator allocator;
    IntVarIntervalRef v = allocator[allocator.intVarInterval(10)];

    REQUIRE(v.removeAbove(5) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
}

TEST_CASE("IntVarIntervalRef removeAbove above max", "[IntVarIntervalRef][host]") {
    Allocator allocator;
    IntVarIntervalRef v = allocator[allocator.intVarInterval(10)];

    REQUIRE(v.removeAbove(9) == DomainChange::none);
    REQUIRE(v.removeAbove(15) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("IntVarIntervalRef removeAbove below min", "[IntVarIntervalRef][host]") {
    Allocator allocator;
    IntVarIntervalRef v = allocator[allocator.intVarInterval(10)];

    REQUIRE(v.removeAbove(-5) == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarIntervalRef fix simple", "[AtomicIntVarIntervalRef]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(10);
    IntVarIntervalRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.fix(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.fixed());
}

TEST_CASE("AtomicIntVarIntervalRef fix already fixed", "[AtomicIntVarIntervalRef]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(1);
    IntVarIntervalRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.fix(0);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.fixed());
}

TEST_CASE("AtomicIntVarIntervalRef fix outside bounds", "[AtomicIntVarIntervalRef]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.fix(15);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarIntervalRef removeBelow simple", "[AtomicIntVarIntervalRef]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(10);
    IntVarIntervalRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.removeBelow(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
}

TEST_CASE("AtomicIntVarIntervalRef removeBelow below min", "[AtomicIntVarIntervalRef]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(10);
    IntVarIntervalRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.removeBelow(0) | v.removeBelow(-5);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntVarIntervalRef removeBelow above max", "[AtomicIntVarIntervalRef]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.removeBelow(15);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarIntervalRef removeAbove simple", "[AtomicIntVarIntervalRef]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(10);
    IntVarIntervalRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.removeAbove(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
}

TEST_CASE("AtomicIntVarIntervalRef removeAbove above max", "[AtomicIntVarIntervalRef]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(10);
    IntVarIntervalRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.removeAbove(9) | v.removeAbove(15);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntVarIntervalRef removeAbove below min", "[AtomicIntVarIntervalRef]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.removeAbove(-5);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarIntervalRef concurrent fix same", "[AtomicIntVarIntervalRef][concurrent]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(80);
    IntVarIntervalRef v = allocator[idx];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.fix(5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.fixed());
}

TEST_CASE("AtomicIntVarIntervalRef concurrent fix different", "[AtomicIntVarIntervalRef][concurrent]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(80);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.fix(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntVarIntervalRef concurrent removeBelow", "[AtomicIntVarIntervalRef][concurrent]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(80);
    IntVarIntervalRef v = allocator[idx];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 68);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 12);
}

TEST_CASE("AtomicIntVarIntervalRef concurrent removeBelow all", "[AtomicIntVarIntervalRef][concurrent]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(63);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntVarIntervalRef concurrent removeAbove", "[AtomicIntVarIntervalRef][concurrent]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(80);
    IntVarIntervalRef v = allocator[idx];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.removeAbove(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
}

TEST_CASE("AtomicIntVarIntervalRef concurrent removeAbove all", "[AtomicIntVarIntervalRef][concurrent]") {
    Allocator allocator;
    IntVarIntervalIdx idx = allocator.intVarInterval(6, 80);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarIntervalRef v = ctx[idx];
        return v.removeAbove(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}