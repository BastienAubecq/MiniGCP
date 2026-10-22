#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <span>
#include "sycl/sycl.hpp"
#include "MiniGCP/Model.hpp"
#include "MiniGCP/solver/Context.hpp"
#include "MiniGCP/state/IntVarBitsetIdx.hpp"
#include "MiniGCP/state/IntVarBitsetRef.hpp"
#include "MiniGCP/state/DomainChange.hpp"
#include "utils.hpp"

using namespace std;
using namespace cp;


TEST_CASE("IntVarBitsetRef allocation size simple", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
    REQUIRE_FALSE(v.fixed());
    for (int i = 0; i < 10; i++) REQUIRE(v.contains(i));
    REQUIRE_FALSE(v.contains(-1));
    REQUIRE_FALSE(v.contains(10));
}

TEST_CASE("IntVarBitsetRef allocation size multiple words", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(70)];

    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 69);
    REQUIRE(v.size() == 70);
    for (int i = 0; i <= 69; i++) REQUIRE(v.contains(i));
    REQUIRE_FALSE(v.contains(-1));
    REQUIRE_FALSE(v.contains(70));
}

TEST_CASE("IntVarBitsetRef allocation min max", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(5, 10)];

    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 10);
    REQUIRE(v.size() == 6);
    for (int i = 5; i <= 10; i++) REQUIRE(v.contains(i));
    REQUIRE_FALSE(v.contains(4));
    REQUIRE_FALSE(v.contains(11));
}

TEST_CASE("IntVarBitsetRef allocation values", "[IntVarBitsetRef][host]") {
    vector<int32_t> values{ -5, -4, -1, 2, 3, 4, 10, 40 };
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(values)];

    REQUIRE(v.min() == -5);
    REQUIRE(v.max() == 40);
    REQUIRE(v.size() == values.size());
    for (int32_t val : values) REQUIRE(v.contains(val));
    REQUIRE_FALSE(v.contains(0));
    REQUIRE_FALSE(v.contains(1));
    REQUIRE_FALSE(v.contains(-6));
    REQUIRE_FALSE(v.contains(41));
}


TEST_CASE("IntVarBitsetRef remove simple", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.remove(5) == DomainChange::changed);
    REQUIRE(v.size() == 9);
    REQUIRE_FALSE(v.contains(5));
    for (int32_t i = 0; i < 5; i++) REQUIRE(v.contains(i));
    for (int32_t i = 6; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("IntVarBitsetRef remove outside bounds", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.remove(-1) == DomainChange::none);
    REQUIRE(v.remove(10) == DomainChange::none);
    REQUIRE(v.size() == 10);
}

TEST_CASE("IntVarBitsetRef remove already removed", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];
    v.remove(5);

    REQUIRE(v.remove(5) == DomainChange::none);
    REQUIRE(v.size() == 9);
}

TEST_CASE("IntVarBitsetRef remove min", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.remove(0) == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 9);
}

TEST_CASE("IntVarBitsetRef remove max", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.remove(9) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 8);
    REQUIRE(v.size() == 9);
}

TEST_CASE("IntVarBitsetRef remove last", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(1)];

    REQUIRE(v.remove(0) == DomainChange::inconsistency);
}


TEST_CASE("IntVarBitsetRef fix simple", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.fix(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.fixed());
    REQUIRE(v.contains(5));
    for (int32_t i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int32_t i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("IntVarBitsetRef fix already fixed", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(1)];

    REQUIRE(v.fix(0) == DomainChange::none);
    REQUIRE(v.fixed());
}

TEST_CASE("IntVarBitsetRef fix outside bounds", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.fix(15) == DomainChange::inconsistency);
}

TEST_CASE("IntVarBitsetRef fix removed", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];
    v.remove(5);

    REQUIRE(v.fix(5) == DomainChange::inconsistency);
}


TEST_CASE("IntVarBitsetRef removeBelow simple", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.removeBelow(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 5; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("IntVarBitsetRef removeBelow below min", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.removeBelow(0) == DomainChange::none);
    REQUIRE(v.removeBelow(-5) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("IntVarBitsetRef removeBelow above max", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.removeBelow(15) == DomainChange::inconsistency);
}


TEST_CASE("IntVarBitsetRef removeAbove simple", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.removeAbove(5) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
    for (int i = 0; i <= 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("IntVarBitsetRef removeAbove above max", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.removeAbove(9) == DomainChange::none);
    REQUIRE(v.removeAbove(15) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("IntVarBitsetRef removeAbove below min", "[IntVarBitsetRef][host]") {
    Allocator allocator;
    IntVarBitsetRef v = allocator[allocator.intVarBitset(10)];

    REQUIRE(v.removeAbove(-5) == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarBitsetRef remove simple", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.remove(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.size() == 9);
    REQUIRE_FALSE(v.contains(5));
    for (int32_t i = 0; i < 5; i++) REQUIRE(v.contains(i));
    for (int32_t i = 6; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("AtomicIntVarBitsetRef remove outside bounds", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.remove(-1) | v.remove(10);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntVarBitsetRef remove already removed", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);
    IntVarBitsetRef v = allocator[idx];
    v.remove(5);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.remove(5);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.size() == 9);
}

TEST_CASE("AtomicIntVarBitsetRef remove min", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.remove(0);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 9);
}

TEST_CASE("AtomicIntVarBitsetRef remove max", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.remove(9);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 8);
    REQUIRE(v.size() == 9);
}

TEST_CASE("AtomicIntVarBitsetRef remove last", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(1);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.remove(0);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarBitsetRef fix simple", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
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

TEST_CASE("AtomicIntVarBitsetRef fix already fixed", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(1);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.fix(0);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.fixed());
}

TEST_CASE("AtomicIntVarBitsetRef fix outside bounds", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.fix(15);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntVarBitsetRef fix removed", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);
    IntVarBitsetRef v = allocator[idx];
    v.remove(5);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.fix(5);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarBitsetRef removeBelow simple", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.removeBelow(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 5; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("AtomicIntVarBitsetRef removeBelow below min", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.removeBelow(0) | v.removeBelow(-5);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntVarBitsetRef removeBelow above max", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.removeBelow(15);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarBitsetRef removeAbove simple", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.removeAbove(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
    for (int i = 0; i <= 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("AtomicIntVarBitsetRef removeAbove above max", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.removeAbove(9) | v.removeAbove(15);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntVarBitsetRef removeAbove below min", "[AtomicIntVarBitsetRef]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.removeAbove(-5);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntVarBitsetRef concurrent remove", "[AtomicIntVarBitsetRef][concurrent]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(80);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.remove(ctx.item.get_global_id(0));
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 64);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 16);
    for (int i = 0; i < 64; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 64; i < 80; i++) REQUIRE(v.contains(i));
}

TEST_CASE("AtomicIntVarBitsetRef concurrent remove all", "[AtomicIntVarBitsetRef][concurrent]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(64);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.remove(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntVarBitsetRef concurrent fix same", "[AtomicIntVarBitsetRef][concurrent]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(80);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.fix(5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.fixed());
}

TEST_CASE("AtomicIntVarBitsetRef concurrent fix different", "[AtomicIntVarBitsetRef][concurrent]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(80);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.fix(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntVarBitsetRef concurrent removeBelow", "[AtomicIntVarBitsetRef][concurrent]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(80);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 68);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 12);
}

TEST_CASE("AtomicIntVarBitsetRef concurrent removeBelow all", "[AtomicIntVarBitsetRef][concurrent]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(63);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntVarBitsetRef concurrent removeAbove", "[AtomicIntVarBitsetRef][concurrent]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(80);
    IntVarBitsetRef v = allocator[idx];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.removeAbove(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
}

TEST_CASE("AtomicIntVarBitsetRef concurrent removeAbove all", "[AtomicIntVarBitsetRef][concurrent]") {
    Allocator allocator;
    IntVarBitsetIdx idx = allocator.intVarBitset(6, 80);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        AtomicIntVarBitsetRef v = ctx[idx];
        return v.removeAbove(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}