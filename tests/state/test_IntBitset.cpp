#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "sycl/sycl.hpp"
#include "MiniGCP/state/IntBitset.hpp"
#include "utils.hpp"

using namespace std;
using namespace cp;


TEST_CASE("IntBitset allocation size simple", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
    REQUIRE_FALSE(v.value());
    for (int i = 0; i < 10; i++) REQUIRE(v.contains(i));
    REQUIRE_FALSE(v.contains(-1));
    REQUIRE_FALSE(v.contains(10));
}

TEST_CASE("IntBitset allocation size multiple words", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 70)[allocator];

    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 69);
    REQUIRE(v.size() == 70);
    for (int i = 0; i <= 69; i++) REQUIRE(v.contains(i));
    REQUIRE_FALSE(v.contains(-1));
    REQUIRE_FALSE(v.contains(70));
}

TEST_CASE("IntBitset allocation values", "[IntBitset][host]") {
    vector<int32_t> values{ 0, 1, 4, 7, 8, 9, 15, 45 };
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, values)[allocator];

    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 45);
    REQUIRE(v.size() == values.size());
    for (int32_t val : values) REQUIRE(v.contains(val));
    REQUIRE_FALSE(v.contains(5));
    REQUIRE_FALSE(v.contains(6));
    REQUIRE_FALSE(v.contains(-1));
    REQUIRE_FALSE(v.contains(46));
}

TEST_CASE("IntBitset allocation array", "[IntBitset][host]") {
    Allocator allocator;
    IntBitsetArray arr(allocator, 10, 10);

    for (int i = 0; i < 10; i++) {
        HostIntBitsetRef v = arr[i][allocator];
        REQUIRE(v.min() == 0);
        REQUIRE(v.max() == 9);
        REQUIRE(v.size() == 10);
        REQUIRE_FALSE(v.value());
        for (int i = 0; i < 10; i++) REQUIRE(v.contains(i));
        REQUIRE_FALSE(v.contains(-1));
        REQUIRE_FALSE(v.contains(10));
    }
}


TEST_CASE("IntBitset remove simple", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.remove(5) == DomainChange::changed);
    REQUIRE(v.size() == 9);
    REQUIRE_FALSE(v.contains(5));
    for (int i = 0; i < 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("IntBitset remove outside bounds", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.remove(-1) == DomainChange::none);
    REQUIRE(v.remove(10) == DomainChange::none);
    REQUIRE(v.size() == 10);
}

TEST_CASE("IntBitset remove already removed", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];
    v.remove(5);

    REQUIRE(v.remove(5) == DomainChange::none);
    REQUIRE(v.size() == 9);
}

TEST_CASE("IntBitset remove min", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.remove(0) == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 9);
}

TEST_CASE("IntBitset remove max", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.remove(9) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 8);
    REQUIRE(v.size() == 9);
}

TEST_CASE("IntBitset remove last", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 1)[allocator];

    REQUIRE(v.remove(0) == DomainChange::inconsistency);
}


TEST_CASE("IntBitset fix simple", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.fix(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 5);
    REQUIRE(v.contains(5));
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("IntBitset fix already fixed", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 1)[allocator];

    REQUIRE(v.fix(0) == DomainChange::none);
    REQUIRE(v.value() == 0);
}

TEST_CASE("IntBitset fix outside bounds", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.fix(15) == DomainChange::inconsistency);
}

TEST_CASE("IntBitset fix removed", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];
    v.remove(5);

    REQUIRE(v.fix(5) == DomainChange::inconsistency);
}


TEST_CASE("IntBitset removeBelow simple", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.removeBelow(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 5; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("IntBitset removeBelow below min", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.removeBelow(0) == DomainChange::none);
    REQUIRE(v.removeBelow(-5) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("IntBitset removeBelow above max", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.removeBelow(15) == DomainChange::inconsistency);
}


TEST_CASE("IntBitset removeAbove simple", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.removeAbove(5) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
    for (int i = 0; i <= 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("IntBitset removeAbove above max", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.removeAbove(9) == DomainChange::none);
    REQUIRE(v.removeAbove(15) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("IntBitset removeAbove below min", "[IntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.removeAbove(-5) == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntBitset remove simple", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.remove(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.size() == 9);
    REQUIRE_FALSE(v.contains(5));
    for (int i = 0; i < 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("AtomicIntBitset remove outside bounds", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.remove(-1) | v.remove(10);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntBitset remove already removed", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];
    v.remove(5);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.remove(5);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.size() == 9);
}

TEST_CASE("AtomicIntBitset remove min", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.remove(0);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 9);
}

TEST_CASE("AtomicIntBitset remove max", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.remove(9);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 8);
    REQUIRE(v.size() == 9);
}

TEST_CASE("AtomicIntBitset remove last", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 1);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.remove(0);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntBitset fix simple", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.fix(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 5);
    REQUIRE(v.contains(5));
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("AtomicIntBitset fix already fixed", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 1);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.fix(0);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.value() == 0);
}

TEST_CASE("AtomicIntBitset fix outside bounds", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.fix(15);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntBitset fix removed", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];
    v.remove(5);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.fix(5);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntBitset removeBelow simple", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.removeBelow(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 5; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("AtomicIntBitset removeBelow below min", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.removeBelow(0) | v.removeBelow(-5);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntBitset removeBelow above max", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.removeBelow(15);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntBitset removeAbove simple", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.removeAbove(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
    for (int i = 0; i <= 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("AtomicIntBitset removeAbove above max", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.removeAbove(9) | v.removeAbove(15);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntBitset removeAbove below min", "[AtomicIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.removeAbove(-5);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntBitset concurrent remove", "[AtomicIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 80);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.remove(ctx.item.get_global_id(0));
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 64);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 16);
    for (int i = 0; i < 64; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 64; i < 80; i++) REQUIRE(v.contains(i));
}

TEST_CASE("AtomicIntBitset concurrent remove all", "[AtomicIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 64);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.remove(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntBitset concurrent fix same", "[AtomicIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 80);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.fix(5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 5);
}

TEST_CASE("AtomicIntBitset concurrent fix different", "[AtomicIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 80);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.fix(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntBitset concurrent removeBelow", "[AtomicIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 80);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 68);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 12);
}

TEST_CASE("AtomicIntBitset concurrent removeBelow all", "[AtomicIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 64);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntBitset concurrent removeAbove", "[AtomicIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 80);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.removeAbove(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
}

TEST_CASE("AtomicIntBitset concurrent removeAbove all", "[AtomicIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 64);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntBitsetRef v = idx[ctx];
        return v.removeAbove(ctx.item.get_global_id(0) - 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}