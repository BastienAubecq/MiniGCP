#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "sycl/sycl.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "utils.hpp"

using namespace std;
using namespace cp;


TEST_CASE("IntInterval allocation size", "[IntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
    REQUIRE_FALSE(v.value());
}

TEST_CASE("IntInterval allocation min max", "[IntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 5, 10)[allocator];

    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 10);
    REQUIRE(v.size() == 6);
}

TEST_CASE("IntInterval allocation array", "[IntInterval][host]") {
    Allocator allocator;
    IntIntervalArray arr(allocator, 10, 10);

    for (int i = 0; i < 10; i++) {
        HostIntInterval& v = arr[i][allocator];
        REQUIRE(v.min() == 0);
        REQUIRE(v.max() == 9);
        REQUIRE(v.size() == 10);
        REQUIRE_FALSE(v.value());
    }
}


TEST_CASE("IntInterval fix simple", "[IntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.fix(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 5);
}

TEST_CASE("IntInterval fix already fixed", "[IntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 1)[allocator];

    REQUIRE(v.fix(0) == DomainChange::none);
    REQUIRE(v.value() == 0);
}

TEST_CASE("IntInterval fix outside bounds", "[IntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.fix(15) == DomainChange::inconsistency);
}


TEST_CASE("IntInterval removeBelow simple", "[IntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.removeBelow(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
}

TEST_CASE("IntInterval removeBelow below min", "[IntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.removeBelow(0) == DomainChange::none);
    REQUIRE(v.removeBelow(-5) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("IntInterval removeBelow above max", "[IntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.removeBelow(15) == DomainChange::inconsistency);
}


TEST_CASE("IntInterval removeAbove simple", "[IntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.removeAbove(5) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
}

TEST_CASE("IntInterval removeAbove above max", "[IntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.removeAbove(9) == DomainChange::none);
    REQUIRE(v.removeAbove(15) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("IntInterval removeAbove below min", "[IntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.removeAbove(-5) == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntInterval fix simple", "[AtomicIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);
    HostIntInterval& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.fix(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 5);
}

TEST_CASE("AtomicIntInterval fix already fixed", "[AtomicIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 1);
    HostIntInterval& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.fix(0);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.value() == 0);
}

TEST_CASE("AtomicIntInterval fix outside bounds", "[AtomicIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.fix(15);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntInterval removeBelow simple", "[AtomicIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);
    HostIntInterval& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.removeBelow(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
}

TEST_CASE("AtomicIntInterval removeBelow below min", "[AtomicIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);
    HostIntInterval& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.removeBelow(0) | v.removeBelow(-5);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntInterval removeBelow above max", "[AtomicIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.removeBelow(15);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntInterval removeAbove simple", "[AtomicIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);
    HostIntInterval& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.removeAbove(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
}

TEST_CASE("AtomicIntInterval removeAbove above max", "[AtomicIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);
    HostIntInterval& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.removeAbove(9) | v.removeAbove(15);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("AtomicIntInterval removeAbove below min", "[AtomicIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.removeAbove(-5);
    })[0];

    REQUIRE(c == DomainChange::inconsistency);
}


TEST_CASE("AtomicIntInterval concurrent fix same", "[AtomicIntInterval][concurrent]") {
    Allocator allocator;
    IntInterval idx(allocator, 80);
    HostIntInterval& v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.fix(5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 5);
}

TEST_CASE("AtomicIntInterval concurrent fix different", "[AtomicIntInterval][concurrent]") {
    Allocator allocator;
    IntInterval idx(allocator, 80);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.fix(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntInterval concurrent removeBelow", "[AtomicIntInterval][concurrent]") {
    Allocator allocator;
    IntInterval idx(allocator, 80);
    HostIntInterval& v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 68);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 12);
}

TEST_CASE("AtomicIntInterval concurrent removeBelow all", "[AtomicIntInterval][concurrent]") {
    Allocator allocator;
    IntInterval idx(allocator, 64);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("AtomicIntInterval concurrent removeAbove", "[AtomicIntInterval][concurrent]") {
    Allocator allocator;
    IntInterval idx(allocator, 80);
    HostIntInterval& v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.removeAbove(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
}

TEST_CASE("AtomicIntInterval concurrent removeAbove all", "[AtomicIntInterval][concurrent]") {
    Allocator allocator;
    IntInterval idx(allocator, 64);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        DeviceIntInterval& v = idx[ctx];
        return v.removeAbove(ctx.item.get_global_id(0) - 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}