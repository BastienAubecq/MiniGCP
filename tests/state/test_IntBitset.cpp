#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "sycl/sycl.hpp"
#include "MiniGCP/state/IntBitset.hpp"
#include "utils.hpp"

using namespace std;
using namespace cp;


TEST_CASE("HostIntBitset allocation size simple", "[HostIntBitset][host]") {
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

TEST_CASE("HostIntBitset allocation size multiple words", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 70)[allocator];

    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 69);
    REQUIRE(v.size() == 70);
    for (int i = 0; i <= 69; i++) REQUIRE(v.contains(i));
    REQUIRE_FALSE(v.contains(-1));
    REQUIRE_FALSE(v.contains(70));
}

TEST_CASE("HostIntBitset allocation values", "[HostIntBitset][host]") {
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

TEST_CASE("HostIntBitset allocation array", "[HostIntBitset][host]") {
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


TEST_CASE("HostIntBitset remove simple", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.remove(5) == DomainChange::changed);
    REQUIRE(v.size() == 9);
    REQUIRE_FALSE(v.contains(5));
    for (int i = 0; i < 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("HostIntBitset remove outside bounds", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.remove(-1) == DomainChange::none);
    REQUIRE(v.remove(10) == DomainChange::none);
    REQUIRE(v.size() == 10);
}

TEST_CASE("HostIntBitset remove already removed", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];
    v.remove(5);

    REQUIRE(v.remove(5) == DomainChange::none);
    REQUIRE(v.size() == 9);
}

TEST_CASE("HostIntBitset remove min", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.remove(0) == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 9);
}

TEST_CASE("HostIntBitset remove max", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.remove(9) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 8);
    REQUIRE(v.size() == 9);
}

TEST_CASE("HostIntBitset remove last", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 1)[allocator];

    REQUIRE((v.remove(0) & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("HostIntBitset fix simple", "[HostIntBitset][host]") {
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

TEST_CASE("HostIntBitset fix already fixed", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 1)[allocator];

    REQUIRE(v.fix(0) == DomainChange::none);
    REQUIRE(v.value() == 0);
}

TEST_CASE("HostIntBitset fix outside bounds", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE((v.fix(15) & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("HostIntBitset fix removed", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];
    v.remove(5);

    REQUIRE((v.fix(5) & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("HostIntBitset removeBelow simple", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.removeBelow(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 5; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("HostIntBitset removeBelow below min", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.removeBelow(0) == DomainChange::none);
    REQUIRE(v.removeBelow(-5) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("HostIntBitset removeBelow above max", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE((v.removeBelow(15) & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("HostIntBitset removeAbove simple", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.removeAbove(5) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
    for (int i = 0; i <= 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("HostIntBitset removeAbove above max", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE(v.removeAbove(9) == DomainChange::none);
    REQUIRE(v.removeAbove(15) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("HostIntBitset removeAbove below min", "[HostIntBitset][host]") {
    Allocator allocator;
    HostIntBitsetRef v = IntBitset(allocator, 10)[allocator];

    REQUIRE((v.removeAbove(-5) & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceIntBitset remove simple", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].remove(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.size() == 9);
    REQUIRE_FALSE(v.contains(5));
    for (int i = 0; i < 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("DeviceIntBitset remove outside bounds", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].remove(-1) | idx[ctx].remove(10);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.size() == 10);
}

TEST_CASE("DeviceIntBitset remove already removed", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];
    v.remove(5);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].remove(5);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.size() == 9);
}

TEST_CASE("DeviceIntBitset remove min", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].remove(0);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 9);
}

TEST_CASE("DeviceIntBitset remove max", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].remove(9);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 8);
    REQUIRE(v.size() == 9);
}

TEST_CASE("DeviceIntBitset remove last", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 1);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].remove(0);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceIntBitset fix simple", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].fix(5);
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

TEST_CASE("DeviceIntBitset fix already fixed", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 1);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].fix(0);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.value() == 0);
}

TEST_CASE("DeviceIntBitset fix outside bounds", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].fix(15);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("DeviceIntBitset fix removed", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];
    v.remove(5);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].fix(5);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceIntBitset removeBelow simple", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeBelow(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 5; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("DeviceIntBitset removeBelow below min", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeBelow(0) | idx[ctx].removeBelow(-5);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("DeviceIntBitset removeBelow above max", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeBelow(15);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceIntBitset removeAbove simple", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeAbove(5);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
    for (int i = 0; i <= 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("DeviceIntBitset removeAbove above max", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeAbove(9) | idx[ctx].removeAbove(15);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("DeviceIntBitset removeAbove below min", "[DeviceIntBitset]") {
    Allocator allocator;
    IntBitset idx(allocator, 10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeAbove(-5);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceIntBitset concurrent remove", "[DeviceIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 80);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].remove(ctx.item.get_global_id(0));
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 64);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 16);
    for (int i = 0; i < 64; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 64; i < 80; i++) REQUIRE(v.contains(i));
}

TEST_CASE("DeviceIntBitset concurrent remove all", "[DeviceIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 64);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].remove(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("DeviceIntBitset concurrent fix same", "[DeviceIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 80);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].fix(5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 5);
}

TEST_CASE("DeviceIntBitset concurrent fix different", "[DeviceIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 80);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].fix(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("DeviceIntBitset concurrent removeBelow", "[DeviceIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 80);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 68);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 12);
}

TEST_CASE("DeviceIntBitset concurrent removeBelow all", "[DeviceIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 64);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("DeviceIntBitset concurrent removeAbove", "[DeviceIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 80);
    HostIntBitsetRef v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].removeAbove(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
}

TEST_CASE("DeviceIntBitset concurrent removeAbove all", "[DeviceIntBitset][concurrent]") {
    Allocator allocator;
    IntBitset idx(allocator, 64);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].removeAbove(ctx.item.get_global_id(0) - 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}