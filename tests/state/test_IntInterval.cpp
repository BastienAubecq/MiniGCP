#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "sycl/sycl.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "utils.hpp"

using namespace std;
using namespace cp;


TEST_CASE("HostIntInterval allocation size", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
    REQUIRE_FALSE(v.value());
    for (int i = 0; i < 10; i++) REQUIRE(v.contains(i));
    REQUIRE_FALSE(v.contains(-1));
    REQUIRE_FALSE(v.contains(10));
}

TEST_CASE("HostIntInterval allocation min max", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 5, 10)[allocator];

    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 10);
    REQUIRE(v.size() == 6);
    for (int i = 5; i <= 10; i++) REQUIRE(v.contains(i));
    REQUIRE_FALSE(v.contains(4));
    REQUIRE_FALSE(v.contains(11));
}

TEST_CASE("HostIntInterval allocation array", "[HostIntInterval][host]") {
    Allocator allocator;
    IntIntervalPtr arr(allocator, 10, 10);

    for (int i = 0; i < 10; i++) {
        HostIntInterval& v = arr[i][allocator];
        REQUIRE(v.min() == 0);
        REQUIRE(v.max() == 9);
        REQUIRE(v.size() == 10);
        REQUIRE_FALSE(v.value());
        for (int i = 0; i < 10; i++) REQUIRE(v.contains(i));
        REQUIRE_FALSE(v.contains(-1));
        REQUIRE_FALSE(v.contains(10));
    }
}


TEST_CASE("HostIntInterval remove min", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.remove(0) == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 9);
}

TEST_CASE("HostIntInterval remove max", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.remove(9) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 8);
    REQUIRE(v.size() == 9);
}


TEST_CASE("HostIntInterval fix simple", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.fix(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 5);
    REQUIRE(v.contains(5));
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("HostIntInterval fix already fixed", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 1)[allocator];

    REQUIRE(v.fix(0) == DomainChange::none);
    REQUIRE(v.value() == 0);
}

TEST_CASE("HostIntInterval fix outside bounds", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE((v.fix(15) & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("HostIntInterval removeBelow simple", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.removeBelow(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 5; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("HostIntInterval removeBelow below min", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.removeBelow(0) == DomainChange::none);
    REQUIRE(v.removeBelow(-5) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("HostIntInterval removeBelow above max", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE((v.removeBelow(15) & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("HostIntInterval removeAbove simple", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.removeAbove(5) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
    for (int i = 0; i <= 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("HostIntInterval removeAbove above max", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE(v.removeAbove(9) == DomainChange::none);
    REQUIRE(v.removeAbove(15) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("HostIntInterval removeAbove below min", "[HostIntInterval][host]") {
    Allocator allocator;
    HostIntInterval& v = IntInterval(allocator, 10)[allocator];

    REQUIRE((v.removeAbove(-5) & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceIntInterval remove min", "[DeviceIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);
    HostIntInterval& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].remove(0);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 9);
}

TEST_CASE("DeviceIntInterval remove max", "[DeviceIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);
    HostIntInterval& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].remove(9);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 8);
    REQUIRE(v.size() == 9);
}


TEST_CASE("DeviceIntInterval fix simple", "[DeviceIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);
    HostIntInterval& v = idx[allocator];

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

TEST_CASE("DeviceIntInterval fix already fixed", "[DeviceIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 1);
    HostIntInterval& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].fix(0);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.value() == 0);
}

TEST_CASE("DeviceIntInterval fix outside bounds", "[DeviceIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].fix(15);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceIntInterval removeBelow simple", "[DeviceIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);
    HostIntInterval& v = idx[allocator];

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

TEST_CASE("DeviceIntInterval removeBelow below min", "[DeviceIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);
    HostIntInterval& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeBelow(0) | idx[ctx].removeBelow(-5);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("DeviceIntInterval removeBelow above max", "[DeviceIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeBelow(15);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceIntInterval removeAbove simple", "[DeviceIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);
    HostIntInterval& v = idx[allocator];

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

TEST_CASE("DeviceIntInterval removeAbove above max", "[DeviceIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);
    HostIntInterval& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeAbove(9) | idx[ctx].removeAbove(15);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("DeviceIntInterval removeAbove below min", "[DeviceIntInterval]") {
    Allocator allocator;
    IntInterval idx(allocator, 10);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeAbove(-5);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceIntInterval concurrent fix same", "[DeviceIntInterval][concurrent]") {
    Allocator allocator;
    IntInterval idx(allocator, 80);
    HostIntInterval& v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].fix(5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 5);
}

TEST_CASE("DeviceIntInterval concurrent fix different", "[DeviceIntInterval][concurrent]") {
    Allocator allocator;
    IntInterval idx(allocator, 80);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].fix(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("DeviceIntInterval concurrent removeBelow", "[DeviceIntInterval][concurrent]") {
    Allocator allocator;
    IntInterval idx(allocator, 80);
    HostIntInterval& v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 68);
    REQUIRE(v.max() == 79);
    REQUIRE(v.size() == 12);
}

TEST_CASE("DeviceIntInterval concurrent removeBelow all", "[DeviceIntInterval][concurrent]") {
    Allocator allocator;
    IntInterval idx(allocator, 64);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].removeBelow(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("DeviceIntInterval concurrent removeAbove", "[DeviceIntInterval][concurrent]") {
    Allocator allocator;
    IntInterval idx(allocator, 80);
    HostIntInterval& v = idx[allocator];

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].removeAbove(ctx.item.get_global_id(0) + 5);
    }));

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
}

TEST_CASE("DeviceIntInterval concurrent removeAbove all", "[DeviceIntInterval][concurrent]") {
    Allocator allocator;
    IntInterval idx(allocator, 64);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 64, [=](Context ctx) {
        return idx[ctx].removeAbove(ctx.item.get_global_id(0) - 5);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}