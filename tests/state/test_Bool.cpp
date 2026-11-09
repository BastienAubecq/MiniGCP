#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "sycl/sycl.hpp"
#include "MiniGCP/state/Bool.hpp"
#include "utils.hpp"

using namespace std;
using namespace cp;


TEST_CASE("HostBool allocation", "[HostBool][host]") {
    Allocator allocator;
    HostBool& v = Bool(allocator)[allocator];

    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 1);
    REQUIRE(v.size() == 2);
    REQUIRE_FALSE(v.value());
    REQUIRE(v.contains(0));
    REQUIRE(v.contains(1));
    REQUIRE_FALSE(v.contains(-1));
    REQUIRE_FALSE(v.contains(2));
}

TEST_CASE("HostBool allocation array", "[HostBool][host]") {
    Allocator allocator;
    BoolPtr arr(allocator, 10);

    for (int i = 0; i < 10; i++) {
        HostBool& v = arr[i][allocator];
        REQUIRE(v.min() == 0);
        REQUIRE(v.max() == 1);
        REQUIRE(v.size() == 2);
        REQUIRE_FALSE(v.value());
        REQUIRE(v.contains(0));
        REQUIRE(v.contains(1));
        REQUIRE_FALSE(v.contains(-1));
        REQUIRE_FALSE(v.contains(2));
    }
}


TEST_CASE("HostBool remove simple", "[HostBool][host]") {
    Allocator allocator;
    Bool idx0(allocator);
    Bool idx1(allocator);
    HostBool& v0 = idx0[allocator];
    HostBool& v1 = idx1[allocator];

    REQUIRE(v0.remove(0) == DomainChange::changed);
    REQUIRE(v1.remove(1) == DomainChange::changed);
    REQUIRE(v0.min() == 1);
    REQUIRE(v0.max() == 1);
    REQUIRE(v0.size() == 1);
    REQUIRE(v0.value() == 1);
    REQUIRE_FALSE(v0.contains(0));
    REQUIRE(v0.contains(1));
    REQUIRE(v1.min() == 0);
    REQUIRE(v1.max() == 0);
    REQUIRE(v1.size() == 1);
    REQUIRE(v1.value() == 0);
    REQUIRE_FALSE(v1.contains(1));
    REQUIRE(v1.contains(0));
}

TEST_CASE("HostBool remove outside bounds", "[HostBool][host]") {
    Allocator allocator;
    HostBool& v = Bool(allocator)[allocator];

    REQUIRE(v.remove(-1) == DomainChange::none);
    REQUIRE(v.remove(2) == DomainChange::none);
    REQUIRE(v.size() == 2);
}

TEST_CASE("HostBool remove already removed", "[HostBool][host]") {
    Allocator allocator;
    Bool idx0(allocator);
    Bool idx1(allocator);
    HostBool& v0 = idx0[allocator];
    HostBool& v1 = idx1[allocator];
    v0.remove(0);
    v1.remove(1);

    REQUIRE(v0.remove(0) == DomainChange::none);
    REQUIRE(v1.remove(1) == DomainChange::none);
    REQUIRE(v0.size() == 1);
    REQUIRE(v1.size() == 1);
}

TEST_CASE("HostBool remove last", "[HostBool][host]") {
    Allocator allocator;
    Bool idx0(allocator);
    Bool idx1(allocator);
    HostBool& v0 = idx0[allocator];
    HostBool& v1 = idx1[allocator];
    v0.remove(0);
    v1.remove(1);

    REQUIRE((v0.remove(1) & DomainChange::inconsistency) == DomainChange::inconsistency);
    REQUIRE((v1.remove(0) & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("HostBool fix simple", "[HostBool][host]") {
    Allocator allocator;
    Bool idx0(allocator);
    Bool idx1(allocator);
    HostBool& v0 = idx0[allocator];
    HostBool& v1 = idx1[allocator];

    REQUIRE(v0.fix(0) == DomainChange::changed);
    REQUIRE(v1.fix(1) == DomainChange::changed);
    REQUIRE(v0.min() == 0);
    REQUIRE(v0.max() == 0);
    REQUIRE(v0.size() == 1);
    REQUIRE(v0.value() == 0);
    REQUIRE_FALSE(v0.contains(1));
    REQUIRE(v0.contains(0));
    REQUIRE(v1.min() == 1);
    REQUIRE(v1.max() == 1);
    REQUIRE(v1.size() == 1);
    REQUIRE(v1.value() == 1);
    REQUIRE_FALSE(v1.contains(0));
    REQUIRE(v1.contains(1));
}

TEST_CASE("HostBool fix already fixed", "[HostBool][host]") {
    Allocator allocator;
    Bool idx0(allocator);
    Bool idx1(allocator);
    HostBool& v0 = idx0[allocator];
    HostBool& v1 = idx1[allocator];
    v0.fix(0);
    v1.fix(1);

    REQUIRE(v0.fix(0) == DomainChange::none);
    REQUIRE(v1.fix(1) == DomainChange::none);
    REQUIRE(v0.value() == 0);
    REQUIRE(v1.value() == 1);
}

TEST_CASE("HostBool fix outside bounds", "[HostBool][host]") {
    Allocator allocator;
    HostBool& v = Bool(allocator)[allocator];

    REQUIRE((v.fix(-1) & DomainChange::inconsistency) == DomainChange::inconsistency);
    REQUIRE((v.fix(2) & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("HostBool fix removed", "[HostBool][host]") {
    Allocator allocator;
    Bool idx0(allocator);
    Bool idx1(allocator);
    HostBool& v0 = idx0[allocator];
    HostBool& v1 = idx1[allocator];
    v0.remove(0);
    v1.remove(1);

    REQUIRE((v0.fix(0) & DomainChange::inconsistency) == DomainChange::inconsistency);
    REQUIRE((v1.fix(1) & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("HostBool removeBelow simple", "[HostBool][host]") {
    Allocator allocator;
    HostBool& v = Bool(allocator)[allocator];

    REQUIRE(v.removeBelow(1) == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 1);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 1);
    REQUIRE_FALSE(v.contains(0));
    REQUIRE(v.contains(1));
}

TEST_CASE("HostBool removeBelow below min", "[HostBool][host]") {
    Allocator allocator;
    HostBool& v = Bool(allocator)[allocator];
    v.remove(0);

    REQUIRE(v.removeBelow(-1) == DomainChange::none);
    REQUIRE(v.removeBelow(0) == DomainChange::none);
    REQUIRE(v.removeBelow(1) == DomainChange::none);
    REQUIRE(v.size() == 1);
}

TEST_CASE("HostBool removeBelow above max", "[HostBool][host]") {
    Allocator allocator;
    HostBool& v = Bool(allocator)[allocator];
    v.remove(1);

    REQUIRE((v.removeBelow(2) & DomainChange::inconsistency) == DomainChange::inconsistency);
    REQUIRE((v.removeBelow(1) & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("HostBool removeAbove simple", "[HostBool][host]") {
    Allocator allocator;
    HostBool& v = Bool(allocator)[allocator];

    REQUIRE(v.removeAbove(0) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 0);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 0);
    REQUIRE_FALSE(v.contains(1));
    REQUIRE(v.contains(0));
}

TEST_CASE("HostBool removeAbove above max", "[HostBool][host]") {
    Allocator allocator;
    HostBool& v = Bool(allocator)[allocator];
    v.remove(1);

    REQUIRE(v.removeAbove(2) == DomainChange::none);
    REQUIRE(v.removeAbove(1) == DomainChange::none);
    REQUIRE(v.removeAbove(0) == DomainChange::none);
    REQUIRE(v.size() == 1);
}

TEST_CASE("HostBool removeAbove below min", "[HostBool][host]") {
    Allocator allocator;
    HostBool& v = Bool(allocator)[allocator];
    v.remove(0);

    REQUIRE((v.removeAbove(-1) & DomainChange::inconsistency) == DomainChange::inconsistency);
    REQUIRE((v.removeAbove(0) & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceBool remove simple", "[DeviceBool]") {
    Allocator allocator;
    Bool idx0(allocator);
    Bool idx1(allocator);
    HostBool& v0 = idx0[allocator];
    HostBool& v1 = idx1[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx0[ctx].remove(0) & idx1[ctx].remove(1);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v0.min() == 1);
    REQUIRE(v0.max() == 1);
    REQUIRE(v0.size() == 1);
    REQUIRE(v0.value() == 1);
    REQUIRE_FALSE(v0.contains(0));
    REQUIRE(v0.contains(1));
    REQUIRE(v1.min() == 0);
    REQUIRE(v1.max() == 0);
    REQUIRE(v1.size() == 1);
    REQUIRE(v1.value() == 0);
    REQUIRE_FALSE(v1.contains(1));
    REQUIRE(v1.contains(0));
}

TEST_CASE("DeviceBool remove outside bounds", "[DeviceBool]") {
    Allocator allocator;
    Bool idx(allocator);
    HostBool& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].remove(-1) | idx[ctx].remove(2);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.size() == 2);
}

TEST_CASE("DeviceBool remove already removed", "[DeviceBool]") {
    Allocator allocator;
    Bool idx0(allocator);
    Bool idx1(allocator);
    HostBool& v0 = idx0[allocator];
    HostBool& v1 = idx1[allocator];
    v0.remove(0);
    v1.remove(1);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx0[ctx].remove(0) | idx1[ctx].remove(1);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v0.size() == 1);
    REQUIRE(v1.size() == 1);
}

TEST_CASE("DeviceBool remove last", "[DeviceBool]") {
    Allocator allocator;
    Bool idx0(allocator);
    Bool idx1(allocator);
    idx0[allocator].remove(0);
    idx1[allocator].remove(1);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx0[ctx].remove(1) & idx1[ctx].remove(0);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceBool fix simple", "[DeviceBool]") {
    Allocator allocator;
    Bool idx0(allocator);
    Bool idx1(allocator);
    HostBool& v0 = idx0[allocator];
    HostBool& v1 = idx1[allocator];
    
    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx0[ctx].fix(0) & idx1[ctx].fix(1);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v0.min() == 0);
    REQUIRE(v0.max() == 0);
    REQUIRE(v0.size() == 1);
    REQUIRE(v0.value() == 0);
    REQUIRE_FALSE(v0.contains(1));
    REQUIRE(v0.contains(0));
    REQUIRE(v1.min() == 1);
    REQUIRE(v1.max() == 1);
    REQUIRE(v1.size() == 1);
    REQUIRE(v1.value() == 1);
    REQUIRE_FALSE(v1.contains(0));
    REQUIRE(v1.contains(1));
}

TEST_CASE("DeviceBool fix already fixed", "[DeviceBool]") {
    Allocator allocator;
    Bool idx0(allocator);
    Bool idx1(allocator);
    HostBool& v0 = idx0[allocator];
    HostBool& v1 = idx1[allocator];
    v0.fix(0);
    v1.fix(1);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx0[ctx].fix(0) | idx1[ctx].fix(1);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v0.value() == 0);
    REQUIRE(v1.value() == 1);
}

TEST_CASE("DeviceBool fix outside bounds", "[DeviceBool]") {
    Allocator allocator;
    Bool idx(allocator);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].fix(-1) & idx[ctx].fix(2);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("DeviceBool fix removed", "[DeviceBool]") {
    Allocator allocator;
    Bool idx0(allocator);
    Bool idx1(allocator);
    idx0[allocator].remove(0);
    idx1[allocator].remove(1);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx0[ctx].fix(0) & idx1[ctx].fix(1);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceBool removeBelow simple", "[DeviceBool]") {
    Allocator allocator;
    Bool idx(allocator);
    HostBool& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeBelow(1);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 1);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 1);
    REQUIRE_FALSE(v.contains(0));
    REQUIRE(v.contains(1));
}

TEST_CASE("DeviceBool removeBelow below min", "[DeviceBool]") {
    Allocator allocator;
    Bool idx(allocator);
    HostBool& v = idx[allocator];
    v.remove(0);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeBelow(-1) | idx[ctx].removeBelow(0) | idx[ctx].removeBelow(1);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.size() == 1);
}

TEST_CASE("DeviceBool removeBelow above max", "[DeviceBool]") {
    Allocator allocator;
    Bool idx(allocator);
    idx[allocator].remove(1);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeBelow(2) & idx[ctx].removeBelow(1);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceBool removeAbove simple", "[DeviceBool]") {
    Allocator allocator;
    Bool idx(allocator);
    HostBool& v = idx[allocator];

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeAbove(0);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 0);
    REQUIRE(v.size() == 1);
    REQUIRE(v.value() == 0);
    REQUIRE_FALSE(v.contains(1));
    REQUIRE(v.contains(0));
}

TEST_CASE("DeviceBool removeAbove above max", "[DeviceBool]") {
    Allocator allocator;
    Bool idx(allocator);
    HostBool& v = idx[allocator];
    v.remove(1);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeAbove(2) | idx[ctx].removeAbove(1) | idx[ctx].removeAbove(0);
    })[0];

    REQUIRE(c == DomainChange::none);
    REQUIRE(v.size() == 1);
}

TEST_CASE("DeviceBool removeAbove below min", "[DeviceBool]") {
    Allocator allocator;
    Bool idx(allocator);
    idx[allocator].remove(0);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return idx[ctx].removeAbove(-1) & idx[ctx].removeAbove(0);
    })[0];

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}


TEST_CASE("DeviceBool concurrent remove all", "[DeviceBool][concurrent]") {
    Allocator allocator;
    Bool idx(allocator);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 2, [=](Context ctx) {
        return idx[ctx].remove(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("DeviceBool concurrent fix different", "[DeviceBool][concurrent]") {
    Allocator allocator;
    Bool idx(allocator);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 2, [=](Context ctx) {
        return idx[ctx].fix(ctx.item.get_global_id(0));
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}