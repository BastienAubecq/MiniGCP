#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <array>
#include "MiniGCP/constraints/Cumulative.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;

TEST_CASE("Cumulative Gay et al. CP2015 Fig 1 example (parallel)", "[Cumulative][concurrent]") {
    Allocator allocator;
    IntIntervalPtr s(allocator, 0, 10, 3);
    s[0][allocator].fix(0);
    s[1][allocator].fix(4);
    s[2][allocator].removeAbove(8);
    Ptr<int32_t> d(allocator, array { 2, 2, 5 });
    Ptr<int32_t> c(allocator, array { 3, 2, 2 });
    Cumulative constraint(allocator, s, d, c, 3, 3);

    DomainChange dc = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(dc == DomainChange::changed);
    REQUIRE(s[2][allocator].min() == 6);
    REQUIRE(s[2][allocator].max() == 8);
}

TEST_CASE("Cumulative right-to-left upper bound filtering", "[Cumulative][concurrent]") {
    // Symmetric of Fig 1:
    // Capacity C = 3
    // Task A (0): S in [11, 11], p = 2, h = 3 -> mandatory [11, 13[ of height 3
    // Task B (1): S in [7, 7],   p = 2, h = 2 -> mandatory [7, 9[ of height 2
    // Task C (2): S in [0, 8],   p = 5, h = 2 -> lct = 13; pulled from 13 -> 11 -> 7 => lst = 2
    Allocator allocator;
    IntIntervalPtr s(allocator, 0, 15, 3);
    s[0][allocator].fix(11);
    s[1][allocator].fix(7);
    s[2][allocator].removeAbove(8);
    Ptr<int32_t> d(allocator, array { 2, 2, 5 });
    Ptr<int32_t> c(allocator, array { 3, 2, 2 });
    Cumulative constraint(allocator, s, d, c, 3, 3);

    DomainChange dc = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(dc == DomainChange::changed);
    REQUIRE(s[2][allocator].min() == 0);
    REQUIRE(s[2][allocator].max() == 2);
}

TEST_CASE("Cumulative unfixed tasks with mandatory parts (no self-conflict)", "[Cumulative][concurrent]") {
    // Capacity C = 4
    // Task 0: S in [1, 2], p = 4, h = 3 -> mandatory [2, 5[ of height 3
    // Task 1: S in [0, 6], p = 3, h = 2 -> conflicts with [2, 5[, but est=0..2 is only 2 units (<3),
    // so Task 1 is pushed to 5 (S_1 in [5, 6])!
    // Meanwhile Task 0 has mandatory part [2, 5[ of height 3; it must NOT conflict with itself!
    Allocator allocator;
    IntIntervalPtr s(allocator, 0, 10, 2);
    s[0][allocator].removeBelow(1);
    s[0][allocator].removeAbove(2);
    s[1][allocator].removeAbove(6);
    Ptr<int32_t> d(allocator, array { 4, 3});
    Ptr<int32_t> c(allocator, array { 3, 2 });
    Cumulative constraint(allocator, s, d, c, 2, 4);

    DomainChange dc = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(dc == DomainChange::changed);
    REQUIRE(s[0][allocator].min() == 1);
    REQUIRE(s[0][allocator].max() == 2);
    REQUIRE(s[1][allocator].min() == 5);
    REQUIRE(s[1][allocator].max() == 6);
}

TEST_CASE("Cumulative overload inconsistency", "[Cumulative][concurrent]") {
    // Capacity C = 4
    // Task 0: S in [0, 1], p = 4, h = 2 -> mandatory [1, 4[ of height 2
    // Task 1: S in [1, 2], p = 4, h = 2 -> mandatory [2, 5[ of height 2
    // Task 2: S in [0, 2], p = 4, h = 2 -> mandatory [2, 4[ of height 2
    // Total height on [2, 4[ is 6 > 4 -> inconsistency!
    Allocator allocator;
    IntIntervalPtr s(allocator, 0, 10, 3);
    s[0][allocator].removeAbove(1);
    s[1][allocator].removeBelow(1);
    s[1][allocator].removeAbove(2);
    s[2][allocator].removeAbove(2);
    Ptr<int32_t> d(allocator, array { 4, 4, 4 });
    Ptr<int32_t> c(allocator, array { 2, 2, 2 });
    Cumulative constraint(allocator, s, d, c, 3, 4);

    DomainChange dc = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE((dc & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("Cumulative task wipeout inconsistency", "[Cumulative][concurrent]") {
    // Capacity C = 3
    // Task 0: fixed at 2, p = 4, h = 2 -> mandatory [2, 6[
    // Task 1: S in [0, 3], p = 4, h = 2 -> cannot fit before 2 (needs 4 units) nor after 2 (lst=3 < 6)
    Allocator allocator;
    IntIntervalPtr s(allocator, 0, 10, 2);
    s[0][allocator].fix(2);
    s[1][allocator].removeAbove(3);
    Ptr<int32_t> d(allocator, array { 4, 4 });
    Ptr<int32_t> c(allocator, array { 2, 2 });
    Cumulative constraint(allocator, s, d, c, 2, 3);

    DomainChange dc = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE((dc & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("Cumulative no mandatory parts", "[Cumulative][concurrent]") {
    Allocator allocator;
    IntIntervalPtr s(allocator, 0, 20, 5);
    Ptr<int32_t> d(allocator, array { 3, 4, 2, 5, 3 });
    Ptr<int32_t> c(allocator, array { 2, 1, 2, 3, 1 });
    Cumulative constraint(allocator, s, d, c, 5, 4);

    DomainChange dc = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(dc == DomainChange::none);
}

TEST_CASE("Cumulative many tasks concurrent (multi-chunk scan and sort)", "[Cumulative][concurrent]") {
    // 50 tasks:
    // Tasks 0..39 are fixed in pairs at [2*k, 2*k + 2[ each of height 2 -> continuous profile [0, 40[ of height 4
    // Tasks 40..49 have S in [0, 60], p = 3, h = 2 with capacity C = 5 -> maxAllowed = 3 < 4
    // So tasks 40..49 must all have their est pushed from 0 to 40!
    constexpr uint32_t n = 50;
    Allocator allocator;
    IntIntervalPtr s(allocator, 0, 60, n);
    vector<int32_t> durations(n);
    vector<int32_t> heights(n);
    for (uint32_t k = 0; k < 20; k++) {
        uint32_t t1 = 2 * k;
        uint32_t t2 = 2 * k + 1;
        s[t1][allocator].fix(2 * k);
        s[t2][allocator].fix(2 * k);
        durations[t1] = 2;
        durations[t2] = 2;
        heights[t1] = 2;
        heights[t2] = 2;
    }
    for (uint32_t i = 40; i < n; i++) {
        durations[i] = 3;
        heights[i] = 2;
    }
    Ptr<int32_t> d(allocator, durations);
    Ptr<int32_t> c(allocator, heights);
    Cumulative constraint(allocator, s, d, c, n, 5);

    DomainChange dc = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(dc == DomainChange::changed);
    for (uint32_t i = 40; i < n; i++) {
        REQUIRE(s[i][allocator].min() == 40);
        REQUIRE(s[i][allocator].max() == 60);
    }
}