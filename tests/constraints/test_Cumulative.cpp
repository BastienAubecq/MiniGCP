#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/constraints/Cumulative.hpp"
#include "MiniGCP/constraints/Precedence.hpp"
#include "MiniGCP/state/IntInterval.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;

// Helper to initialize global arrays for durations and heights
static pair<Ptr<int32_t, Scope::global>, Ptr<int32_t, Scope::global>>
makeTaskParams(Allocator& allocator, vector<int32_t> const& durations, vector<int32_t> const& heights) {
    uint32_t n = durations.size();
    Ptr<int32_t, Scope::global> p(allocator, n);
    Ptr<int32_t, Scope::global> h(allocator, n);
    for (uint32_t i = 0; i < n; i++) {
        p[allocator, i] = durations[i];
        h[allocator, i] = heights[i];
    }
    return {p, h};
}

TEST_CASE("CumulativeSeq Gay et al. CP2015 Fig 1 example", "[CumulativeSeq]") {
    // Capacity C = 3
    // Task A (0): S in [0, 0], p = 2, h = 3 -> mandatory [0, 2[ of height 3
    // Task B (1): S in [4, 4], p = 2, h = 2 -> mandatory [4, 6[ of height 2
    // Task C (2): S in [0, 8], p = 5, h = 2 -> no mandatory part; pushed from 0 -> 2 -> 6
    Allocator allocator;
    IntIntervalPtr s(allocator, 0, 10, 3);
    s[0][allocator].fix(0);
    s[1][allocator].fix(4);
    s[2][allocator].removeAbove(8);

    auto [p, h] = makeTaskParams(allocator, {2, 2, 5}, {3, 2, 2});
    CumulativeSeq constraint(allocator, s, p, h, 3, 3);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE(s[2][allocator].min() == 6);
    REQUIRE(s[2][allocator].max() == 8);
}

TEST_CASE("Cumulative Gay et al. CP2015 Fig 1 example (parallel)", "[Cumulative][concurrent]") {
    Allocator allocator;
    IntIntervalPtr s(allocator, 0, 10, 3);
    s[0][allocator].fix(0);
    s[1][allocator].fix(4);
    s[2][allocator].removeAbove(8);

    auto [p, h] = makeTaskParams(allocator, {2, 2, 5}, {3, 2, 2});
    Cumulative constraint(allocator, s, p, h, 3, 3);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
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

    auto [p, h] = makeTaskParams(allocator, {2, 2, 5}, {3, 2, 2});
    Cumulative constraint(allocator, s, p, h, 3, 3);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
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

    auto [p, h] = makeTaskParams(allocator, {4, 3}, {3, 2});
    Cumulative constraint(allocator, s, p, h, 2, 4);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
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

    auto [p, h] = makeTaskParams(allocator, {4, 4, 4}, {2, 2, 2});
    Cumulative constraint(allocator, s, p, h, 3, 4);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("Cumulative task wipeout inconsistency", "[Cumulative][concurrent]") {
    // Capacity C = 3
    // Task 0: fixed at 2, p = 4, h = 2 -> mandatory [2, 6[
    // Task 1: S in [0, 3], p = 4, h = 2 -> cannot fit before 2 (needs 4 units) nor after 2 (lst=3 < 6)
    Allocator allocator;
    IntIntervalPtr s(allocator, 0, 10, 2);
    s[0][allocator].fix(2);
    s[1][allocator].removeAbove(3);

    auto [p, h] = makeTaskParams(allocator, {4, 4}, {2, 2});
    Cumulative constraint(allocator, s, p, h, 2, 3);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE((c & DomainChange::inconsistency) == DomainChange::inconsistency);
}

TEST_CASE("Cumulative no mandatory parts", "[Cumulative][concurrent]") {
    Allocator allocator;
    IntIntervalPtr s(allocator, 0, 20, 5);
    auto [p, h] = makeTaskParams(allocator, {3, 4, 2, 5, 3}, {2, 1, 2, 3, 1});
    Cumulative constraint(allocator, s, p, h, 5, 4);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::none);
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

    auto [p, h] = makeTaskParams(allocator, durations, heights);
    Cumulative constraint(allocator, s, p, h, n, 5);

    DomainChange c = combineChanges(runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    }));

    REQUIRE(c == DomainChange::changed);
    for (uint32_t i = 40; i < n; i++) {
        REQUIRE(s[i][allocator].min() == 40);
        REQUIRE(s[i][allocator].max() == 60);
    }
}

TEST_CASE("Precedence and PrecedenceVec", "[Precedence][concurrent]") {
    Allocator allocator;
    IntIntervalPtr s(allocator, 0, 20, 4);
    s[0][allocator].removeBelow(3);
    s[3][allocator].removeAbove(15);

    // Chain: s[0] + 4 <= s[1], s[1] + 3 <= s[2], s[2] + 2 <= s[3]
    Ptr<uint32_t, Scope::global> pred(allocator, 3);
    Ptr<uint32_t, Scope::global> succ(allocator, 3);
    Ptr<int32_t, Scope::global> d(allocator, 3);
    pred[allocator, 0] = 0; succ[allocator, 0] = 1; d[allocator, 0] = 4;
    pred[allocator, 1] = 1; succ[allocator, 1] = 2; d[allocator, 1] = 3;
    pred[allocator, 2] = 2; succ[allocator, 2] = 3; d[allocator, 2] = 2;

    PrecedenceVec constraint(allocator, s, pred, succ, d, 3);
    // Run 3 passes on device to propagate across the 3-edge chain
    for (int pass = 0; pass < 3; pass++) {
        runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
            return constraint.propagate(ctx);
        });
    }

    REQUIRE(s[1][allocator].min() == 7);
    REQUIRE(s[2][allocator].min() == 10);
    REQUIRE(s[3][allocator].min() == 12);
    REQUIRE(s[2][allocator].max() == 13);
    REQUIRE(s[1][allocator].max() == 10);
    REQUIRE(s[0][allocator].max() == 6);
}
