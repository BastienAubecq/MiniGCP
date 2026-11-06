#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/constraints/Circuit.hpp"
#include "MiniGCP/state/IntBitset.hpp"
#include "utils.hpp"

using namespace cp;
using namespace std;


TEST_CASE("CircuitSeq one cycle", "[CircuitSeq]") {
    Allocator allocator;
    IntBitsetArray x(allocator, 10, 10);
    x[allocator, 3][allocator].fix(5);
    x[allocator, 5][allocator].fix(7);
    x[allocator, 7][allocator].fix(2);
    x[allocator, 2][allocator].fix(8);
    CircuitSeq constraint(allocator, x);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(x[allocator, 8][allocator].contains(3));
    REQUIRE(x[allocator, 0][allocator].size() == 9);
    REQUIRE(x[allocator, 1][allocator].size() == 9);
    REQUIRE(x[allocator, 4][allocator].size() == 9);
    REQUIRE(x[allocator, 6][allocator].size() == 9);
    REQUIRE(x[allocator, 9][allocator].size() == 9);
}

TEST_CASE("CircuitSeq few cycles", "[CircuitSeq]") {
    Allocator allocator;
    IntBitsetArray x(allocator, 10, 10);
    x[allocator, 3][allocator].fix(5);
    x[allocator, 0][allocator].fix(1);
    x[allocator, 1][allocator].fix(9);
    x[allocator, 8][allocator].fix(2);
    x[allocator, 2][allocator].fix(4);
    x[allocator, 4][allocator].fix(7);
    CircuitSeq constraint(allocator, x);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(x[allocator, 5][allocator].contains(3));
    REQUIRE_FALSE(x[allocator, 9][allocator].contains(0));
    REQUIRE_FALSE(x[allocator, 7][allocator].contains(8));
    REQUIRE(x[allocator, 6][allocator].size() == 9);
}

TEST_CASE("CircuitSeq full cycle", "[CircuitSeq]") {
    Allocator allocator;
    IntBitsetArray x(allocator, 10, 10);
    for (int i = 0; i < 9; i++) {
        x[allocator, i][allocator].fix(i + 1);
    }
    CircuitSeq constraint(allocator, x);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}


TEST_CASE("Circuit one cycle", "[Circuit]") {
    Allocator allocator;
    IntBitsetArray x(allocator, 10, 10);
    x[allocator, 3][allocator].fix(5);
    x[allocator, 5][allocator].fix(7);
    x[allocator, 7][allocator].fix(2);
    x[allocator, 2][allocator].fix(8);
    Circuit constraint(allocator, x);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(x[allocator, 8][allocator].contains(3));
    REQUIRE(x[allocator, 0][allocator].size() == 9);
    REQUIRE(x[allocator, 1][allocator].size() == 9);
    REQUIRE(x[allocator, 4][allocator].size() == 9);
    REQUIRE(x[allocator, 6][allocator].size() == 9);
    REQUIRE(x[allocator, 9][allocator].size() == 9);
}

TEST_CASE("Circuit few cycles", "[Circuit]") {
    Allocator allocator;
    IntBitsetArray x(allocator, 10, 10);
    x[allocator, 3][allocator].fix(5);
    x[allocator, 0][allocator].fix(1);
    x[allocator, 1][allocator].fix(9);
    x[allocator, 8][allocator].fix(2);
    x[allocator, 2][allocator].fix(4);
    x[allocator, 4][allocator].fix(7);
    Circuit constraint(allocator, x);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    REQUIRE_FALSE(x[allocator, 5][allocator].contains(3));
    REQUIRE_FALSE(x[allocator, 9][allocator].contains(0));
    REQUIRE_FALSE(x[allocator, 7][allocator].contains(8));
    REQUIRE(x[allocator, 6][allocator].size() == 9);
}

TEST_CASE("Circuit full cycle", "[Circuit]") {
    Allocator allocator;
    IntBitsetArray x(allocator, 10, 10);
    for (int i = 0; i < 9; i++) {
        x[allocator, i][allocator].fix(i + 1);
    }
    Circuit constraint(allocator, x);

    DomainChange c = runOnDevice<DomainChange>(allocator, 1, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::none);
}


TEST_CASE("Circuit concurrent few variables", "[Circuit][concurrent]") {
    Allocator allocator;
    IntBitsetArray x(allocator, 10, 10);
    for (int i = 1; i < 8; i += 2) {
        x[allocator, i][allocator].fix(i + 2);
    }
    Circuit constraint(allocator, x);

    DomainChange c = runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 9; i++) {
        if (i % 2 == 0) REQUIRE(x[allocator, i][allocator].size() == 9);
    }
    REQUIRE_FALSE(x[allocator, 9][allocator].contains(1));
}

TEST_CASE("Circuit concurrent many variables", "[Circuit][concurrent]") {
    Allocator allocator;
    IntBitsetArray x(allocator, 100, 100);
    for (int i = 1; i < 98; i += 2) {
        x[allocator, i][allocator].fix(i + 2);
    }
    Circuit constraint(allocator, x);

    DomainChange c = runOnDevice<DomainChange>(allocator, 16, [=](Context ctx) {
        return constraint.propagate(ctx);
    })[0];

    REQUIRE(c == DomainChange::changed);
    for (int i = 0; i < 99; i++) {
        if (i % 2 == 0) REQUIRE(x[allocator, i][allocator].size() == 99);
    }
    REQUIRE_FALSE(x[allocator, 99][allocator].contains(1));
}