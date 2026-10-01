#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/state/IntVarRef.hpp"
#include "MiniGCP/state/DomainChange.hpp"

using namespace std;
using namespace cp;


TEST_CASE("addVariable(size) simple", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
    REQUIRE_FALSE(v.fixed());
    for (int i = 0; i < 10; i++) REQUIRE(v.contains(i));
    REQUIRE_FALSE(v.contains(-1));
    REQUIRE_FALSE(v.contains(10));
}

TEST_CASE("addVariable(size) multiple words", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(70)];

    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 69);
    REQUIRE(v.size() == 70);
    for (int i = 0; i <= 69; i++) REQUIRE(v.contains(i));
    REQUIRE_FALSE(v.contains(-1));
    REQUIRE_FALSE(v.contains(70));
}

TEST_CASE("addVariable(min,max)", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(5, 10)];

    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 10);
    REQUIRE(v.size() == 6);
    for (int i = 5; i <= 10; i++) REQUIRE(v.contains(i));
    REQUIRE_FALSE(v.contains(4));
    REQUIRE_FALSE(v.contains(11));
}

TEST_CASE("addVariable(values)", "[IntVarRef][host]") {
    vector<int32_t> values{ -5, -4, -1, 2, 3, 4, 10, 40 };
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(values)];

    REQUIRE(v.min() == -5);
    REQUIRE(v.max() == 40);
    REQUIRE(v.size() == values.size());
    for (int32_t val : values) REQUIRE(v.contains(val));
    REQUIRE_FALSE(v.contains(0));
    REQUIRE_FALSE(v.contains(1));
    REQUIRE_FALSE(v.contains(-6));
    REQUIRE_FALSE(v.contains(41));
}


TEST_CASE("remove simple", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.remove(5) == DomainChange::changed);
    REQUIRE(v.size() == 9);
    REQUIRE_FALSE(v.contains(5));
    for (int32_t i = 0; i < 5; i++) REQUIRE(v.contains(i));
    for (int32_t i = 6; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("remove outside bounds", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.remove(-1) == DomainChange::none);
    REQUIRE(v.remove(10) == DomainChange::none);
    REQUIRE(v.size() == 10);
}

TEST_CASE("remove already removed", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];
    v.remove(5);

    REQUIRE(v.remove(5) == DomainChange::none);
    REQUIRE(v.size() == 9);
}

TEST_CASE("remove min", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.remove(0) == DomainChange::changed);
    REQUIRE(v.min() == 1);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 9);
}

TEST_CASE("remove max", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.remove(9) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 8);
    REQUIRE(v.size() == 9);
}

TEST_CASE("remove last", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(1)];

    REQUIRE(v.remove(0) == DomainChange::inconsistency);
}


TEST_CASE("fix simple", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.fix(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 1);
    REQUIRE(v.fixed());
    REQUIRE(v.contains(5));
    for (int32_t i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int32_t i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("fix already fixed", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(1)];

    REQUIRE(v.fix(0) == DomainChange::none);
    REQUIRE(v.fixed());
}

TEST_CASE("fix outside bounds", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.fix(15) == DomainChange::inconsistency);
}

TEST_CASE("fix removed", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];
    v.remove(5);

    REQUIRE(v.fix(5) == DomainChange::inconsistency);
}


TEST_CASE("removeBelow simple", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.removeBelow(5) == DomainChange::changed);
    REQUIRE(v.min() == 5);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 5);
    for (int i = 0; i < 5; i++) REQUIRE_FALSE(v.contains(i));
    for (int i = 5; i < 10; i++) REQUIRE(v.contains(i));
}

TEST_CASE("removeBelow below min", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.removeBelow(0) == DomainChange::none);
    REQUIRE(v.removeBelow(-5) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("removeBelow above max", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.removeBelow(15) == DomainChange::inconsistency);
}


TEST_CASE("removeAbove simple", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.removeAbove(5) == DomainChange::changed);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 5);
    REQUIRE(v.size() == 6);
    for (int i = 0; i <= 5; i++) REQUIRE(v.contains(i));
    for (int i = 6; i < 10; i++) REQUIRE_FALSE(v.contains(i));
}

TEST_CASE("removeAbove above max", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.removeAbove(9) == DomainChange::none);
    REQUIRE(v.removeAbove(15) == DomainChange::none);
    REQUIRE(v.min() == 0);
    REQUIRE(v.max() == 9);
    REQUIRE(v.size() == 10);
}

TEST_CASE("removeAbove below min", "[IntVarRef][host]") {
    Allocator allocator;
    IntVarRef v = allocator[allocator.variable(10)];

    REQUIRE(v.removeAbove(-5) == DomainChange::inconsistency);
}