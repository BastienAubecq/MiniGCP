#include <catch2/catch_test_macros.hpp>
#include <vector>
#include "MiniGCP.hpp"
#include "utils.hpp"

using namespace std;
using namespace cp;

namespace {

    template<IntVar V, bool _parallel = false>
    class SpyConstraint {
    public:
        constexpr static bool parallel = _parallel;

        SpyConstraint(Allocator& allocator, V x, DomainEvent ev) :
            count(allocator, 0), x(x), ev(ev) {}

        void post(EventRegistrar& reg) const {
            x.registerEvent(reg, ev);
        }

        DomainChange propagate(Context ctx) const {
            if constexpr (parallel) {
                if (ctx.item.get_local_id(0) == 0) count[ctx] += 1;
                sycl::group_barrier(ctx.item.get_group());
            } else {
                count[ctx] += 1;
            }
            return DomainChange::none;
        }

        Ref<uint32_t, Scope::local> count;

    private:
        V x;
        DomainEvent ev;
    };

    uint8_t extractEvent(uint32_t const* varEvents, uint32_t varId) {
        return static_cast<uint8_t>((varEvents[varId >> 3] >> ((varId & 7u) << 2)) & 0xFu);
    }

}

TEST_CASE("View event registration forwarding", "[Events][host]") {
    Allocator allocator;
    IntBitset x(allocator, 10);
    IntInterval y(allocator, 0, 10);
    Const<int32_t> c(5);

    std::vector<EventSubscription> subs;
    EventRegistrar reg(allocator, subs, 0, 0);

    reg.onMinChange(x + 3);
    reg.onMinChange(-x);       // should swap to max on x
    reg.onMaxChange(y * -2);   // should swap to min on y
    reg.onMinChange(y * 2);    // stays min on y
    reg.onBoundChange(y * 0);  // no-op
    reg.onDomainChange(c);     // no-op

    REQUIRE(subs.size() == 4);
    // x + 3 -> min (slot 1) on x
    REQUIRE(subs[0].varId == x.id());
    REQUIRE(subs[0].eventSlot == 1);
    // -x -> max (slot 2) on x
    REQUIRE(subs[1].varId == x.id());
    REQUIRE(subs[1].eventSlot == 2);
    // y * -2 -> min (slot 1) on y
    REQUIRE(subs[2].varId == y.id());
    REQUIRE(subs[2].eventSlot == 1);
    // y * 2 -> min (slot 1) on y
    REQUIRE(subs[3].varId == y.id());
    REQUIRE(subs[3].eventSlot == 1);
}

TEST_CASE("Device event emission granularity", "[Events]") {
    Allocator allocator;
    IntBitset b(allocator, 10);
    IntInterval iv(allocator, 0, 10);
    Bool bl(allocator);

    uint32_t words = (allocator.getNumVars() + 7u) / 8u;
    uint32_t evOffset = allocator.allocate<uint32_t, Scope::local>(words);

    auto runOp = [&](auto fn) -> uint8_t {
        uint32_t* hostEv = allocator.getData<uint32_t, Scope::local>(evOffset);
        for (uint32_t i = 0; i < words; ++i) hostEv[i] = 0;

        auto data = allocator.getData();
        sycl::queue q(testParams.selector, { sycl::property::queue::in_order() });
        std::array<std::byte*, 3> deviceData{};
        for (size_t i = 0; i < data.size(); i++) {
            if (!data[i].empty()) {
                deviceData[i] = sycl::malloc_device<std::byte>(data[i].size(), q);
                q.memcpy(deviceData[i], data[i].data(), data[i].size()).wait();
            }
        }
        q.submit([=](sycl::handler& h) {
            sycl::stream error(8192, 1024, h);
            sycl::local_accessor<std::byte> temporary(16, h);
            h.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) {
                uint32_t* varEv = reinterpret_cast<uint32_t*>(deviceData[1] + evOffset);
                cp::Context ctx(item, error, deviceData, temporary.get_multi_ptr<sycl::access::decorated::no>().get(), varEv);
                fn(ctx);
            });
        }).wait();

        q.memcpy(const_cast<std::byte*>(data[1].data()), deviceData[1], data[1].size()).wait();
        q.memcpy(const_cast<std::byte*>(data[2].data()), deviceData[2], data[2].size()).wait();
        for (std::byte* p : deviceData) {
            if (p != nullptr) sycl::free(p, q);
        }
        return extractEvent(allocator.getData<uint32_t, Scope::local>(evOffset), b.id());
    };

    // Interior remove on IntBitset -> domain only
    uint8_t ev1 = runOp([=](Context ctx) { b[ctx].remove(5); });
    REQUIRE(ev1 == static_cast<uint8_t>(DomainEvent::domain));

    // Remove min on IntBitset -> domain | min
    uint8_t ev2 = runOp([=](Context ctx) { b[ctx].remove(0); });
    REQUIRE(ev2 == static_cast<uint8_t>(DomainEvent::domain | DomainEvent::min));

    // RemoveAbove on IntBitset -> domain | max
    uint8_t ev3 = runOp([=](Context ctx) { b[ctx].removeAbove(7); });
    REQUIRE(ev3 == static_cast<uint8_t>(DomainEvent::domain | DomainEvent::max));

    // Fix on IntBitset -> domain | fix | min | max
    uint8_t ev4 = runOp([=](Context ctx) { b[ctx].fix(3); });
    REQUIRE(ev4 == static_cast<uint8_t>(DomainEvent::domain | DomainEvent::fix | DomainEvent::min | DomainEvent::max));

    // IntInterval removeBelow -> domain | min, then removeAbove to fix -> domain | max | fix
    runOp([=](Context ctx) { iv[ctx].removeBelow(4); });
    REQUIRE(extractEvent(allocator.getData<uint32_t, Scope::local>(evOffset), iv.id()) ==
            static_cast<uint8_t>(DomainEvent::domain | DomainEvent::min));

    runOp([=](Context ctx) { iv[ctx].removeAbove(4); });
    REQUIRE(extractEvent(allocator.getData<uint32_t, Scope::local>(evOffset), iv.id()) ==
            static_cast<uint8_t>(DomainEvent::domain | DomainEvent::max | DomainEvent::fix));

    // Bool fix(1) -> domain | fix | min
    runOp([=](Context ctx) { bl[ctx].fix(1); });
    REQUIRE(extractEvent(allocator.getData<uint32_t, Scope::local>(evOffset), bl.id()) ==
            static_cast<uint8_t>(DomainEvent::domain | DomainEvent::fix | DomainEvent::min));
}

TEST_CASE("Selective constraint wakeup in ConstraintStore", "[Events][ConstraintStore]") {
    Model<SpyConstraint<IntBitset, false>, SpyConstraint<IntBitset, true>> model;
    IntBitset x(model, 10);
    IntBitset y(model, 10);

    SpyConstraint<IntBitset, false> onDomX(model, x, DomainEvent::domain);
    SpyConstraint<IntBitset, false> onMinX(model, x, DomainEvent::min);
    SpyConstraint<IntBitset, false> onMaxX(model, x, DomainEvent::max);
    SpyConstraint<IntBitset, false> onFixX(model, x, DomainEvent::fix);
    SpyConstraint<IntBitset, false> onDomY(model, y, DomainEvent::domain);
    SpyConstraint<IntBitset, true>  parOnMinX(model, x, DomainEvent::min);
    SpyConstraint<IntBitset, true>  parOnFixY(model, y, DomainEvent::fix);

    model.constraint(onDomX);
    model.constraint(onMinX);
    model.constraint(onMaxX);
    model.constraint(onFixX);
    model.constraint(onDomY);
    model.constraint(parOnMinX);
    model.constraint(parOnFixY);

    auto layout = model.prepareEventStore();
    sycl::queue q(testParams.selector, { sycl::property::queue::in_order() });

    ConstraintStore<SpyConstraint<IntBitset, false>, SpyConstraint<IntBitset, true>> store(
        q, model.getConstraints(), layout, model.getNumVars(), model.getSubscriptions()
    );

    auto data = model.getData();
    std::array<std::byte*, 3> deviceData{};
    for (size_t i = 0; i < data.size(); i++) {
        if (!data[i].empty()) {
            deviceData[i] = sycl::malloc_device<std::byte>(data[i].size(), q);
            q.memcpy(deviceData[i], data[i].data(), data[i].size()).wait();
        }
    }
    uint32_t tmpSize = model.getTemporarySize();

    auto runStep = [&](auto mutator) {
        q.submit([=](sycl::handler& h) {
            sycl::stream error(8192, 1024, h);
            sycl::local_accessor<std::byte> temporary(tmpSize + 16, h);
            h.parallel_for(sycl::nd_range<1>(64, 64), [=](sycl::nd_item<1> item) {
                uint32_t* varEv = reinterpret_cast<uint32_t*>(deviceData[1] + layout.varEventsOffset);
                cp::Context ctx(item, error, deviceData, temporary.get_multi_ptr<sycl::access::decorated::no>().get() + 16, varEv);
                if (item.get_local_id(0) == 0) mutator(ctx);
                sycl::group_barrier(item.get_group());
                store.fixpoint(ctx);
            });
        }).wait();
        q.memcpy(const_cast<std::byte*>(data[1].data()), deviceData[1], data[1].size()).wait();
    };

    // 1. Initial fixpoint: all constraints run once
    runStep([](Context) {});
    REQUIRE(onDomX.count[model] == 1);
    REQUIRE(onMinX.count[model] == 1);
    REQUIRE(onMaxX.count[model] == 1);
    REQUIRE(onFixX.count[model] == 1);
    REQUIRE(onDomY.count[model] == 1);
    REQUIRE(parOnMinX.count[model] == 1);
    REQUIRE(parOnFixY.count[model] == 1);

    // 2. Interior remove on x -> only onDomX wakes up
    runStep([=](Context ctx) { x[ctx].remove(5); });
    REQUIRE(onDomX.count[model] == 2);
    REQUIRE(onMinX.count[model] == 1);
    REQUIRE(onMaxX.count[model] == 1);
    REQUIRE(onFixX.count[model] == 1);
    REQUIRE(onDomY.count[model] == 1);
    REQUIRE(parOnMinX.count[model] == 1);
    REQUIRE(parOnFixY.count[model] == 1);

    // 3. removeBelow(2) on x -> onDomX, onMinX, parOnMinX wake up; onMaxX, onFixX, y-constraints stay asleep
    runStep([=](Context ctx) { x[ctx].removeBelow(2); });
    REQUIRE(onDomX.count[model] == 3);
    REQUIRE(onMinX.count[model] == 2);
    REQUIRE(onMaxX.count[model] == 1);
    REQUIRE(onFixX.count[model] == 1);
    REQUIRE(onDomY.count[model] == 1);
    REQUIRE(parOnMinX.count[model] == 2);
    REQUIRE(parOnFixY.count[model] == 1);

    // 4. fix(4) on x -> onDomX, onMinX, onMaxX, onFixX, parOnMinX wake up; y-constraints stay asleep
    runStep([=](Context ctx) { x[ctx].fix(4); });
    REQUIRE(onDomX.count[model] == 4);
    REQUIRE(onMinX.count[model] == 3);
    REQUIRE(onMaxX.count[model] == 2);
    REQUIRE(onFixX.count[model] == 2);
    REQUIRE(onDomY.count[model] == 1);
    REQUIRE(parOnMinX.count[model] == 3);
    REQUIRE(parOnFixY.count[model] == 1);

    // 5. fix(7) on y -> onDomY and parOnFixY wake up; x-constraints stay asleep
    runStep([=](Context ctx) { y[ctx].fix(7); });
    REQUIRE(onDomX.count[model] == 4);
    REQUIRE(onMinX.count[model] == 3);
    REQUIRE(onMaxX.count[model] == 2);
    REQUIRE(onFixX.count[model] == 2);
    REQUIRE(onDomY.count[model] == 2);
    REQUIRE(parOnMinX.count[model] == 3);
    REQUIRE(parOnFixY.count[model] == 2);

    store.destroy(q);
    for (std::byte* p : deviceData) {
        if (p != nullptr) sycl::free(p, q);
    }
}
