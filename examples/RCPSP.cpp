#include <span>
#include <iostream>
#include <fstream>
#include <numeric>
#include <random>
#include <vector>
#include "MiniGCP.hpp"

using namespace std;
using namespace cp;


/**
 * @brief Resource-Constrained Project Scheduling Problem
 */
class RCPSP {
public:
    uint32_t nActivities;
    uint32_t nResources;
    vector<int32_t> capacities;
    vector<int32_t> durations;
    vector<int32_t> consumptions;
    vector<uint32_t> pred;
    vector<uint32_t> succ;


    /**
     * @brief Read an RCPSP instance from a file
     * @param path Path to the file
     */
    explicit RCPSP(char const* path) {
        ifstream input(path);
        input.exceptions(std::ios::failbit);

        input >> nActivities;
        input >> nResources;
        capacities.resize(nResources);
        durations.resize(nActivities);
        consumptions.resize(nActivities * nResources);

        for (uint32_t iRes = 0; iRes < nResources; iRes++) {
            input >> capacities[iRes];
        }
        for (uint32_t iAct = 0; iAct < nActivities; iAct++) {
            input >> durations[iAct];
            for (uint32_t iRes = 0; iRes < nResources; iRes++) {
                input >> consumptions[iRes * nActivities + iAct];
            }

            uint32_t nSucc;
            input >> nSucc;
            for (uint32_t iSucc = 0; iSucc < nSucc; iSucc++) {
                uint32_t s;
                input >> s;
                pred.push_back(iAct);
                succ.push_back(s - 1);
            }
        }
    }


    /**
     * @brief Generate a random reproducible RCPSP instance
     * @param nActivities Number of non-dummy activities
     * @param nResources Number of renewable resources
     * @param seed Random seed
     */
    RCPSP(uint32_t nActivities, uint32_t nResources, uint32_t seed = 0) :
        nActivities(nActivities + 2),
        nResources(nResources),
        capacities(nResources, 6),
        durations(nActivities + 2, 0),
        consumptions((nActivities + 2) * nResources) {
        mt19937 gen(seed);
        uniform_int_distribution<int32_t> durDist(2, 6);
        uniform_int_distribution<int32_t> reqDist(1, 4);

        for (uint32_t iAct = 1; iAct <= nActivities; iAct++) {
            durations[iAct] = durDist(gen);
            for (uint32_t iRes = 0; iRes < nResources; iRes++) {
                consumptions[iRes * nActivities + iAct] = reqDist(gen);
            }
        }

        for (uint32_t i = 1; i <= nActivities; i++) {
            // Source precedes all activities; all activities precede sink
            pred.push_back(0);
            succ.push_back(i);
            pred.push_back(i);
            succ.push_back(nActivities + 1);

            // Random precedences
            for (uint32_t j = i + 1; j <= nActivities; j++) {
                if (j - i <= 3 && (gen() % 3 == 0)) {
                    pred.push_back(i);
                    succ.push_back(j);
                }
            }
        }
    }


    /**
     * @brief Get a solver for the RCPSP instance
     * @param params Device parameters
     * @return The solver 
     */
    auto solver(IsDeviceParams auto params) const {
        Model<
            LessOrEqual<Add<IntInterval, int32_t>, IntInterval>,
            Cumulative<IntIntervalPtr, Ptr<int32_t>, Ptr<int32_t>>,
            Maximum<AddVec<IntIntervalPtr, Ptr<int32_t>>, IntInterval>
        > model;

        int32_t horizon = accumulate(durations.begin(), durations.end(), 0);
        IntIntervalPtr start(model, 0, horizon, nActivities);
        for (uint32_t i = 0; i < pred.size(); i++) {
            model.constraint(start[pred[i]] + durations[pred[i]] <= start[succ[i]]);
        }
        Ptr<int32_t> durationsPtr(model, span(durations.data(), nActivities));
        for (uint32_t r = 0; r < nResources; r++) {
            Ptr<int32_t> consumptionsPtr(model, span(consumptions.data() + r * nActivities, nActivities));
            model.constraint(Cumulative(model, start, durationsPtr, consumptionsPtr, nActivities, capacities[r]));
        }
        IntInterval objective(model);
        model.constraint(Maximum(model, start + durationsPtr, nActivities, objective));

        MinFirst strategy(start, nActivities);
        return DFSSolver(model, Minimization(objective), start, nActivities, strategy, params);
    }


    /**
     * @brief Get a solver for the RCPSP instance.
     * The problem is decomposed into binary constraints.
     * @param params Device parameters
     * @return The solver 
     */
    auto binarySolver(IsDeviceParams auto params) const {
        Model<
            LessOrEqual<Add<IntInterval, int32_t>, IntInterval>,
            IsLessOrEqual<IntInterval, Const<int32_t>, Bool>,
            IsLess<Const<int32_t>, IntInterval, Bool>,
            IsAnd<Bool, Bool, Bool>,
            SumSeq<MulVec<BoolPtr, Ptr<int32_t>>, IntInterval>,
            LessOrEqual<IntInterval, Const<int32_t>>,
            Maximum<AddVec<IntIntervalPtr, Ptr<int32_t>>, IntInterval>
        > model;

        int32_t horizon = accumulate(durations.begin(), durations.end(), 0);
        Ptr<int32_t> durationsPtr(model, span(durations.data(), nActivities));
        vector<Ptr<int32_t>> consumptionsPtrs;
        for (int32_t i = 0; i < nResources; i++) {
            consumptionsPtrs.push_back(Ptr(model, span(consumptions.data() + i * nActivities, nActivities)));
        }

        IntIntervalPtr start(model, 0, horizon, nActivities);
        for (uint32_t i = 0; i < pred.size(); i++) {
            model.constraint(start[pred[i]] + durations[pred[i]] <= start[succ[i]]);
        }
        for (int32_t t = 0; t < horizon; t++) {
            BoolPtr overlap(model, nActivities);
            for (int32_t i = 0; i < nActivities; i++) {
                Bool first = isLessOrEqual(model, start[i], Const(t));
                Bool second = isLess(model, Const(t - durations[i]), start[i]);
                model.constraint(IsAnd(first, second, overlap[i]));
            }
            for (int32_t r = 0; r < nResources; r++) {
                IntInterval usage(model);
                model.constraint(SumSeq(model, overlap * consumptionsPtrs[r], nActivities, usage));
                model.constraint(usage <= Const(capacities[r]));
            }
        }
        IntInterval objective(model);
        model.constraint(Maximum(model, start + durationsPtr, nActivities, objective));

        MinFirst strategy(start, nActivities);
        return DFSSolver(model, Minimization(objective), start, nActivities, strategy, params);
    }


    int32_t consumption(int32_t iAct, int32_t iRes) const {
        return consumptions[iRes * nActivities + iAct];
    }
};


#ifndef NO_MAIN
int main(int argc, char const* const* argv) {
    if (argc != 2) {
        cerr << "Arguments: input path" << endl;
        return 1;
    }

    RCPSP rcpsp(argv[1]);
    DeviceParams<128> params(sycl::gpu_selector_v, 1);
    auto solver = rcpsp.solver(params);
    solver.solve(noLimit);

    cout << solver.getStatistics() << endl;
    for (int32_t s : solver.getSolution()) {
        cout << s << " ";
    }
    cout << endl;
}
#endif