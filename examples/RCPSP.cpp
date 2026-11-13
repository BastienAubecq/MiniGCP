#include <span>
#include <iostream>
#include <fstream>
#include <numeric>
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
    vector<vector<int32_t>> successors;

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
                input >> consumption(iAct, iRes);
            }

            uint32_t nSucc;
            input >> nSucc;
            successors.push_back(vector<int32_t>(nSucc));
            for (uint32_t iSucc = 0; iSucc < nSucc; iSucc++) {
                input >> successors[iAct][iSucc];
                --successors[iAct][iSucc];
            }
        }
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
            Sum<MulVec<BoolPtr, Ptr<int32_t>>, IntInterval>,
            LessOrEqual<IntInterval, Const<int32_t>>,
            Maximum<AddVec<IntIntervalPtr, Ptr<int32_t>>, IntInterval>
        > model;

        int32_t horizon = accumulate(durations.begin(), durations.end(), 0);
        IntIntervalPtr start(model, 0, horizon, nActivities);
        for (int32_t i = 0; i < nActivities; i++) {
            for (int32_t j : successors[i]) {
                model.constraint(start[i] + durations[i] <= start[j]);
            }
        }
        for (int32_t t = 0; t < horizon; t++) {
            BoolPtr overlap(model, nActivities);
            for (int32_t i = 0; i < nActivities; i++) {
                Bool first = isLessOrEqual(model, start[i], Const(t));
                Bool second = isLess(model, Const(t - durations[i]), start[i]);
                model.constraint(IsAnd(first, second, overlap[i]));
            }
            for (int32_t i = 0; i < nResources; i++) {
                span<int32_t const> consumptionsSpan(consumptions.data() + i * nActivities, nActivities);
                model.constraint(sum(model, overlap * Ptr(model, consumptionsSpan), nActivities) <= Const(capacities[i]));
            }
        }
        span<int32_t const> durationsSpan(durations.data(), nActivities);
        IntInterval objective = maximum(model, start + Ptr(model, durationsSpan), nActivities);

        FirstFail strategy(start, nActivities);
        return DFSSolver(model, Minimization(objective), start, nActivities, strategy, params);
    }


    int32_t& consumption(int32_t iAct, int32_t iRes) {
        return consumptions[iRes * nActivities + iAct];
    }

private:
    vector<int32_t> consumptions;
};


#ifndef NO_MAIN
int main(int argc, char const* const* argv) {
    if (argc != 2) {
        cerr << "Arguments: input path" << endl;
        return 1;
    }

    RCPSP rcpsp(argv[1]);
    DeviceParams<128> params(sycl::gpu_selector_v, 1);
    auto solver = rcpsp.binarySolver(params);
    solver.solve(noLimit);

    cout << solver.getStatistics() << endl;
    for (int32_t s : solver.getSolution()) {
        cout << s << " ";
    }
    cout << endl;
}
#endif