#include <iostream>
#include <vector>
#include <string>
#include "MiniGCP.hpp"

using namespace std;
using namespace cp;


/**
 * @brief Get a solver for an NQueens problem instance
 * @param n Number of queens
 * @param params Device parameters
 * @return The solver
 */
static auto getSolver(uint32_t n, DeviceParams params) {
    Model<AllDifferentFWC> model(ProblemType::satisfaction);
    vector<IntVarIdx> variables;
    vector<IntVarIdx> diag1;
    vector<IntVarIdx> diag2;
    for (int i = 0; i < n; i++) {
        IntVarIdx q = model.variable(n);
        variables.push_back(q);
        diag1.push_back(q + i);
        diag2.push_back(q - i);
    }
    model.constraint(AllDifferentFWC(model, variables));
    model.constraint(AllDifferentFWC(model, diag1));
    model.constraint(AllDifferentFWC(model, diag2));
    model.output = model.array<Scope::global, IntVarIdx>(variables);

    FirstFail strategy(model.output);
    return DFSSolver(model, strategy, params);
}


#ifndef NO_MAIN
int main(int argc, char const* const* argv) {
    if (argc < 2 || argc > 3) {
        cerr << "Arguments: number of queens, maximum number of solutions (optional)" << endl;
        return 1;
    }

    uint32_t n = std::stoi(argv[1]);
    uint32_t limit = argc == 3 ? std::stoi(argv[2]) : UINT32_MAX;
    DeviceParams params(sycl::cpu_selector_v, 1, 1);
    auto solver = getSolver(n, params);
    solver.solve([=](SearchStatistics stats) { return stats.nSolutions >= limit; });

    cout << solver.getStatistics() << endl;
    for (int32_t s : solver.getSolution()) {
        cout << s << " ";
    }
    cout << endl;
}
#endif