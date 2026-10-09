#include <iostream>
#include <vector>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/constraints/AllDifferentFWC.hpp"
#include "MiniGCP/search/FirstFail.hpp"
#include "MiniGCP/solver/DFSSolver.hpp"

using namespace std;
using namespace cp;

int main() {
    int n = 30;

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
    DeviceParams params(sycl::gpu_selector_v, 64, 1);
    auto solver = DFSSolver(model, strategy, params);
    solver.solve([](SearchStatistics stats) { return stats.nSolutions >= 10000; });

    cout << solver.getStatistics() << endl;
    span<int32_t const> solution = solver.getSolution();
    for (int i = 0; i < n; i++) {
        cout << solution[i] << " ";
    }
    cout << endl;
}