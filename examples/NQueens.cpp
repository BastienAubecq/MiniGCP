#include <iostream>
#include <vector>
#include "MiniGCP/Model.hpp"
#include "MiniGCP/constraints/NotEqual.hpp"
#include "MiniGCP/search/FirstFail.hpp"
#include "MiniGCP/solver/EPSSolver.hpp"

using namespace std;
using namespace cp;

int main() {
    int n = 12;
    Model<NotEqual> model;
    vector<IntVarIdx> variables;
    for (int i = 0; i < n; i++) {
        variables.push_back(model.variable(n));
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            model.constraint(variables[i] != variables[j]);
            model.constraint(variables[i] + j - i != variables[j]);
            model.constraint(variables[i] - j + i != variables[j]);
        }
    }
    auto variablesArray = model.array<Scope::global, IntVarIdx>(variables);
    FirstFail strategy(variablesArray);
    DeviceParams params(sycl::gpu_selector_v, 64, 1);
    auto solver = EPSSolver(model, ProblemType::satisfaction, strategy, noLimit, params, variablesArray);
    solver.solve();

    SearchStatistics stats = solver.getStatistics();
    span<int32_t const> solution = solver.getSolution();
    cout << "Solutions: " << stats.nSolutions << "\nNodes: " << stats.nNodes << "\nFailures: " << stats.nFailures << endl;
    for (int i = 0; i < n; i++) {
        cout << solution[i] << " ";
    }
    cout << endl;
}