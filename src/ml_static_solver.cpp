#include <goal_control.hpp>
#include <goal_discretization.hpp>
#include "ml_static_solver.hpp"

namespace ml {

static ParameterList get_valid_params() {
  ParameterList p;
  p.sublist("discretization");
  return p;
}

static void validate_params(ParameterList const& p) {
  GOAL_ALWAYS_ASSERT(p.isSublist("discretization"));
  p.validateParameters(get_valid_params(), 0);
}

StaticSolver::StaticSolver(ParameterList const& p)
    : params(p),
      disc(0),
      physics(0),
      info(0),
      output(0) {
  validate_params(params);
  auto dp = params.sublist("discretization");
  disc = goal::create_disc(dp);
}

StaticSolver::~StaticSolver() {
}

void StaticSolver::solve() {
  goal::print("solving");
}

} // end namespace ml