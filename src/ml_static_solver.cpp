#include <goal_control.hpp>
#include <goal_discretization.hpp>

#include "ml_mechanics.hpp"
#include "ml_static_solver.hpp"

namespace ml {

static ParameterList get_valid_params() {
  ParameterList p;
  p.set<std::string>("solver type", "");
  p.sublist("discretization");
  p.sublist("physics");
  return p;
}

static void validate_params(ParameterList const& p) {
  GOAL_ALWAYS_ASSERT(p.isSublist("discretization"));
  GOAL_ALWAYS_ASSERT(p.isSublist("physics"));
  p.validateParameters(get_valid_params(), 0);
}

StaticSolver::StaticSolver(ParameterList const& p)
    : params(p),
      disc(0),
      mech(0),
      info(0),
      output(0) {
  validate_params(params);
  auto dp = params.sublist("discretization");
  auto pp = params.sublist("physics");
  disc = goal::create_disc(dp);
  mech = ml::create_mech(pp, disc);
}

StaticSolver::~StaticSolver() {
}

void StaticSolver::solve() {
  goal::print("solving");
}

} // end namespace ml
