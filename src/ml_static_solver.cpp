#include <goal_control.hpp>
#include <goal_discretization.hpp>
#include <goal_output.hpp>

#include "ml_mechanics.hpp"
#include "ml_static_solver.hpp"

namespace ml {

static ParameterList get_valid_params() {
  ParameterList p;
  p.set<std::string>("solver type", "");
  p.sublist("discretization");
  p.sublist("physics");
  p.sublist("output");
  return p;
}

static void validate_params(ParameterList const& p) {
  GOAL_ALWAYS_ASSERT(p.isSublist("discretization"));
  GOAL_ALWAYS_ASSERT(p.isSublist("physics"));
  GOAL_ALWAYS_ASSERT(p.isSublist("output"));
  p.validateParameters(get_valid_params(), 0);
}

StaticSolver::StaticSolver(ParameterList const& p)
    : params(p),
      disc(0),
      mech(0),
      info(0),
      out(0) {
  validate_params(params);
  auto dp = params.sublist("discretization");
  auto pp = params.sublist("physics");
  auto op = params.sublist("output");
  disc = goal::create_disc(dp);
  out = goal::create_output(op, disc);
  mech = ml::create_mech(pp, disc);
}

StaticSolver::~StaticSolver() {
  ml::destroy_mech(mech);
  goal::destroy_output(out);
  goal::destroy_disc(disc);
}

void StaticSolver::solve() {
  goal::print("solving");
}

} // end namespace ml
