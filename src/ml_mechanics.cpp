#include <goal_control.hpp>
#include <goal_discretization.hpp>
#include <goal_field.hpp>
#include <goal_states.hpp>
#include "ml_mechanics.hpp"

namespace ml {

Mechanics::Mechanics(ParameterList const& p, goal::Discretization* d)
    : goal::Physics(d),
      params(p) {
}

ParameterList const& Mechanics::get_dbc_params() {
  return params;
}

void Mechanics::build_primal_volumetric(FieldManager fm) {
  (void)fm;
}

void Mechanics::build_primal_neumann(FieldManager fm) {
  (void)fm;
}

void Mechanics::build_dual_volumetric(FieldManager fm) {
  (void)fm;
}

void Mechanics::build_dual_neumann(FieldManager fm) {
  (void)fm;
}

void Mechanics::build_error_volumetric(FieldManager fm) {
  (void)fm;
}

void Mechanics::build_error_neumann(FieldManager fm) {
  (void)fm;
}

Mechanics* create_mech(ParameterList const& p, goal::Discretization* d) {
  return new Mechanics(p, d);
}

void destroy_mech(Mechanics* p) {
  delete p;
}

} // end namespace ml
