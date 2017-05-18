#include <goal_control.hpp>
#include <goal_discretization.hpp>
#include <goal_field.hpp>
#include <goal_states.hpp>
#include "ml_mechanics.hpp"

namespace ml {

static ParameterList get_valid_params(goal::Discretization* d) {
  ParameterList p;
  p.sublist("dirichlet bcs");
  p.sublist("neumann bcs");
  for (int i = 0; i < d->get_num_elem_sets(); ++i)
    p.sublist(d->get_elem_set_name(i));
  return p;
}

static void validate_params(
    ParameterList const& p, goal::Discretization* d) {
  GOAL_ALWAYS_ASSERT_VERBOSE(
      p.isSublist("dirichlet bcs"),
      "mechanics: dirichlet bcs have not been defined");
  for (int i = 0; i < d->get_num_elem_sets(); ++i)
    GOAL_ALWAYS_ASSERT_VERBOSE(
        p.isSublist(d->get_elem_set_name(i)),
        "mechanics: missing material definition");
  p.validateParameters(get_valid_params(d));
}

Mechanics::Mechanics(ParameterList const& p, goal::Discretization* d)
    : goal::Physics(d),
      params(p),
      states(0),
      is_primal(false),
      is_dual(false),
      is_error(false)
{
  validate_params(p, d);
}

ParameterList const& Mechanics::get_dbc_params() {
  return params.sublist("dirichlet bcs");
}

void Mechanics::set_primal() {
  is_primal = true;
  is_dual = false;
  is_error = false;
}

void Mechanics::set_dual() {
  is_primal = false;
  is_dual = true;
  is_error = false;
}

void Mechanics::set_error() {
  is_primal = false;
  is_dual = false;
  is_error = true;
}

template <typename T>
static void write_graph(goal::FieldManager fm, const char* n) {
  fm->writeGraphvizFile<T>(n, true, true);
}

void Mechanics::build_primal_volumetric(FieldManager fm) {
  set_primal();
  register_volumetric<Residual>(fm);
  register_volumetric<Jacobian>(fm);
  write_graph<Jacobian>(fm, "p_volumetric.dot");
}

void Mechanics::build_primal_neumann(FieldManager fm) {
  set_primal();
  register_neumann<Residual>(fm);
  register_neumann<Jacobian>(fm);
  write_graph<Jacobian>(fm, "p_neumann.dot");
}

void Mechanics::build_dual_volumetric(FieldManager fm) {
  set_dual();
  register_volumetric<Jacobian>(fm);
  write_graph<Jacobian>(fm, "d_volumetric.dot");
}

void Mechanics::build_dual_neumann(FieldManager fm) {
  set_dual();
  register_neumann<Jacobian>(fm);
  write_graph<Jacobian>(fm, "d_neumann.dot");
}

void Mechanics::build_error_volumetric(FieldManager fm) {
  set_error();
  register_volumetric<Residual>(fm);
  write_graph<Residual>(fm, "e_volumetric.dot");
}

void Mechanics::build_error_neumann(FieldManager fm) {
  set_error();
  register_neumann<Residual>(fm);
  write_graph<Residual>(fm, "e_neumann.dot");
}

Mechanics* create_mech(ParameterList const& p, goal::Discretization* d) {
  return new Mechanics(p, d);
}

void destroy_mech(Mechanics* p) {
  delete p;
}

} // end namespace ml
