#include <goal_discretization.hpp>
#include <goal_ev_gather.hpp>
#include <goal_ev_basis.hpp>
#include <goal_ev_interpolate.hpp>

#include "ml_mechanics.hpp"

using Teuchos::rcp;
using goal::Traits;

template <typename EvalT>
void ml::Mechanics::register_volumetric(goal::FieldManager fm) {

  auto type = disc->get_elem_type(elem_set);
  if (type < 0) { // no entities in this set for this elem set
    goal::set_extended_data_type_dims(indexer, fm, 0);
    fm->postRegistrationSetupForType<EvalT>(NULL);
    return;
  }

  { // gather all fields
    auto ev = rcp(new goal::Gather<EvalT, Traits>(indexer, u, type));
    fm->registerEvaluator<EvalT>(ev); }

  { // set the field basis functions
    auto ev = rcp(new goal::Basis<EvalT, Traits>(u[0], type));
    fm->registerEvaluator<EvalT>(ev); }

  { // interpolate the fields to integration points
    auto ev = rcp(new goal::Interpolate<EvalT, Traits>(u, type));
    fm->registerEvaluator<EvalT>(ev);
    fm->requireField<EvalT>(*ev->evaluatedFields()[0]); }

  // set the FAD data and finalize the PHX field maanger registration.
  goal::set_extended_data_type_dims(indexer, fm, type);
  fm->postRegistrationSetupForType<EvalT>(NULL);
}

template void ml::Mechanics::register_volumetric<goal::Traits::Residual>(goal::FieldManager fm);
template void ml::Mechanics::register_volumetric<goal::Traits::Jacobian>(goal::FieldManager fm);
