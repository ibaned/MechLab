#include <goal_discretization.hpp>
#include <goal_ev_gather.hpp>
#include <goal_ev_basis.hpp>
#include <goal_ev_interpolate.hpp>

#include "ml_mechanics.hpp"
#include "ml_ev_kinematics.hpp"
#include "ml_ev_elastic.hpp"
#include "ml_ev_first_pk.hpp"

using Teuchos::RCP;
using Teuchos::rcp;
using goal::Traits;

template <typename EvalT>
void ml::Mechanics::register_volumetric(goal::FieldManager fm) {

  std::vector<goal::Field*> disp = u;
  goal::Field* press = 0;

  auto type = disc->get_elem_type(elem_set);
  if (type < 0) { // no entities in this set for this elem set
    goal::set_extended_data_type_dims(indexer, fm, 0);
    fm->postRegistrationSetupForType<EvalT>(NULL);
    return;
  }

  { // gather all fields
    auto ev = rcp(new goal::Gather<EvalT, Traits>(indexer, disp, type));
    fm->registerEvaluator<EvalT>(ev); }

  { // set the field basis functions
    auto ev = rcp(new goal::Basis<EvalT, Traits>(disp[0], type));
    fm->registerEvaluator<EvalT>(ev); }

  { // interpolate the fields to integration points
    auto ev = rcp(new goal::Interpolate<EvalT, Traits>(disp, type));
    fm->registerEvaluator<EvalT>(ev); }

  { // compute kinematic quantities
    auto ev = rcp(new ml::Kinematics<EvalT, Traits>(disp, type));
    fm->registerEvaluator<EvalT>(ev); }

  auto es_name = disc->get_elem_set_name(elem_set);
  auto mp = params.sublist(es_name);

  { // compute the Cauchy stress tensor
    RCP<PHX::Evaluator<Traits> > ev;
    if (model == "elastic")
      ev = rcp(new ml::Elastic<EvalT, Traits>(disp, states, mp, type));
    fm->registerEvaluator<EvalT>(ev); }

  { // pull back the Cauchy stress tensor
    auto ev = rcp(new FirstPK<EvalT, Traits>(disp, press, small_strain, type));
    fm->registerEvaluator<EvalT>(ev);
    fm->requireField<EvalT>(*ev->evaluatedFields()[0]); }

  // set the FAD data and finalize the PHX field maanger registration.
  goal::set_extended_data_type_dims(indexer, fm, type);
  fm->postRegistrationSetupForType<EvalT>(NULL);

}

template void ml::Mechanics::register_volumetric<goal::Traits::Residual>(goal::FieldManager fm);
template void ml::Mechanics::register_volumetric<goal::Traits::Jacobian>(goal::FieldManager fm);
