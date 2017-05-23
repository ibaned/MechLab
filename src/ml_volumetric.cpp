#include <goal_discretization.hpp>
#include <goal_ev_gather.hpp>
#include <goal_ev_basis.hpp>
#include <goal_ev_interpolate.hpp>
#include <goal_ev_resid.hpp>

#include "ml_mechanics.hpp"
#include "ml_ev_kinematics.hpp"
#include "ml_ev_elastic.hpp"
#include "ml_ev_first_pk.hpp"
#include "ml_ev_momentum_resid.hpp"

using Teuchos::RCP;
using Teuchos::rcp;
using goal::Traits;

template <typename EvalT>
void ml::Mechanics::register_volumetric(goal::FieldManager fm) {

  std::vector<goal::Field*> disp = u;
  goal::Field* press = 0;

  // get the current entity type to operate on
  auto type = disc->get_elem_type(elem_set);

  // bail if there are no entities to operate on in this elem set
  if (type < 0) {
    goal::set_extended_data_type_dims(indexer, fm, 0);
    fm->postRegistrationSetupForType<EvalT>(NULL);
    return;
  }

  // get information specific to this element set
  auto es_name = disc->get_elem_set_name(elem_set);
  auto mp = params.sublist(es_name);

  { // gather the displacement fields
    auto ev = rcp(new goal::Gather<EvalT, Traits>(indexer, u, type));
    fm->registerEvaluator<EvalT>(ev);
  }

  { // set the displacement field basis functions
    auto ev = rcp(new goal::Basis<EvalT, Traits>(disp[0], type));
    fm->registerEvaluator<EvalT>(ev);
  }

  { // interpolate the displacement fields to integration points
    auto ev = rcp(new goal::Interpolate<EvalT, Traits>(disp, type));
    fm->registerEvaluator<EvalT>(ev);
  }

  { // compute kinematic quantities
    auto ev = rcp(new ml::Kinematics<EvalT, Traits>(disp, type));
    fm->registerEvaluator<EvalT>(ev);
  }

  { // compute the Cauchy stress tensor
    RCP<PHX::Evaluator<Traits> > ev;
    if (model == "elastic")
      ev = rcp(new ml::Elastic<EvalT, Traits>(disp, states, mp, type));
    fm->registerEvaluator<EvalT>(ev);
  }

  { // pull back the Cauchy stress tensor
    auto ev = rcp(new FirstPK<EvalT, Traits>(disp, press, small_strain, type));
    fm->registerEvaluator<EvalT>(ev);
  }

  // compute the weighted momentum residual
  if (is_primal || is_dual) {
    auto ev = rcp(new MomentumResid<EvalT, Traits>(disp, type));
    fm->registerEvaluator<EvalT>(ev);
  }

  // fill in the global residual-related data structures
  if (is_primal || is_dual) {
    auto ev = rcp(new goal::Resid<EvalT, Traits>(indexer, disp, type, is_dual));
    fm->registerEvaluator<EvalT>(ev);
    fm->requireField<EvalT>(*ev->evaluatedFields()[0]);
  }

  // set the FAD data and finalize the PHX field maanger registration.
  goal::set_extended_data_type_dims(indexer, fm, type);
  fm->postRegistrationSetupForType<EvalT>(NULL);

}

template void ml::Mechanics::register_volumetric<goal::Traits::Residual>(goal::FieldManager fm);
template void ml::Mechanics::register_volumetric<goal::Traits::Jacobian>(goal::FieldManager fm);
