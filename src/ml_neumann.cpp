#include <goal_discretization.hpp>
#include <goal_ev_basis.hpp>

#include "ml_mechanics.hpp"

using Teuchos::rcp;
using goal::Traits;

template <typename EvalT>
void ml::Mechanics::register_neumann(goal::FieldManager fm) {

  std::vector<goal::Field*> disp = u;

  //  get the current entity type to operate on
  auto type = disc->get_side_type(side_set);

  // bail if there are no entities to operate on in this side set
  if (type < 0) {
    goal::set_extended_data_type_dims(indexer, fm, 0);
    fm->postRegistrationSetupForType<EvalT>(NULL);
    return;
  }

  // set the displacement field basis functions
  if (is_primal || is_dual) {
    auto ev = rcp(new goal::Basis<EvalT, Traits>(disp[0], type));
    fm->registerEvaluator<EvalT>(ev);
  }

  // compute tractions if needed for this side set
  if (traction_map.count(side_set)) {
  }

  // set the FAD data and finalize the PHX field maanger registration.
  goal::set_extended_data_type_dims(indexer, fm, type);
  fm->postRegistrationSetupForType<EvalT>(NULL);
}

template void ml::Mechanics::register_neumann<goal::Traits::Residual>(goal::FieldManager fm);
template void ml::Mechanics::register_neumann<goal::Traits::Jacobian>(goal::FieldManager fm);
