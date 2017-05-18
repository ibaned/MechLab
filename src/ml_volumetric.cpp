#include "ml_mechanics.hpp"

template <typename EvalT>
void ml::Mechanics::register_volumetric(goal::FieldManager fm) {
  (void)fm;
}

template void ml::Mechanics::register_volumetric<goal::Traits::Residual>(goal::FieldManager fm);
template void ml::Mechanics::register_volumetric<goal::Traits::Jacobian>(goal::FieldManager fm);
