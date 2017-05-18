#include "ml_mechanics.hpp"

template <typename EvalT>
void ml::Mechanics::register_neumann(goal::FieldManager fm) {
  (void)fm;
}

template void ml::Mechanics::register_neumann<goal::Traits::Residual>(goal::FieldManager fm);
template void ml::Mechanics::register_neumann<goal::Traits::Jacobian>(goal::FieldManager fm);
