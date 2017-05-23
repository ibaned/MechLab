#include <goal_control.hpp>
#include <goal_field.hpp>
#include <goal_states.hpp>
#include <goal_traits.hpp>
#include <goal_workset.hpp>

#include "ml_ev_traction.hpp"

namespace ml {

template <typename EVALT, typename TRAITS>
Traction<EVALT, TRAITS>::Traction(
    std::vector<goal::Field*> const& u,
    Teuchos::Array<std::string> const& bc,
    goal::Indexer* i,
    int type) {
}

PHX_POST_REGISTRATION_SETUP(Traction, data, fm) {
  (void)data;
  (void)fm;
}

PHX_PRE_EVALUATE_FIELDS(Traction, info) {
}

PHX_EVALUATE_FIELDS(Traction, workset) {
}

PHX_POST_EVALUATE_FIELDS(Traction, info) {
  (void)info;
}

} // end namespace ml
