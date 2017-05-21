#ifndef ml_ev_first_pk_hpp
#define ml_ev_first_pk_hpp

#include <Phalanx_Evaluator_Macros.hpp>
#include <goal_dimension.hpp>

/// @cond
namespace goal {
class Field;
}
/// @endcond

namespace ml {

PHX_EVALUATOR_CLASS(FirstPK)

  public:

    /// @brief Construct the first Piola-Kirchhoff evaluator.
    /// @param u The displacement fields.
    /// @param p The (optional) pressure field.
    /// @param small True if small strain should be used.
    /// @param type The entity type to operate on.
    FirstPK(
        std::vector<goal::Field*> const& u,
        goal::Field* p,
        bool small,
        int type);

  private:

    using Ent = goal::Ent;
    using IP = goal::IP;
    using Dim = goal::Dim;

    int num_ips;
    int num_dims;

    bool small_strain;
    bool have_pressure;

    // input
    PHX::MDField<const ScalarT, Ent, IP, Dim, Dim> def_grad;
    PHX::MDField<const ScalarT, Ent, IP> det_def_grad;
    PHX::MDField<const ScalarT, Ent, IP, Dim, Dim> cauchy;
    PHX::MDField<const ScalarT, Ent, IP> pressure;

    // output
    PHX::MDField<ScalarT, Ent, IP, Dim, Dim> first_pk;

PHX_EVALUATOR_CLASS_END

} // end namespace ml

#endif
