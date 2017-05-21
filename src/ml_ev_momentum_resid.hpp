#ifndef ml_ev_momentum_resid_hpp
#define ml_ev_momentum_resid_hpp

/// @file ml_ev_momentum_resid.hpp

#include <Phalanx_Evaluator_Macros.hpp>
#include <goal_dimension.hpp>

/// @cond
namespace goal {
class Field;
}
/// @endcond

namespace ml {

PHX_EVALUATOR_CLASS(MomentumResid)

  public:

    /// @brief Construct the momentum residual evaluator.
    /// @param u The displacement fields.
    /// @param type The type of entity to operate on.
    MomentumResid(std::vector<goal::Field*> const& u, int type);

  private:

    using Node = goal::Node;
    using Dim = goal::Dim;
    using Ent = goal::Ent;
    using IP = goal::IP;

    int num_nodes;
    int num_ips;
    int num_dims;

    // input
    PHX::MDField<const double, Ent, IP> wdv;
    std::vector<PHX::MDField<const double, Ent, Node, IP, Dim> > grad_w;
    PHX::MDField<const ScalarT, Ent, IP, Dim, Dim> stress;

    // output
    std::vector<PHX::MDField<ScalarT, Ent, Node> > resid;

PHX_EVALUATOR_CLASS_END

} // namespace ml

#endif
