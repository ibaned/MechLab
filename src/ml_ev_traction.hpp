#ifndef ml_ev_traction_hpp
#define ml_ev_traction_hpp

/// @file ml_ev_traction.hpp

#include <Phalanx_Evaluator_Macros.hpp>
#include <goal_dimension.hpp>

/// @cond
namespace goal {
class Indexer;
class Discretization;
class SolInfo;
}
/// @endcond

namespace ml {

PHX_EVALUATOR_CLASS_PP(Traction)

  public:

    /// @brief Construct the tractions evaluator.
    /// @brief The traction parameter list.
    /// @param u The displacement fields.
    /// @param bc The boundary condition array.
    /// @param i The linear algebra indexer.
    /// @param type The entity to operate on.
    Traction(
        std::vector<goal::Field*> const& u,
        Teuchos::Array<std::string> const& bc,
        goal::Indexer* i,
        int type);

  private:

    using Dummy = goal::Dummy;
    using Node = goal::Node;
    using Ent = goal::Ent;
    using IP = goal::IP;

    std::vector<goal::Field*> disp;
    Teuchos::Array<std::string> bc;
    goal::Indexer* indexer;
    goal::SolInfo* info;

    int num_nodes;
    int num_ips;
    int num_dims;

    // input
    PHX::MDField<const double, Ent, IP> wdv;
    std::vector<PHX::MDField<const double, Ent, Node, IP> > w;

PHX_EVALUATOR_CLASS_END

} // end namespace ml

#endif
