#ifndef ml_physics_hpp
#define ml_physics_hpp

/// @file ml_physics.hpp

#include <goal_physics.hpp>
#include <Teuchos_ParameterList.hpp>

/// @cond
namespace goal {
class States;
class Discretization;
}
/// @endcond

namespace ml {

using Teuchos::ParameterList;

/// @brief The mechanics physics class.
/// @details This class is responsible for defining the primal, dual,
/// and error models for a total Lagrangian description of the balance
/// of linear momementum.
class Mechanics : public goal::Physics {

  public:

    /// @brief Construct the physics class.
    /// @param p A relevant parameter list.
    /// @param d The relevant discretization object.
    Mechanics(ParameterList const& p, goal::Discretization* d);

    /// @brief Returns the Dirichlet bc parameters.
    ParameterList const& get_dbc_params();

  public:

    /// @brief FieldManager type.
    using FieldManager = goal::FieldManager;
    /// @brief FieldManagers type.
    using FieldManagers = goal::FieldManagers;

  protected:

    void set_primal();
    void set_dual();
    void set_error();

    void build_primal_volumetric(FieldManager fm);
    void build_primal_neumann(FieldManager fm);

    void build_dual_volumetric(FieldManager fm);
    void build_dual_neumann(FieldManager fm);

    void build_error_volumetric(FieldManager fm);
    void build_error_neumann(FieldManager fm);

    template <typename EvalT>
    void register_volumetric(FieldManager fm);

    template <typename EvalT>
    void register_neumann(FieldManager fm);

    ParameterList params;
    goal::States* states;
};

/// @brief Create a mechanics physics object.
/// @param p The relevant parameter list defining the physics.
/// @param d The relevant goal Discretization object.
Mechanics* create_mech(ParameterList const& p, goal::Discretization* d);

/// @brief Destroy a physics object.
/// @param p The \ref ml::Mechanics object to destroy.
void destroy_mech(Mechanics* p);

} // end namespace ml

#endif
