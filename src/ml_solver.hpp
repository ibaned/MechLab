#ifndef ml_solver_hpp
#define ml_solver_hpp

/// @file ml_solver.hpp

/// @cond
namespace Teuchos {
class ParameteList;
}
/// @endcond

namespace ml {

using Teuchos::ParameterList;

/// @brief An abstract solver interface.
class Solver {

  public:

    /// @brief Construct the solver.
    Solver();

    /// @brief Destroy the solver.
    /// @details This does nothing.
    virtual ~Solver();

    /// @brief Run the solver.
    /// @details As implemented by a derived class.
    virtual void solve() = 0;
};

/// @brief Create a solver.
/// @param p The full parameter list describing the solver.
Solver* create_solver(ParameterList const& p);

/// @brief Destroy a solver.
void destroy_solver(Solver* s);

} // end namespace ml

#endif
