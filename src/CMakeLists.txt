set(ML_SOURCES
ml_mechanics.cpp
ml_volumetric.cpp
ml_neumann.cpp
ml_solver.cpp
ml_static_solver.cpp
ml_ev_kinematics.cpp
ml_ev_elastic.cpp
ml_ev_first_pk.cpp
ml_ev_momentum_resid.cpp
ml_ev_traction.cpp
main.cpp
)

add_executable(MechLab ${ML_SOURCES})
target_link_libraries(MechLab Goal::Goal)
bob_export_target(MechLab)

bob_end_subdir()
