if(NOT BUILD_TESTING)
  return()
endif()

set(MPIEXE mpirun)
set(MPIFLAGS -np)
set(MLEXE ../src/MechLab)

function(copy input)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/${input}
    ${CMAKE_CURRENT_BINARY_DIR}/${input}
    COPYONLY)
endfunction()

function(mpi_test testname np)
  copy(${testname}.yaml)
  add_test(
    NAME ${testname}
    COMMAND ${MPIEXE} ${MPIFLAGS} ${np} ${MLEXE} "${testname}.yaml")
endfunction()

include(meshgen.cmake)

mpi_test(static_elast_p1_1p 1)
mpi_test(static_elast_p1_4p 4)
mpi_test(static_elast_p2_4p 4)

add_custom_target(pretest COMMAND)
add_dependencies(pretest meshgen)

bob_end_subdir()
