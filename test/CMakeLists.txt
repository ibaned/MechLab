if(NOT BUILD_TESTING)
  return()
endif()

set(MPIEXE mpirun)
set(MPIFLAGS -np)

function(copy input)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/${input}
    ${CMAKE_CURRENT_BINARY_DIR}/${input}
    COPYONLY)
endfunction()

include(meshgen.cmake)

copy(in.yaml)

add_custom_target(pretest COMMAND)
add_dependencies(pretest meshgen)
