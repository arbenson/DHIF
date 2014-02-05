#
# Stanford University
#
cmake_minimum_required(VERSION 2.8.5) # for the new FindMPI module
project(DHIF)

set(DHIF_VERSION_MAJOR 0)
set(DHIF_VERSION_MINOR 2)

option(TIME_MULTIPLY "Instrument HIF with timers." OFF)

option(DHIF_TESTS "Build the test drivers" ON)

option(MEMORY_INFO "Collect memory infomation during the code." OFF)

add_subdirectory(${PROJECT_SOURCE_DIR}/external/Elemental)
include_directories(${PROJECT_BINARY_DIR}/external/Elemental/include)
include_directories(${MPI_CXX_INCLUDE_PATH})
set(ELEM_CMAKE_DIR ${PROJECT_SOURCE_DIR}/external/Elemental/cmake)

# Store all of our system-specific definitions
configure_file(${PROJECT_SOURCE_DIR}/cmake/config.h.cmake
               ${PROJECT_BINARY_DIR}/include/DHIF/config.h)
install(FILES ${PROJECT_BINARY_DIR}/include/DHIF/config.h
        DESTINATION include/DHIF)

# Grab all of the .c, .cpp, .h, and .hpp DHIF files
file(GLOB_RECURSE DHIF_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     "src/*.c" "src/*.cpp" "src/*-incl.hpp")
file(GLOB_RECURSE DHIF_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     "include/*.h" "include/*.hpp")
set(DHIF_SRC "${DHIF_CPP};${DHIF_HEADERS}")

# Build the DHIF source
add_library(DHIF STATIC ${DHIF_SRC})
if(MANUAL_ELEMENTAL)
    target_link_libraries(DHIF ${ELEM_LIBS})
else()
    target_link_libraries(DHIF Elemental)
endif()
install(TARGETS DHIF DESTINATION lib)

# Copy the headers into the build directory
set(PREPARED_HEADERS)
foreach(HEADER ${DHIF_HEADERS})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${HEADER}
            ${CMAKE_CURRENT_BINARY_DIR}/${HEADER}
    DEPENDS "${PROJECT_SOURCE_DIR}/${HEADER}")
  list(APPEND PREPARED_HEADERS ${PROJECT_BINARY_DIR}/${HEADER})

  get_filename_component(HEADER_PATH ${HEADER} PATH)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${HEADER} DESTINATION ${HEADER_PATH})
endforeach()
add_custom_target(prepare_DHIF_headers DEPENDS ${PREPARED_HEADERS})
add_dependencies(DHIF prepare_DHIF_headers)

# Make sure the DHIF headers can be found
include_directories("${PROJECT_BINARY_DIR}/include")

if(BUILD_TESTS)
  set(TEST_DIR ${PROJECT_SOURCE_DIR}/tests)

  foreach(TYPE ${TEST_TYPES})
    set(OUTPUT_DIR "${PROJECT_BINARY_DIR}/bin/${TYPE}")
    foreach(TEST ${${TYPE}_TESTS})
      add_executable(${TYPE}-${TEST} ${TEST_DIR}/${TYPE}/${TEST}.cpp)
      target_link_libraries(${TYPE}-${TEST} DHIF cmake-dummy-lib)
      set_target_properties(${TYPE}-${TEST}
                            PROPERTIES OUTPUT_NAME ${TEST}
                            RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
      if(MPI_LINK_FLAGS)
        set_target_properties(${TYPE}-${TEST} PROPERTIES
          LINK_FLAGS ${MPI_LINK_FLAGS})
      endif()
      install(TARGETS ${TYPE}-${TEST} DESTINATION bin/${TYPE})
    endforeach()
  endforeach()
endif()
