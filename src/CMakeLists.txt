include_directories(
  ${CMAKE_SOURCE_DIR}/ext
  ${CMAKE_SOURCE_DIR}/src
)


add_definitions(-DEIGEN_DEFAULT_TO_ROW_MAJOR)

# gcc 4.7 workaround:
# execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
#                 OUTPUT_VARIABLE GCC_VERSION)
# if (GCC_VERSION VERSION_GREATER 4.6)
#   message("GCC > 4.6 detected")
#    add_definitions(-D_GLIBCXX_HAVE_GTHR_DEFAULT)
# endif()

set(OpenRAVE_BOTH_LIBRARIES ${OpenRAVE_LIBRARIES} ${OpenRAVE_CORE_LIBRARIES})


add_subdirectory(trajopt)
add_subdirectory(utils)


if(NOT APPLE)
endif()
