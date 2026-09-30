# A backend that ships separately from Proton cannot be part of this build, so the
# fixture is a standalone library that links against Proton and registers itself
# when loaded -- the same shape as a real out-of-tree backend.
add_library(ProtonTestRuntimeBackend SHARED RuntimeBackendFixture.cpp)

target_include_directories(ProtonTestRuntimeBackend
  PRIVATE
    "${PROTON_COMMON_DIR}/include"
    "${PROTON_SRC_DIR}/include"
)

target_link_libraries(ProtonTestRuntimeBackend PRIVATE proton)

# Next to libproton, so the test finds it through triton._C.
set_target_properties(ProtonTestRuntimeBackend PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "$<TARGET_FILE_DIR:proton>"
  OUTPUT_NAME "proton_test_runtime_backend"
)
