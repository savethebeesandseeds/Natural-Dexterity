cmake_minimum_required(VERSION 3.20)
project(humanoid_walk LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_RPATH "/opt/libtorch/lib;/opt/mujoco/lib")
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(JPEG REQUIRED)
find_package(yaml-cpp REQUIRED)
find_library(OSMESA_LIBRARY OSMesa REQUIRED)

# We use the precompiled CUDA kernels in the official C++ distribution.
# Linking these shared libraries needs no Python or CUDA source compiler.
add_library(torch_bundle INTERFACE)
target_include_directories(torch_bundle SYSTEM INTERFACE
  /opt/libtorch/include /opt/libtorch/include/torch/csrc/api/include)
target_compile_definitions(torch_bundle INTERFACE _GLIBCXX_USE_CXX11_ABI=1)
target_link_libraries(torch_bundle INTERFACE
  "-Wl,--no-as-needed" /opt/libtorch/lib/libtorch.so
  /opt/libtorch/lib/libtorch_cpu.so /opt/libtorch/lib/libtorch_cuda.so
  /opt/libtorch/lib/libc10.so /opt/libtorch/lib/libc10_cuda.so "-Wl,--as-needed")

add_library(humanoid_sim STATIC src/simulation.cpp)
target_include_directories(humanoid_sim PUBLIC include /opt/nlohmann-json/include /opt/mujoco/include)
target_link_libraries(humanoid_sim PUBLIC torch_bundle /opt/mujoco/lib/libmujoco.so
  yaml-cpp OpenSSL::Crypto JPEG::JPEG ${OSMESA_LIBRARY} Threads::Threads)
target_compile_options(humanoid_sim PRIVATE -Wall -Wextra -Wpedantic)

add_executable(humanoid-viewer src/server.cpp)
target_include_directories(humanoid-viewer PRIVATE /opt/cpp-httplib/include)
target_link_libraries(humanoid-viewer PRIVATE humanoid_sim)

add_executable(humanoid-evaluate src/evaluate.cpp)
target_link_libraries(humanoid-evaluate PRIVATE humanoid_sim)

add_executable(humanoid-render-proof src/render_proof.cpp)
target_link_libraries(humanoid-render-proof PRIVATE humanoid_sim)

add_executable(humanoid-goal-evaluate src/goal_evaluate.cpp)
target_link_libraries(humanoid-goal-evaluate PRIVATE humanoid_sim)

add_executable(humanoid-gpu-check src/gpu_check.cpp)
target_link_libraries(humanoid-gpu-check PRIVATE torch_bundle)

add_executable(humanoid-train training/train.cpp)
target_link_libraries(humanoid-train PRIVATE humanoid_sim)

add_executable(humanoid-walk-train training/walk_train.cpp)
target_link_libraries(humanoid-walk-train PRIVATE humanoid_sim)

add_executable(humanoid-compare-checkpoints training/compare_checkpoint.cpp)
target_include_directories(humanoid-compare-checkpoints PRIVATE /opt/nlohmann-json/include)
target_link_libraries(humanoid-compare-checkpoints PRIVATE torch_bundle OpenSSL::Crypto)

add_executable(humanoid-distill training/distill.cpp)
target_link_libraries(humanoid-distill PRIVATE humanoid_sim)

include(CTest)
if(BUILD_TESTING)
  add_executable(humanoid-ppo-math-test training/ppo_math_test.cpp)
  add_test(NAME ppo_math COMMAND humanoid-ppo-math-test)
  add_executable(humanoid-goal-criteria-test training/goal_criteria_test.cpp)
  target_include_directories(humanoid-goal-criteria-test PRIVATE include /opt/nlohmann-json/include)
  add_test(NAME goal_criteria COMMAND humanoid-goal-criteria-test)
  add_executable(humanoid-swing-diagnostics-test training/swing_diagnostics_test.cpp)
  target_include_directories(humanoid-swing-diagnostics-test PRIVATE /opt/nlohmann-json/include)
  add_test(NAME swing_diagnostics COMMAND humanoid-swing-diagnostics-test)
endif()
