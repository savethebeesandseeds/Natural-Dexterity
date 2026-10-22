cmake_minimum_required(VERSION 3.25)

project(droid_blocks VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
find_package(Threads REQUIRED)

find_path(
  MUJOCO_INCLUDE_DIR
  NAMES mujoco/mujoco.h
  HINTS /opt/mujoco/include
  REQUIRED
)
find_library(
  MUJOCO_LIBRARY
  NAMES mujoco
  HINTS /opt/mujoco/lib
  REQUIRED
)
find_path(
  NLOHMANN_JSON_INCLUDE_DIR
  NAMES nlohmann/json.hpp
  HINTS /opt/nlohmann-json/include
  REQUIRED
)
find_path(
  CPP_HTTPLIB_INCLUDE_DIR
  NAMES httplib.h
  HINTS /opt/cpp-httplib/include
  REQUIRED
)

add_library(
  droid_core
  STATIC
  src/environment.cpp
  src/construction.cpp
  src/module_catalog.cpp
  src/simulation.cpp
)
target_include_directories(
  droid_core
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${NLOHMANN_JSON_INCLUDE_DIR}"
  PRIVATE
    "${MUJOCO_INCLUDE_DIR}"
)
target_compile_definitions(
  droid_core
  PUBLIC
    DROID_MODULE_CATALOG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/module_catalog.json"
)
target_compile_options(droid_core PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(
  droid_core
  PUBLIC
    "${MUJOCO_LIBRARY}"
    Threads::Threads
    "${CMAKE_DL_LIBS}"
)

add_library(
  droid_agent_core
  STATIC
  src/baseline.cpp
  src/body_controller.cpp
  src/construction_session.cpp
  src/light_session.cpp
  src/light_learner.cpp
  src/light_context.cpp
  src/light_prediction.cpp
  src/motion_explorer.cpp
  src/evaluation.cpp
  src/learner.cpp
  src/playground.cpp
  src/policy.cpp
)
target_compile_options(droid_agent_core PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid_agent_core PUBLIC droid_core)

add_executable(droid-blocks-server src/main.cpp)
target_include_directories(droid-blocks-server PRIVATE "${CPP_HTTPLIB_INCLUDE_DIR}")
target_compile_options(droid-blocks-server PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-blocks-server PRIVATE droid_agent_core)

add_executable(droid-agent src/agent_cli.cpp)
target_compile_options(droid-agent PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-agent PRIVATE droid_agent_core)

add_executable(droid-native-tests tests/native_contract_tests.cpp)
target_compile_options(droid-native-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-tests PRIVATE droid_core)

add_executable(
  droid-native-simulation-tests
  tests/native_simulation_parity_tests.cpp
)
target_compile_options(
  droid-native-simulation-tests
  PRIVATE -Wall -Wextra -Wpedantic
)
target_link_libraries(droid-native-simulation-tests PRIVATE droid_core)

add_executable(
  droid-native-environment-tests
  tests/native_environment_tests.cpp
)
target_compile_options(
  droid-native-environment-tests
  PRIVATE -Wall -Wextra -Wpedantic
)
target_link_libraries(droid-native-environment-tests PRIVATE droid_core)

add_executable(
  droid-native-baseline-tests
  tests/native_baseline_tests.cpp
)
target_compile_options(
  droid-native-baseline-tests
  PRIVATE -Wall -Wextra -Wpedantic
)
target_link_libraries(droid-native-baseline-tests PRIVATE droid_agent_core)

add_executable(
  droid-native-policy-tests
  tests/native_policy_tests.cpp
)
target_compile_options(
  droid-native-policy-tests
  PRIVATE -Wall -Wextra -Wpedantic
)
target_link_libraries(droid-native-policy-tests PRIVATE droid_agent_core)

add_executable(
  droid-native-training-tests
  tests/native_training_tests.cpp
)
target_compile_options(
  droid-native-training-tests
  PRIVATE -Wall -Wextra -Wpedantic
)
target_link_libraries(droid-native-training-tests PRIVATE droid_agent_core)

add_executable(droid-native-playground-tests tests/native_playground_tests.cpp)
target_compile_options(droid-native-playground-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-playground-tests PRIVATE droid_agent_core)

add_executable(droid-native-body-tests tests/native_body_tests.cpp)
target_compile_options(droid-native-body-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-body-tests PRIVATE droid_core)
add_executable(droid-native-body-controller-tests tests/native_body_controller_tests.cpp)
target_compile_options(droid-native-body-controller-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-body-controller-tests PRIVATE droid_agent_core)
add_executable(droid-body-experiment src/body_experiment_cli.cpp)
target_compile_options(droid-body-experiment PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-body-experiment PRIVATE droid_agent_core)

add_executable(droid-native-construction-tests tests/native_construction_tests.cpp)
target_compile_options(droid-native-construction-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-construction-tests PRIVATE droid_core)
add_executable(droid-native-motion-explorer-tests tests/native_motion_explorer_tests.cpp)
target_compile_options(droid-native-motion-explorer-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-motion-explorer-tests PRIVATE droid_agent_core)
add_executable(droid-native-construction-session-tests tests/native_construction_session_tests.cpp)
target_compile_options(droid-native-construction-session-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-construction-session-tests PRIVATE droid_agent_core)
add_executable(droid-construction-experiment src/construction_experiment_cli.cpp)
target_compile_options(droid-construction-experiment PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-construction-experiment PRIVATE droid_agent_core)

add_executable(droid-native-construction-light-tests tests/native_construction_light_tests.cpp)
target_compile_options(droid-native-construction-light-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-construction-light-tests PRIVATE droid_core)
add_executable(droid-native-light-learner-tests tests/native_light_learner_tests.cpp)
target_compile_options(droid-native-light-learner-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-light-learner-tests PRIVATE droid_agent_core)
add_executable(droid-native-light-session-tests tests/native_light_session_tests.cpp)
target_compile_options(droid-native-light-session-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-light-session-tests PRIVATE droid_agent_core)
add_executable(droid-light-experiment src/light_experiment_cli.cpp)
target_compile_options(droid-light-experiment PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-light-experiment PRIVATE droid_agent_core)

add_executable(droid-light-adaptation src/light_adaptation_cli.cpp)
target_compile_options(droid-light-adaptation PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-light-adaptation PRIVATE droid_agent_core)

add_executable(droid-light-retention src/light_retention_cli.cpp)
target_compile_options(droid-light-retention PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-light-retention PRIVATE droid_agent_core)

add_executable(droid-light-context src/light_context_cli.cpp)
target_compile_options(droid-light-context PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-light-context PRIVATE droid_agent_core)

add_executable(droid-native-light-context-tests tests/native_light_context_tests.cpp)
target_compile_options(droid-native-light-context-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-light-context-tests PRIVATE droid_agent_core)

add_executable(droid-light-prediction src/light_prediction_cli.cpp)
target_compile_options(droid-light-prediction PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-light-prediction PRIVATE droid_agent_core)

add_executable(droid-native-light-prediction-tests tests/native_light_prediction_tests.cpp)
target_compile_options(droid-native-light-prediction-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-light-prediction-tests PRIVATE droid_agent_core)

set(_droid_mujoco_rpath "/opt/mujoco/lib")
add_executable(droid-transfer-env src/transfer_env_cli.cpp)
target_compile_options(droid-transfer-env PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-transfer-env PRIVATE droid_agent_core)
add_executable(droid-native-construction-actuation-tests tests/native_construction_actuation_tests.cpp)
target_compile_options(droid-native-construction-actuation-tests PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(droid-native-construction-actuation-tests PRIVATE droid_agent_core)
set_target_properties(
  droid-transfer-env
  droid-native-construction-actuation-tests
  droid-blocks-server
  droid-agent
  droid-native-tests
  droid-native-simulation-tests
  droid-native-environment-tests
  droid-native-baseline-tests
  droid-native-policy-tests
  droid-native-training-tests
  droid-native-playground-tests
  droid-native-body-tests
  droid-native-body-controller-tests
  droid-body-experiment
  droid-native-construction-tests
  droid-native-motion-explorer-tests
  droid-native-construction-session-tests
  droid-construction-experiment
  droid-native-construction-light-tests
  droid-native-light-learner-tests
  droid-native-light-session-tests
  droid-light-experiment
  droid-light-adaptation
  droid-light-retention
  droid-light-context
  droid-native-light-context-tests
  droid-light-prediction
  droid-native-light-prediction-tests
  PROPERTIES
    BUILD_RPATH "${_droid_mujoco_rpath}"
    INSTALL_RPATH "${_droid_mujoco_rpath}"
)

# Docker validation and the development launcher build this one target. Keep
# the authoritative executable inventory here so those callers cannot drift.
add_custom_target(
  droid-native-validation
  DEPENDS
    droid-transfer-env
    droid-native-construction-actuation-tests
    droid-blocks-server
    droid-agent
    droid-native-tests
    droid-native-simulation-tests
    droid-native-environment-tests
    droid-native-baseline-tests
    droid-native-policy-tests
    droid-native-training-tests
    droid-native-playground-tests
    droid-native-body-tests
    droid-native-body-controller-tests
    droid-body-experiment
    droid-native-construction-tests
    droid-native-motion-explorer-tests
    droid-native-construction-session-tests
    droid-construction-experiment
  droid-native-construction-light-tests
  droid-native-light-learner-tests
  droid-native-light-session-tests
  droid-light-experiment
  droid-light-adaptation
  droid-light-retention
  droid-light-context
  droid-native-light-context-tests
  droid-light-prediction
  droid-native-light-prediction-tests
)

if(BUILD_TESTING)
  add_test(NAME native-construction-actuation COMMAND droid-native-construction-actuation-tests)
  set_tests_properties(native-construction-actuation PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME native-light-prediction COMMAND droid-native-light-prediction-tests)
  add_test(NAME native-light-context COMMAND droid-native-light-context-tests)
  add_test(NAME native-contract COMMAND droid-native-tests)
  add_test(
    NAME native-simulation-parity
    COMMAND droid-native-simulation-tests
  )
  add_test(
    NAME native-environment
    COMMAND droid-native-environment-tests
  )
  add_test(
    NAME native-baseline
    COMMAND droid-native-baseline-tests
  )
  add_test(
    NAME native-policy
    COMMAND droid-native-policy-tests
  )
  add_test(
    NAME native-training
    COMMAND droid-native-training-tests
  )
  add_test(NAME native-playground COMMAND droid-native-playground-tests)
  add_test(NAME native-body COMMAND droid-native-body-tests)
  add_test(NAME native-body-controller COMMAND droid-native-body-controller-tests)
  add_test(NAME native-construction COMMAND droid-native-construction-tests)
  add_test(NAME native-motion-explorer COMMAND droid-native-motion-explorer-tests)
  add_test(NAME native-construction-session COMMAND droid-native-construction-session-tests)
  add_test(NAME native-construction-light COMMAND droid-native-construction-light-tests)
  add_test(NAME native-light-learner COMMAND droid-native-light-learner-tests)
  add_test(NAME native-light-session COMMAND droid-native-light-session-tests)
  set_tests_properties(
    native-contract
    native-simulation-parity
    native-environment
    native-baseline
    native-policy
    native-training
    native-playground
    native-body
    native-body-controller
    native-construction
    native-motion-explorer
    native-construction-session
    native-construction-light
    native-light-learner
    native-light-session
    native-light-context
    native-light-prediction
    PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()
