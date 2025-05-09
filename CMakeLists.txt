cmake_minimum_required(VERSION 3.10)
project(crew_ai)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")

# Find Python interpreter
# Needed for pybind11
set(Python_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/python")
find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)

# Add pybind11 subdirectory
add_subdirectory(src/cpp_game/extern/pybind11)

# Add source files
set(SOURCES
    src/cpp_game/engine.cc
    src/cpp_game/tasks.cc
    src/cpp_game/rollout.cc
)

# Add header files
set(HEADERS
    src/cpp_game/engine.h
    src/cpp_game/state.h
    src/cpp_game/settings.h
    src/cpp_game/tasks.h
    src/cpp_game/types.h
    src/cpp_game/utils.h
    src/cpp_game/rollout.h
)

# Add the C++ library called "game"
add_library(game
    ${SOURCES}
    ${HEADERS}
)
target_include_directories(game PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp_game
)
target_link_libraries(game PRIVATE Python::Python)
target_link_libraries(game PRIVATE pybind11::headers)

# Add pybind11 module called "cpp_game"
pybind11_add_module(cpp_game src/cpp_game/bindings.cc)
target_link_libraries(cpp_game PRIVATE game)

# Add test
add_executable(test_rollout src/cpp_game/test_rollout.cc)
target_link_libraries(test_rollout PRIVATE Python::Python)
target_link_libraries(test_rollout PRIVATE pybind11::headers)
target_link_libraries(test_rollout PRIVATE game)

# Install rules
install(TARGETS game
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install the Python module
install(TARGETS cpp_game DESTINATION ${Python_SITEARCH}) 