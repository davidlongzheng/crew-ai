cmake_minimum_required(VERSION 3.10)
project(crew_ai)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# sanitize build
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")

# perf build
# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")  

# Find Python interpreter
# Needed for pybind11
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/python")
    set(Python_EXECUTABLE "${CMAKE_CURRENT_SOURCE_DIR}/venv/bin/python")
endif()
find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)

# Add pybind11 subdirectory
add_subdirectory(src/cpp_game/extern/pybind11)

# Add source files
set(SOURCES
    src/cpp_game/engine.cc
    src/cpp_game/settings.cc
    src/cpp_game/tasks.cc
    src/cpp_game/rollout.cc
    src/cpp_game/thread_pool.cc
    src/cpp_game/tree_search.cc
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
    src/cpp_game/thread_pool.h
    src/cpp_game/lockfree_pool.h
    src/cpp_game/tree_search.h
    src/cpp_game/featurizer.h
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

# Install game library into lib directory
install(TARGETS game
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install the Python module into venv
message(STATUS "Python site-packages directory: ${Python_SITEARCH}")
install(TARGETS cpp_game DESTINATION ${Python_SITEARCH}) 