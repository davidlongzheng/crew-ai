cmake_minimum_required(VERSION 3.10)
project(crew_ai)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(Python_EXECUTABLE "/Users/davidzheng/projects/crew-ai/crew-ai/venv/bin/python")
set(Torch_DIR "/Users/davidzheng/projects/crew-ai/crew-ai/cpp_lib/libtorch/share/cmake/Torch")

# Find Python and pybind11
find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)

find_package(pybind11 CONFIG REQUIRED)

find_package(Torch REQUIRED)

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

# Add the C++ library
add_library(game
    ${SOURCES}
    ${HEADERS}
)

# Add include directories
target_include_directories(game PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp_game
)

# Set compiler flags
if(MSVC)
    target_compile_options(game PRIVATE /W4)
else()
    target_compile_options(game PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Add the Python bindings
pybind11_add_module(cpp_game src/cpp_game/bindings.cc)
target_link_libraries(cpp_game PRIVATE game)

target_link_libraries(cpp_game PRIVATE ${TORCH_LIBRARIES})

# Install rules
install(TARGETS game
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install the Python module
install(TARGETS cpp_game DESTINATION ${Python_SITEARCH}) 