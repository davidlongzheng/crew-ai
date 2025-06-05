# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Crew AI** is a deep learning project that trains an AI agent to play "The Crew" card game using Proximal Policy Optimization (PPO). Currently achieves 75% win rate on a simplified version. Uses a hybrid C++/Python architecture for performance optimization.

## Core Commands

### Build & Development

```bash
# Build C++ components (must be done first)
./build.sh

# Install Python dependencies
pip install -r requirements.txt

# Run tests (excludes slow tests by default)
pytest

# Run all tests including slow ones
pytest -m "not slow"

# Run specific test file
pytest src/ai/tests/test_ai.py

# Linting (configured in pyproject.toml)
ruff check .
ruff format .

# Type checking
mypy src/

# Frontend development
cd src/frontend
yarn dev        # Development server with Turbopack
yarn build      # Production build
yarn lint       # ESLint
```

### Performance Analysis

```bash
# Generate flamegraph for C++ performance profiling
./gen_flamegraph.sh

# Run C++ unit tests
./build/test_rollout
```

## Architecture

### Hybrid C++/Python Design

- **C++ Core** (`src/cpp_game/`): High-performance game engine, rollouts, tree search
- **Python AI Layer** (`src/ai/`): Neural networks, training loops, experimentation
- **PyBind11 Integration**: Seamless interop via `src/cpp_game/bindings.cc`

### Key Components

#### C++ Game Engine (`src/cpp_game/`)

- `engine.h/cc`: Core game state management and logic
- `rollout.h/cc`: Vectorized rollout execution for training data
- `tree_search.h/cc`: Monte Carlo Tree Search implementation
- `featurizer.h`: Converts game states to neural network features
- `settings.h/cc`: Configurable game parameters and variants
- Thread-safe utilities: lock-free pools, thread pools

#### Python AI System (`src/ai/`)

- `ai.py`: Main AI interface integrating models with tree search
- `train.py`: PPO training loop with GAE and experience replay
- `models.py`: Policy-Value networks with LSTM memory
- `rollout.py`: Python interface to C++ batch rollout system
- `tree_search.py`: Tree search configuration and utilities

#### Game Logic (`src/game/`)

- `engine.py`: Python wrapper for C++ engine
- `state.py`: Game state representation
- `tasks.py`: Game objectives and task definitions
- `types.py`: Core data types (Cards, Actions, Signals)

#### Web Application

- **Backend** (`src/backend/`): FastAPI REST API for game interactions
- **Frontend** (`src/frontend/`): Next.js React app with WebSocket support
- Purpose: Human vs AI gameplay interface (TODO: not fully implemented)

### Performance Patterns

- **Batch Processing**: All operations vectorized for training efficiency
- **Memory Management**: Lock-free pools and efficient C++ data structures
- **Parallel Execution**: Multi-threaded tree search and rollout generation
- **Feature Engineering**: Custom game state featurizers for card game domain

## Development Workflow

### Building Changes

1. Always run `./build.sh` after modifying C++ code
2. The build creates the `cpp_game` Python module that AI code depends on
3. CMake configuration expects virtual environment at `./venv/`

### Testing Strategy

- **Fast Tests**: Default pytest run excludes slow model training tests
- **Slow Tests**: Include with `pytest` (no `-m "not slow"` filter)
- **C++ Tests**: Direct executable testing for core engine logic
- **Integration Tests**: Python tests that exercise C++/Python boundary

### Experimentation

- **Jupyter Notebooks**: 11 analysis notebooks in `/notebooks/` for debugging and visualization
- **Hyperparameter Tuning**: Optuna integration for optimization
- **Logging**: TensorBoard integration for training metrics
- **Remote Development**: `runpod_sync.sh` for cloud GPU training

### Code Quality Standards

- **Ruff**: Configured for Python formatting and linting (line length 88)
- **MyPy**: Type checking (ignores missing torchvision imports)
- **C++20**: Modern C++ standards with CMake build system
- **Testing**: Comprehensive test coverage with performance test separation

## Key Technical Details

### Model Architecture

- Policy-Value networks with shared backbone
- LSTM memory for sequential decision making
- Custom featurizers for card game state representation

### Training Pipeline

- PPO with Generalized Advantage Estimation (GAE)
- Tree search integration for stronger training signal
- Curriculum learning support for progressive difficulty

### Game Engine Features

- Configurable game variants through settings system
- Deterministic execution with controllable randomness
- Support for different task configurations and objectives
