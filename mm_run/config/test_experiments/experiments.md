# Integration Test Experiments
This directory contains integration test experiments designed to verify all components.

## Test Experiments

### 1. `test_end_stop_velocity.yaml`
**Purpose**: Tests PathPlanner end_stop with velocity checks

**Key Features**:
- Base and EE PathPlanner with end_stop enabled
- Verifies velocity check in checkFinished method
- Tests state dictionary velocity extraction

### 2. `test_orientation_tracking.yaml`
**Purpose**: Tests orientation-only movement

**Key Features**:
- Two sequential tasks with same position, different orientation
- Verifies angular velocity in state dictionary
- Tests state dictionary structure when only orientation changes

### 3. `test_complex_sequence.yaml`
**Purpose**: Tests complex multi-task sequence

**Key Features**:
- Sequential base and EE tasks
- TaskManager sequencing
- State dictionary format throughout task transitions

### 4. `test_base_path.yaml`
**Purpose**: Tests PathPlanner for base movement

**Key Features**:
- PathPlanner for base (SE2)
- Coordination with end-effector tasks
- Base path following with EE tracking
- State dictionary format verification

### 5. `test_position_only.yaml`
**Purpose**: Edge case test for position-only movement

**Key Features**:
- Position change with no orientation change
- Verifies angular velocities are zero/negligible in state dictionary
- Tests state dictionary structure edge case

## Running Tests

To run a test experiment:

```bash
python3 mm_control/scripts/generate_acados_code.py --config mm_run/config/test_experiments/test_<name>.yaml
python3 mm_run/scripts/experiment.py --config mm_run/config/test_experiments/test_<name>.yaml --GUI
```
