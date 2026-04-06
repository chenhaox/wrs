# SEALP — Implementation Plan

> **S**equence-**A**ware **L**ayout **P**lanner for dual-arm furniture assembly.

Furniture assembly is a long-horizon sequential manipulation task in which a prescribed assembly sequence must be executed under tight geometric constraints and limited workspace. A valid sequence alone does not ensure executability: workspace layout—including robot base placement, part staging, and assembly-station configuration—fundamentally determines whether each step admits feasible dual-arm motion.

We study **sequence-aware workspace layout optimization for dual-arm furniture assembly**. Given an assembly sequence, each part is assigned a motion primitive, and step-wise start/goal constraints are derived from grasp and assembly requirements. We formulate layout planning as constrained optimization under sequential feasibility.

---

## Module Overview

```
sealp/
├── config/               ✅ DONE  — YAML project config, robot registry, setup facade
├── colliders/            ✅ DONE  — Obstacle manager, static environment, collision world
├── assembly_sequence/    ✅ DONE  — Data model, YAML I/O, DAG validation, builder
├── editor/               ✅ DONE  — Panda3D-based assembly sequence editor GUI
├── examples/
│   ├── grasp/            ✅ DONE  — Planning, filtering, visualization
│   └── motion/           ✅ DONE  — Single-arm & dual-arm pick-and-place
└── (future modules below)
```

---

## Phase 1: Foundation (✅ Complete)

### 1.1 Assembly Sequence Data Model
- [x] `AssemblyPart` / `AssemblyStep` dataclasses
- [x] `AssemblySequence` container with DAG validation
- [x] YAML / CSV I/O via `sequence_io.py`
- [x] Builder pattern via `SequenceGenerator`
- [x] Strict validation: model path existence check

### 1.2 Project Configuration
- [x] YAML-based `SEALPConfig` with robot type, environment, and sequence file
- [x] Robot registry (piper, cobotta, nova2_wg, xarmlite6_wg) — extensible
- [x] `setup_from_config()` — unified initialization facade

### 1.3 Collision Environment
- [x] `ObstacleManager` — named obstacle dictionary
- [x] `StaticEnvironment` — config-driven obstacle definitions (box, STL)
- [x] `CollisionWorld` — combines static + runtime obstacles

### 1.4 Examples
- [x] Grasp planning, filtering, visualization (`examples/grasp/`)
- [x] Single-arm pick-and-place with RRT (`examples/motion/pnp_demo.py`)
- [x] Dual-arm concurrent pick-and-place (`examples/motion/dual_arm_pnp.py`)
- [x] `PiperPickAndPlace` / `DualPiperPickAndPlace` wrapper classes

### 1.5 Assembly Editor
- [x] Panda3D-based GUI with dark theme
- [x] Part selection, grab (G), rotate (R) with axis constraints
- [x] Load/save YAML sequences
- [x] Console window for logging
- [x] Properties panel with position/rotation editing

---

## Phase 2: Sequential Manipulation (🔧 In Progress)

### 2.1 Motion Primitive Library

Define a compact set of motion primitives for assembly operations.

| Primitive | Description | Arms | Status |
|-----------|-------------|------|--------|
| `transport_place` | Pick from staging, transport, place at assembly pose | Single | ✅ Implemented |
| `dual_transport` | Both arms coordinate to carry a large part | Dual | 🔧 Skeleton |
| `insert` | Linear insertion along a constrained axis | Single | ⬜ TODO |
| `hold_and_insert` | One arm holds, other inserts | Dual | ⬜ TODO |
| `regrasp` | Place on a fixture, regrasp with better grip | Single | ⬜ TODO |

**Files to create:**
- `sealp/primitives/__init__.py`
- `sealp/primitives/transport.py` — single-arm transport-and-place
- `sealp/primitives/dual_transport.py` — dual-arm cooperative transport
- `sealp/primitives/insert.py` — constrained linear insertion
- `sealp/primitives/hold_insert.py` — hold-and-insert coordination
- `sealp/primitives/regrasp.py` — fixture-based regrasp

### 2.2 Sequence Executor

Execute an assembly sequence step-by-step, assigning primitives and dynamically updating the obstacle list as objects are placed.

**Key pattern** (from reference `pick_and_place_chair.py`):
```python
mot_data_list = []
current_conf = start_conf
for step in assembly_sequence.get_execution_order():
    obj = parts[step.part_id]
    obstacle_this_round = [o for o in obs_list if o is not obj]
    mot = planner.gen_pick_and_place(
        obj_cmodel=obj,
        end_jnt_values=current_conf,
        obstacle_list=obstacle_this_round,
        ...
    )
    current_conf = mot.jv_list[-1]
    # Move obj to placed pose, add to obstacles
    placed = obj.copy(); placed.pos = goal_pos; placed.rotmat = goal_rot
    obs_list.append(placed)
    mot_data_list.append(mot)
```

**Files to create:**
- `sealp/executor/__init__.py`
- `sealp/executor/sequence_executor.py` — step-by-step execution with obstacle updates
- `sealp/executor/primitive_selector.py` — auto-assign primitive based on part properties
- `sealp/examples/motion/sequence_execution.py` — demo: execute a multi-step assembly

---

## Phase 3: Layout Optimization (⬜ Planned)

### 3.1 Feasibility Evaluation

For a candidate layout (robot base positions + part staging positions), evaluate whether each assembly step is kinematically feasible.

**Pipeline:**
1. For each step: check IK reachability at pick pose and place pose
2. Check collision-free path exists (fast RRT query)
3. Compute manipulability score at key poses
4. Return a `FeasibilityReport`: per-step pass/fail + aggregate score

**Files to create:**
- `sealp/layout/feasibility.py` — `evaluate_layout(layout, sequence, robot)`
- `sealp/layout/reachability.py` — voxelized reachability map (cached)
- `sealp/layout/manipulability.py` — workspace manipulability scoring

### 3.2 Layout Representation

```python
@dataclass
class WorkspaceLayout:
    robot_base_pos: np.ndarray        # [x, y, z]
    robot_base_rotmat: np.ndarray     # 3x3
    staging_positions: dict           # part_id → (pos, rotmat)
    assembly_station_pos: np.ndarray  # Assembly station location
    fixtures: list                    # Fixed obstacle positions
```

**Files to create:**
- `sealp/layout/layout.py` — `WorkspaceLayout` dataclass + YAML serialization

### 3.3 Layout Optimizer

Formulate layout planning as constrained optimization:
- **Objective**: minimize total execution time + trajectory length
- **Constraints**: feasibility (IK + collision-free paths), workspace bounds, safety margins
- **Method**: feasibility-first search → random/GA/CMA-ES refinement

Approach from reference (`tiaozhanbei/task_sim`):
- Voxelized workspace analysis (`viz_layout_voxel.py`)
- Manipulability maps (`m-map2_refined.py`, `manipulability5.py`)
- GA-based optimizer (`optimizer_ga.py`)
- Random search baseline (`optimizer_random.py`)

**Files to create:**
- `sealp/layout/optimizer.py` — abstract optimizer interface
- `sealp/layout/optimizer_ga.py` — genetic algorithm optimizer
- `sealp/layout/optimizer_random.py` — random search baseline
- `sealp/layout/constraints.py` — layout constraint definitions
- `sealp/layout/objectives.py` — objective function definitions

### 3.4 Evaluation & Metrics

| Metric | Description |
|--------|-------------|
| `total_time` | Sum of motion execution times |
| `total_path_length` | Sum of joint-space path lengths |
| `manipulability_avg` | Average manipulability across steps |
| `feasibility_rate` | Fraction of steps with valid plans |
| `collision_clearance` | Minimum clearance to obstacles |

**Files to create:**
- `sealp/layout/metrics.py` — standardized evaluation metrics

---

## Phase 4: Dual-Arm Coordination (⬜ Planned)

### 4.1 Task Allocation

Given an assembly sequence and dual-arm robot, decide which arm handles which step. Consider:
- Workspace partitioning (left/right regions)
- Part size → single-arm vs. dual-arm primitive
- Load balancing between arms
- Handover requirements

**Files to create:**
- `sealp/coordination/task_allocator.py`

### 4.2 Concurrent Motion Planning

Plan motions for both arms simultaneously with mutual collision avoidance:
- Both arms' swept volumes must not intersect
- Synchronized timing for cooperative primitives (dual_transport)
- Staggered execution for independent operations

**Files to create:**
- `sealp/coordination/dual_arm_planner.py`
- `sealp/coordination/synchronizer.py`

### 4.3 Handover Planning

When a part must transfer between arms:
1. Determine handover configuration (shared grasp region)
2. Plan Arm A → handover pose, Arm B → grasp at handover, Arm A releases

**Files to create:**
- `sealp/coordination/handover.py` — based on `wrs.manipulation.handover_regrasp`

---

## Phase 5: Integration & Validation (⬜ Planned)

### 5.1 End-to-End Pipeline

```
YAML config → load sequence → optimize layout → assign primitives
  → plan all motions → validate → animate / execute
```

**Files to create:**
- `sealp/pipeline.py` — end-to-end `run_assembly(config_path)` function

### 5.2 Benchmark Furniture Sets

| Assembly | Parts | Steps | Dual-arm | Complexity |
|----------|-------|-------|----------|------------|
| Chair (small) | 5 | 4 | No | Low |
| Chair (medium) | 9 | 8 | Mixed | Medium |
| Table | 5 | 4 | Yes | Medium |
| Shelf | 10+ | 10+ | Mixed | High |

**Files to create:**
- `sealp/benchmarks/` — standard assembly definitions + evaluation scripts

### 5.3 Visualization & Reporting

- Full assembly animation in Panda3D
- Per-step feasibility visualization (reachability overlays)
- Layout comparison plots (matplotlib)
- Execution time breakdown charts

---

## Development Priority

| Phase | Focus | Priority | Estimated Effort |
|-------|-------|----------|-----------------|
| 1 | Foundation | ✅ Done | — |
| 2 | Sequential Manipulation | 🔧 Active | 1–2 weeks |
| 3 | Layout Optimization | ⬜ Next | 2–3 weeks |
| 4 | Dual-Arm Coordination | ⬜ Later | 2–3 weeks |
| 5 | Integration & Validation | ⬜ Final | 1–2 weeks |

---

## Environment & Tools

| Item | Value |
|------|-------|
| Python interpreter | `D:\code\venv312\.venv\Scripts\python.exe` |
| PYTHONPATH | `D:\code\layout_sq\wrs` |
| Working directory | `D:\code\layout_sq\wrs` |
| Framework | WRS (Robot Planning & Control) |
| Visualization | Panda3D |
| Primary robot | Piper 6-DoF (single + dual) |
| Config format | YAML (PyYAML) |

### Reference Projects

| Project | Path | What it provides |
|---------|------|-----------------|
| tiaozhanbei | `D:\code\layout_sq\layout-planning-main\myproject\tiaozhanbei` | Grasp planning, task sim, layout optimization reference |
| wrs_tbm | `D:\code\wrs_tbm` | Collision environment patterns, robot API adapter |
| panda3dstudio | `D:\code\layout_sq\panda3dstudio` | Panda3D editor reference |
