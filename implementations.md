Furniture assembly is a long-horizon sequential manipulation task in which a prescribed assembly sequence must be executed under tight geometric constraints and limited workspace. A valid sequence alone, however, does not ensure executability: workspace layout—including robot base placement, part staging, and assembly-station configuration—fundamentally determines whether each step admits feasible dual-arm motion and how efficiently the sequence can be completed. We study sequence-aware workspace layout optimization for dual-arm furniture assembly. Execution is represented with a compact library of motion primitives, including single-arm transport-and-place for small parts and dual-arm cooperative transport for large parts. Given an assembly sequence, each part is assigned a primitive, and step-wise start/goal constraints are derived from grasp and assembly requirements. We formulate layout planning as constrained optimization under sequential feasibility, seeking layouts that (i) maximize space utilization, (ii) minimize total execution time and trajectory length, and (iii) satisfy safety and collision constraints. To make search tractable, we introduce a feasibility-first evaluation pipeline that quickly filters candidate layouts and refines promising ones with motion planning. Experiments on representative furniture assemblies show that the proposed method consistently finds executable layouts and reduces execution time compared with sequence-agnostic baselines, enabling reliable dual-arm execution of long-horizon assembly sequences.

# WRS Implementation Ideas

> Brainstormed implementation projects based on the WRS Robot Planning & Control System.
> Each section is a self-contained project idea with motivation, scope, and suggested approach.
> Pick any that interests you, expand it, or combine multiple ideas.

---

## Table of Contents

1. [Layout-Aware Pick-and-Place Sequencer](#1-layout-aware-pick-and-place-sequencer)
2. [Workspace Reachability Heatmap Visualizer](#2-workspace-reachability-heatmap-visualizer)
3. [Multi-Robot Collaborative Assembly Planner](#3-multi-robot-collaborative-assembly-planner)
4. [Grasp Transfer Learning Pipeline](#4-grasp-transfer-learning-pipeline)
5. [Online Regrasp Planning with Sensor Feedback](#5-online-regrasp-planning-with-sensor-feedback)
6. [Constrained Motion Planning with Task-Space Regions](#6-constrained-motion-planning-with-task-space-regions)
7. [Digital Twin Synchronization Framework](#7-digital-twin-synchronization-framework)
8. [Bin-Picking with Point Cloud Segmentation](#8-bin-picking-with-point-cloud-segmentation)
9. [Null-Space Optimization for Secondary Objectives](#9-null-space-optimization-for-secondary-objectives)
10. [Benchmarking Suite for Motion Planners](#10-benchmarking-suite-for-motion-planners)

---

## 1. Layout-Aware Pick-and-Place Sequencer

### Motivation
The current `PickPlacePlanner` handles individual pick-and-place operations. In real
manufacturing/logistics, you need to sequence many objects (e.g., bin packing, pallet loading,
kit assembly) while respecting spatial constraints and optimizing total cycle time.

### Scope
- **Input**: A set of objects with initial poses, a set of goal poses/slots, and environment obstacles.
- **Output**: An optimal (or near-optimal) ordering of pick-and-place operations with collision-free
  motion plans.
- **Key Challenges**:
  - Object-to-slot assignment optimization (combinatorial).
  - Order-dependent obstacle changes (placed objects become obstacles for subsequent operations).
  - Cycle time minimization vs. feasibility.

### Suggested Approach
1. Build on `wrs.manipulation.pick_place.PickPlacePlanner`.
2. Use a greedy heuristic or TSP-style solver for sequencing.
3. Incrementally update the obstacle list as objects are placed.
4. Evaluate with the Panda3D visualizer — animate the full sequence.

### Files to Create/Modify
- `wrs/manipulation/pick_place_sequencer.py` [NEW]
- `0000_examples/sequencer_demo.py` [NEW]

---

## 2. Workspace Reachability Heatmap Visualizer

### Motivation
Understanding a robot's reachable workspace (considering IK feasibility, collision, and
manipulability) is critical for workcell design. Currently, users must manually sample poses
and test IK. A systematic voxelized reachability map would be extremely useful.

### Scope
- **Input**: Robot model, environment obstacles, voxel resolution.
- **Output**: A 3D color-coded point cloud (heatmap) showing reachability and manipulability
  at each point, rendered in Panda3D.
- **Metrics per voxel**:
  - Binary reachability (at least one IK solution exists).
  - Manipulability index (Yoshikawa's measure).
  - Number of feasible orientations.

### Suggested Approach
1. Discretize the workspace into a 3D grid.
2. For each voxel, sample N orientations and solve IK.
3. Compute manipulability via `robot.manipulability_val()`.
4. Color-code (red → low, green → high) and render as a point cloud.
5. Support caching results to disk for reuse.

### Files to Create/Modify
- `wrs/robot_sim/_kinematics/reachability_map.py` [NEW]
- `0000_examples/reachability_heatmap_demo.py` [NEW]

---

## 3. Multi-Robot Collaborative Assembly Planner

### Motivation
The system already supports dual-arm robots (UR3 dual, UR3e dual, YuMi, xArm7 dual, Diana7 dual).
A collaborative assembly planner would coordinate both arms to assemble parts — e.g., one arm
holds a workpiece while the other inserts a component.

### Scope
- **Input**: Assembly sequence graph (which parts mate with which), dual-arm robot, part models.
- **Output**: Coordinated motion plans for both arms, including:
  - Handover sequences between arms.
  - Synchronized approach/depart motions.
  - Collision avoidance between the two arms.

### Suggested Approach
1. Build on `wrs.manipulation.handover_regrasp` and `dual_arm_robot_interface`.
2. Define an `AssemblyTask` data structure: list of (part_A, part_B, mating_transform).
3. For each assembly step, plan: which arm picks which part, handover if needed, insertion motion.
4. Use the existing RRT-Connect planner for collision-free paths between arms.

### Files to Create/Modify
- `wrs/manipulation/assembly_planner.py` [NEW]
- `wrs/manipulation/assembly_task.py` [NEW]
- `0000_examples/collaborative_assembly_demo.py` [NEW]

---

## 4. Grasp Transfer Learning Pipeline

### Motivation
The current grasp planning pipeline (`grasping/planning/antipodal.py`) computes grasps per
object geometry. When a new but similar object appears, all grasps must be recomputed. A grasp
transfer system would map known good grasps from a source object to a similar target object.

### Scope
- **Input**: Source object with annotated grasps, target object mesh.
- **Output**: Transferred grasp candidates on the target object, ranked by quality.
- **Key Ideas**:
  - Shape correspondence via ICP or feature matching.
  - Grasp quality re-evaluation on the target (force closure check).
  - Filter invalid transfers (collision, kinematic infeasibility).

### Suggested Approach
1. Use Open3D for point cloud registration between source and target meshes.
2. Transform source grasp poses using the estimated correspondence.
3. Re-evaluate each transferred grasp using existing `GraspReasoner`.
4. Benchmark against from-scratch grasp planning speed.

### Files to Create/Modify
- `wrs/grasping/planning/grasp_transfer.py` [NEW]
- `0000_examples/grasp_transfer_demo.py` [NEW]

---

## 5. Online Regrasp Planning with Sensor Feedback

### Motivation
The existing `flatsurface_regrasp` and `handover_regrasp` are offline planners — they build
full regrasp graphs and search for solutions before execution. In practice, execution errors
(slip, misalignment) require replanning from the current (possibly unexpected) state.

### Scope
- **Input**: Current robot state (joint values), current object pose (from vision), goal pose.
- **Output**: A replanned regrasp sequence from the current state to the goal.
- **Key Features**:
  - Fast incremental replanning (reuse partial regrasp graphs).
  - Interface with vision (`wrs/vision/`) to detect current object pose.
  - Graceful recovery from execution failures.

### Suggested Approach
1. Cache the regrasp graph structure (placement stability, grasp feasibility).
2. On failure detection, recompute only the affected subgraph.
3. Use `GraspReasoner.reason_incremental_common_gids` for fast incremental reasoning.
4. Integrate with AR markers or depth cameras for pose estimation.

### Files to Create/Modify
- `wrs/manipulation/online_regrasp.py` [NEW]
- `wrs/manipulation/regrasp_graph.py` [NEW] — refactor regrasp graph to be reusable.
- `0000_examples/online_regrasp_demo.py` [NEW]

---

## 6. Constrained Motion Planning with Task-Space Regions

### Motivation
Many real tasks require the robot to maintain constraints during motion — e.g., keeping a glass
of water upright, maintaining visual contact with a workpiece, or sliding along a surface. The
current RRT planners work in joint space without explicit task-space constraints.

### Scope
- **Input**: Start/goal configurations, task-space constraint definition (position bounds,
  orientation bounds, or an arbitrary constraint function).
- **Output**: A collision-free, constraint-satisfying trajectory.
- **Constraint Types**:
  - Orientation constraints (keep end-effector upright).
  - Position constraints (stay on a surface/plane).
  - Custom user-defined constraints.

### Suggested Approach
1. Extend `wrs/motion/probabilistic/rrt_connect.py` with a projection-based constraint sampler.
2. After each RRT extension, project the new node onto the constraint manifold using IK.
3. Implement common constraint types as built-in classes.
4. Demo: pour a virtual liquid (keep upright constraint) while navigating obstacles.

### Files to Create/Modify
- `wrs/motion/probabilistic/constrained_rrt_connect.py` [NEW]
- `wrs/motion/constraints.py` [NEW]
- `0000_examples/constrained_motion_demo.py` [NEW]

---

## 7. Digital Twin Synchronization Framework

### Motivation
The system has both simulation (`robot_sim`) and real robot control (`robot_con`, `drivers`).
A digital twin framework would keep the simulation state synchronized with the real robot in
real-time — useful for monitoring, predictive collision checking, and AR overlay.

### Scope
- **Input**: Real robot connection (via existing drivers), simulation robot model.
- **Output**: Live-updating 3D visualization that mirrors the real robot's state.
- **Features**:
  - Real-time joint state polling and FK update.
  - Predictive: simulate a commanded trajectory before executing on real hardware.
  - Record and playback session logs.

### Suggested Approach
1. Create a synchronization layer between `robot_con` and `robot_sim`.
2. Use a threaded update loop: poll real joints → update sim FK → refresh visualization.
3. Add a "predict" mode: plan and visualize motion before committing to real execution.
4. Build on the Panda3D `taskMgr` loop for real-time rendering.

### Files to Create/Modify
- `wrs/robot_sim/digital_twin.py` [NEW]
- `0000_examples/digital_twin_demo.py` [NEW]

---

## 8. Bin-Picking with Point Cloud Segmentation

### Motivation
Bin-picking is a fundamental industrial task. The system already has depth camera support
(`vision/depth_camera`), collision models, and grasp planning. Connecting these into an
end-to-end bin-picking pipeline would be a high-value demo.

### Scope
- **Input**: Depth image of a bin with randomly piled objects, known object CAD model.
- **Output**: Planned pick motions for each detected object.
- **Pipeline**:
  1. Capture depth image → point cloud.
  2. Segment individual objects (RANSAC plane removal + clustering).
  3. Pose estimation (ICP alignment with CAD model).
  4. Grasp planning using existing `antipodal` planner.
  5. Motion planning using existing approach-depart planner.

### Suggested Approach
1. Use Open3D for point cloud processing.
2. Implement a simple segmentation pipeline (background subtraction + DBSCAN clustering).
3. Use ICP to estimate 6DOF pose of each segment.
4. Feed estimated poses into `PickPlacePlanner`.
5. Visualize the full pipeline in Panda3D.

### Files to Create/Modify
- `wrs/vision/bin_picking/segmentation.py` [NEW]
- `wrs/vision/bin_picking/pose_estimation.py` [NEW]
- `wrs/vision/bin_picking/pipeline.py` [NEW]
- `0000_examples/bin_picking_demo.py` [NEW]

---

## 9. Null-Space Optimization for Secondary Objectives

### Motivation
For redundant robots (7+ DOF, like xArm7, Franka, Diana7), the IK has infinite solutions.
The null-space can be exploited to optimize secondary objectives while achieving the primary
task. Examples exist (`0000_examples/cobotta_nullspace.py`, `xs_nullspace.py`) but there's
no systematic framework.

### Scope
- **Input**: Primary task (target TCP pose), secondary objective function (e.g., maximize
  manipulability, minimize joint torques, avoid obstacles).
- **Output**: Joint configuration that satisfies the primary task while optimizing the
  secondary objective.
- **Secondary Objectives**:
  - Maximize manipulability.
  - Stay close to a preferred joint configuration.
  - Maximize distance to joint limits.
  - Maximize distance to obstacles.

### Suggested Approach
1. Build on `wrs/robot_sim/_kinematics/ik_num.py` (numerical IK).
2. Use gradient projection in the null-space: `q_dot = J_pinv * x_dot + (I - J_pinv*J) * q0_dot`.
3. Implement pluggable secondary objective functions.
4. Demo: have a 7-DOF arm track a Cartesian path while dynamically optimizing posture.

### Files to Create/Modify
- `wrs/robot_sim/_kinematics/nullspace_opt.py` [NEW]
- `0000_examples/nullspace_optimization_demo.py` [NEW]

---

## 10. Benchmarking Suite for Motion Planners

### Motivation
The system has multiple RRT variants (RRT, RRT-Connect, RRT*, kinodynamic, differential wheel).
There's no systematic way to compare their performance across standardized scenarios. A
benchmarking suite would help users pick the right planner for their application.

### Scope
- **Input**: A set of benchmark scenarios (start/goal pairs, obstacle environments).
- **Output**: Performance metrics per planner — planning time, path length, smoothness,
  success rate, number of collision checks.
- **Benchmark Scenarios**:
  - Narrow passage (gap between two walls).
  - Cluttered environment (many small obstacles).
  - Long-range navigation (start and goal are far apart).
  - Constrained workspace (robot near joint limits).

### Suggested Approach
1. Define scenario configs in YAML/JSON.
2. Create a `BenchmarkRunner` that loads scenarios, runs each planner N times, collects metrics.
3. Generate comparison plots (matplotlib) and tables.
4. Include the existing `comparison_rrt.py` and `comparison_rrt_connect.py` as baselines.

### Files to Create/Modify
- `wrs/bench_mark/motion_planner_benchmark.py` [NEW]
- `wrs/bench_mark/scenarios/` [NEW] — folder with scenario definitions.
- `0000_examples/planner_benchmark_demo.py` [NEW]

---

## Quick-Start Implementation Guide

### Recommended Order (by increasing complexity)

| Priority | Project | Difficulty | Dependencies |
|----------|---------|-----------|--------------|
| ⭐ | #10 Benchmarking Suite | Low | Existing RRT planners |
| ⭐ | #2 Reachability Heatmap | Low-Medium | IK solvers, Panda3D |
| ⭐⭐ | #9 Null-Space Optimization | Medium | Numerical IK, Jacobian |
| ⭐⭐ | #1 Pick-Place Sequencer | Medium | PickPlacePlanner |
| ⭐⭐ | #6 Constrained Motion | Medium-High | RRT-Connect, IK |
| ⭐⭐⭐ | #7 Digital Twin | Medium-High | robot_con, Panda3D |
| ⭐⭐⭐ | #4 Grasp Transfer | Medium-High | Open3D, grasp planning |
| ⭐⭐⭐ | #3 Collaborative Assembly | High | Dual-arm, handover |
| ⭐⭐⭐⭐ | #5 Online Regrasp | High | Regrasp graph, vision |
| ⭐⭐⭐⭐ | #8 Bin-Picking Pipeline | High | Vision, grasp, motion |

### How to Use This Document

1. **Choose a project** that matches your interest and skill level.
2. **Delete or archive** the other sections you don't plan to implement.
3. **Expand your chosen section** with detailed implementation steps, class designs, and test plans.
4. **Create the files** listed in the "Files to Create/Modify" section.
5. **Develop iteratively** — start with a minimal working version and refine.

---

## Notes

- All projects assume the existing WRS package structure and conventions (Panda3D visualization,
  NumPy-based math, collision model patterns).
- Robot models: YuMi and Cobotta are the best-tested robots for simulation demos. UR3e dual
  and xArm7 are good for dual-arm and redundancy projects.
- The `0000_examples/` folder convention is used for demo scripts that exercise the new
  functionality.
