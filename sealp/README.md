# SEALP — Sequence-Aware Layout Planner

> **S**equence-**A**ware **L**ayout **P**lanner for dual-arm furniture assembly.

This package provides tools for:
1. **Pick-and-place planning** using the Piper 6-DoF arm
2. **Assembly sequence definition** (YAML-based format with 3D model references)
3. *(Forthcoming)* Layout optimization under sequential feasibility constraints

## Folder Structure

```
sealp/
├── __init__.py
├── README.md
├── pick_and_place/
│   ├── __init__.py
│   ├── piper_pnp.py          # High-level Piper pick-and-place wrapper
│   └── pnp_demo.py           # Runnable demo: Piper arm FK/IK test
└── assembly_sequence/
    ├── __init__.py
    ├── assembly_part.py       # AssemblyPart dataclass
    ├── assembly_step.py       # AssemblyStep dataclass
    ├── assembly_sequence.py   # AssemblySequence container + DAG validation
    ├── sequence_io.py         # YAML / CSV / text I/O
    ├── sequence_generator.py  # Builder-pattern sequence construction
    └── demo_sequence.py       # Runnable demo: build, save, load a sequence
```

## Quick Start

### 1. Piper Pick-and-Place

```python
from sealp.pick_and_place.piper_pnp import PiperPickAndPlace
import wrs.modeling.collision_model as mcm

pnp = PiperPickAndPlace(enable_cc=True)

# Given an object and grasps:
# mot_data = pnp.pick_and_place(obj_cmodel, grasp_collection, goal_pose_list)
```

### 2. Assembly Sequence (YAML)

**Create and save:**

```python
from sealp.assembly_sequence import SequenceGenerator, save_sequence

seq = (
    SequenceGenerator("MyFurniture")
    .add_part("top", "Table Top", "top.stl", assembly_pos=[0, 0, 0.4])
    .add_part("leg", "Leg", "leg.stl", assembly_pos=[0.1, 0, 0])
    .add_step(0, "top", parent="fixture")
    .add_step(1, "leg", parent="top", deps=[0])
    .build()
)

save_sequence(seq, "my_assembly.yaml")
```

**Load and inspect:**

```python
from sealp.assembly_sequence import load_sequence

seq = load_sequence("my_assembly.yaml")
print(seq.summary())

for step in seq.get_execution_order():
    print(f"Step {step.step_id}: assemble {step.part_id}")
```

### 3. Run Demos

```bash
# Piper arm demo (opens Panda3D viewer)
python -m sealp.pick_and_place.pnp_demo

# Assembly sequence demo (terminal output + generates YAML)
python -m sealp.assembly_sequence.demo_sequence
```

## Assembly Sequence YAML Format

```yaml
name: SimpleTable
description: A small table with 2 legs and a cross-bar.
parts:
  - part_id: table_top
    name: Table Top
    model_path: models/table_top.stl
    init_pos: [0.5, 0.0, 0.1]
    assembly_pos: [0.0, 0.0, 0.4]
    assembly_rotmat: [[1,0,0],[0,1,0],[0,0,1]]
    mass: 2.0
    color_rgba: [0.82, 0.71, 0.55, 1.0]
  - part_id: leg_a
    name: Front-Left Leg
    model_path: models/leg.stl
    ...
steps:
  - step_id: 0
    part_id: table_top
    parent_part_id: fixture
    assembly_pos: [0.0, 0.0, 0.4]
    dependencies: []
    primitive_type: single_arm_transport
  - step_id: 1
    part_id: leg_a
    parent_part_id: table_top
    dependencies: [0]
    ...
```

## Dependencies

- [WRS](https://github.com/wanweiwei07/wrs) — Robot planning & control framework
- [PyYAML](https://pypi.org/project/PyYAML/) — `pip install pyyaml`
- Panda3D (bundled with WRS)
