# SEALP Implementation Details

> Internal documentation for the Sequence-Aware Layout Planner project.

## Project Overview

**SEALP** (Sequence-Aware Layout Planner) is a framework for sequence-aware
workspace layout optimization for dual-arm furniture assembly.

### Core Modules

| Module | Description |
|--------|------------|
| `pick_and_place/` | Pick-and-place planning using the Piper 6-DoF arm |
| `assembly_sequence/` | YAML-based assembly sequence data format & I/O |
| `editor/` | Interactive Panda3D-based assembly sequence editor |

---

## 1. Piper Robot Migration

The AgileX PiPER 6-DoF arm was migrated from `wrs-main` to this project:

- **Source:** `d:\code\HTW\wrs-main\wrs\robot_sim\manipulators\piper\`
- **Destination:** `d:\code\layout_sq\wrs\wrs\robot_sim\manipulators\piper\`
- **Files:** `piper.py`, `piper_description_v100_camera.urdf`, `meshes/`

The Piper class inherits from `ManipulatorInterface` and supports:
- 6 revolute joints with limits from the URDF
- STL collision meshes for each link
- Optional TracIK solver (falls back to numerical IK)

---

## 2. Pick-and-Place Module (`pick_and_place/`)

### `piper_pnp.py` — PiperPickAndPlace

High-level wrapper around `wrs.manipulation.pick_place.PickPlacePlanner`:
- Instantiates a Piper arm with collision checking
- Provides `pick_and_place()` and `pick_and_moveto()` convenience methods
- Includes `animate()` static method for Panda3D visualization

### `pnp_demo.py`

Smoke test that:
1. Creates a Panda3D world
2. Shows the Piper arm in home + random FK configuration
3. Tests FK→IK round-trip

---

## 3. Assembly Sequence Format (`assembly_sequence/`)

### Data Model

```
AssemblySequence
├── name, description
├── parts: Dict[str, AssemblyPart]
│   └── part_id, name, model_path, init_pos, init_rotmat,
│       assembly_pos, assembly_rotmat, mass, color_rgba, metadata
└── steps: List[AssemblyStep]
    └── step_id, part_id, parent_part_id, assembly_pos,
        assembly_rotmat, dependencies, primitive_type, grasp_id, notes
```

### YAML Format

Parts and steps are serialized to/from YAML via PyYAML. Rotation matrices
are stored as nested lists. Positions as flat lists.

### Validation

`AssemblySequence.validate()` checks:
1. Every step references a known part_id
2. Every dependency step_id exists
3. The dependency graph is acyclic (DAG via Kahn's algorithm)
4. No part_id appears in more than one step

### Builder Pattern

`SequenceGenerator` provides a fluent API:
```python
seq = (SequenceGenerator("name")
       .add_part("id", "Name", "file.stl", assembly_pos=[...])
       .add_step(0, "id", parent="fixture")
       .build())
```

---

## 4. Assembly Sequence Editor (`editor/`)

### Architecture

```
assembly_editor.py     ←  Main application (extends WRS World)
    ├── editor_gui.py      ←  DirectGUI widget factories
    ├── transform_handler.py ← Grab/Rotate state machine
    ├── part_manager.py    ←  3D scene part management
    └── run_editor.py      ←  Entry point
```

### Key Bindings

| Key | Action |
|-----|--------|
| `G` | Enter **grab/position** mode |
| `R` | Enter **rotate** mode |
| `X` / `Y` / `Z` | Constrain to axis (during grab/rotate) |
| `Escape` | Cancel transform / deselect |
| `Delete` | Remove selected part |
| `Ctrl+S` | Save YAML |
| `Ctrl+O` | Load YAML |
| Left click | Confirm transform / select part |

### GUI Layout

- **Right sidebar:** Parts list (scrollable, selectable), Assembly steps list,
  Load/Save/New buttons
- **Left sidebar:** Properties panel — position (X,Y,Z), rotation (Rx,Ry,Rz),
  part info, Apply button
- **Status bar:** Current mode, selected part, active step

### Transform System

`TransformHandler` implements a state machine:
- `NONE → GRAB` (via G key): mouse motion → XY translation (or Z with Z key)
- `NONE → ROTATE` (via R key): mouse motion → rotation around axis
- `confirm()` → apply and return to NONE
- `cancel()` → revert to snapshot pose

Axis constraints use Rodrigues' rotation formula for arbitrary axis rotation.

### Part Management

`PartManager` handles:
- Loading STL/OBJ into `CollisionModel` instances
- Auto-assigning distinct colors from a tab10-like palette
- Selection highlighting (yellow)
- Assembly ghost visualization (semi-transparent copy at target pose)
- Pose synchronization back to `AssemblySequence`

### File I/O

Uses `tkinter.filedialog` for native OS file picker dialogs.
Integrates with `sealp.assembly_sequence.sequence_io` for YAML serialization.

---

## 5. Dependencies

| Package | Purpose |
|---------|---------|
| WRS | Robot planning & control framework (Panda3D, numpy) |
| PyYAML | Assembly sequence YAML serialization |
| tkinter | File dialogs in the editor (stdlib) |

---

## 6. Entry Points

```bash
# Piper arm demo
python -m sealp.pick_and_place.pnp_demo

# Assembly sequence demo (terminal, generates YAML)
python -m sealp.assembly_sequence.demo_sequence

# Assembly editor (GUI)
python -m sealp.editor.run_editor
python -m sealp.editor.run_editor path/to/assembly.yaml
```
