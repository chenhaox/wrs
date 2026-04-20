---
description: How to run Python scripts and tests in this project
---

# Running Python in this Project

## Environment
- **Python interpreter**: `D:\code\venv312\.venv\Scripts\python.exe`
- **PYTHONPATH**: `D:\code\layout_sq\wrs`
- **Working directory**: `D:\code\layout_sq\wrs`

## Run a script
// turbo-all

1. Set environment and run:
```powershell
$env:PYTHONPATH="D:\code\layout_sq\wrs"; D:\code\venv312\.venv\Scripts\python.exe <script_path>
```

## Run a module
```powershell
$env:PYTHONPATH="D:\code\layout_sq\wrs"; D:\code\venv312\.venv\Scripts\python.exe -m <module.path>
```

## Quick import check
```powershell
$env:PYTHONPATH="D:\code\layout_sq\wrs"; D:\code\venv312\.venv\Scripts\python.exe -c "import <module>; print('OK')"
```

## Common modules to run

| What | Command |
|------|---------|
| Assembly editor | `python -m sealp.editor.run_editor` |
| Grasp planning demo | `python -m sealp.examples.grasp.planning` |
| Grasp filtering demo | `python -m sealp.examples.grasp.filtering` |
| Grasp visualization | `python -m sealp.examples.grasp.visualization` |
| Single-arm PnP demo | `python -m sealp.examples.motion.pnp_demo` |
| Dual-arm PnP demo | `python -m sealp.examples.motion.dual_arm_pnp` |
| Layout evaluation (3D) | `python -m sealp.examples.layout.eval_layout` |
| Layout comparison (charts) | `python -m sealp.examples.layout.compare_layouts` |
| Layout optimization (3D) | `python -m sealp.examples.layout.optimize_layout` |
| Config demo | `python -m sealp.config.demo_config` |
| Assembly sequence demo | `python -m sealp.assembly_sequence.demo_sequence` |
