# Yanpu UR7E + DH50 Pick And Place

This folder is a structured replacement for the older monolithic
`yanpu/ur7e_dh50_pickandplace_inside.py` example.

## Main Files

- `config/default.yaml`: rack, object, scene, task, and planner parameters.
- `config/rack_search.yaml`: sampling grid for rack search.
- `main_pickplace.py`: run the full dual-arm pick-and-place example.
- `show_robot.py`: display the robot and environment from YAML.
- `rack_search.py`: evaluate/search rack parameters.

`rack.arm_distance` is the absolute distance between the left and right arm
bases. It is independent of `horizontal_frame.y_length`.

## Modules

- `config.py`: YAML loading and conversion to numpy poses/rotations.
- `robot.py`: build `DualUR7EDH50`; imports TracIK at module load.
- `scene.py`: build environment, payloads, and planning obstacle lists.
- `grasping.py`: load grasp pickle data and compose object/grasp poses.
- `ik.py`: exact IK search and collision filtering.
- `planner.py`: multi-arm RRT, frame generation, payload following.
- `animation.py`: playback helpers.
- `debug.py`: failure visualization helpers.

## Commands

```powershell
python -m yanpu_pnp.show_robot
python -m yanpu_pnp.main_pickplace
python -m yanpu_pnp.rack_search --max-candidates 10
python -m yanpu_pnp.test.visualize_rack_base_pos_samples
python -m yanpu_pnp.test.visualize_arm_mount_samples
python -m yanpu_pnp.test.generate_pickplace_path_from_search --max-candidates 50
python -m yanpu_pnp.test.generate_single_pickplace_path --no-visual
python -m yanpu_pnp.test.search_one_pickplace_rack_result --max-candidates 50
python -m yanpu_pnp.test.search_all_pickplace_rack_results
python -m yanpu_pnp.test.show_rack_search_result
```

Use `--config path\to\config.yaml` on any command to test another rack/object setup.
Use `--search-config path\to\rack_search.yaml` on search/visualization commands
to test another sampling grid.
`search_one_pickplace_rack_result` samples rack candidates from the `grid`
section of `rack_search.yaml`. Its default worker count is 8; use
`--workers N` to override it. Parallel search returns the first feasible
candidate that finishes; use `--workers 1` when candidate index order matters.
When a feasible candidate is found, it is saved to
`yanpu_pnp/results/last_rack_search_result.yaml` by default. Use
`--result-path path\to\result.yaml` to save another file. Use
`show_rack_search_result` to load and display a saved rack result.
`search_all_pickplace_rack_results` scans all requested rack candidates and
writes every feasible candidate to `yanpu_pnp/results/all_rack_search_results.yaml`.
It searches all object pick poses by default; pass `--lft-pick-index` or
`--rgt-pick-index` only when you want to restrict an arm to one pick pose.
Use `show_rack_search_result --result yanpu_pnp/results/all_rack_search_results.yaml`
to display the best saved result, or add `--candidate-index N`.

Colored terminal output uses the optional `colorama` package. Install it with
`pip install colorama`; scripts still run without it, but output falls back to plain text.

Task symmetry can be controlled independently for pick and place:
`pick_symmetry_angle_count`, `place_symmetry_angle_count`, or explicit
`pick_symmetry_angles_deg` / `place_symmetry_angles_deg`.
For rack search, put search-specific symmetry overrides in
`config/rack_search.yaml` under `task_overrides`; these are applied while
evaluating rack candidates.
