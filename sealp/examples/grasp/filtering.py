"""
Grasp Filtering Example
========================

Demonstrates filtering a ``GraspCollection`` by geometric criteria:
- Gripper orientation (e.g., keep only top-down grasps)
- Grasp position bounds (e.g., height range)
- Jaw width range

This is the second step in a typical grasp pipeline:
    1. Plan grasps  (``grasp_planning.py``)
    2. **Filter grasps** (this script)
    3. Use filtered grasps in pick-and-place planning

Usage::

    python -m sealp.examples.grasp.filtering

Adapted from tiaozhanbei/grasp/filter_grasp.py
"""

import os
from typing import Callable, List, Optional

import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.robot_sim.end_effectors.grippers.piper_gripper.piper_gripper as pg
from wrs.grasping.grasp import GraspCollection


# ======================================================================
# Filter functions
# ======================================================================
def filter_by_orientation(grasp_collection,
                          axis_idx=2,
                          direction="down",
                          threshold=0.0):
    """Keep grasps whose gripper axis points in a given direction.

    Parameters
    ----------
    grasp_collection : GraspCollection
        Input grasps.
    axis_idx : int
        Column index of ``ac_rotmat`` to check (0=x, 1=y, 2=z).
    direction : str
        ``"down"`` keeps grasps where the axis z-component < -threshold.
        ``"up"`` keeps grasps where the axis z-component > threshold.
    threshold : float
        Cutoff value for the z-component.

    Returns
    -------
    GraspCollection
        Filtered grasps.
    """
    filtered = GraspCollection()
    for grasp in grasp_collection:
        z_component = grasp.ac_rotmat[2, axis_idx]
        if direction == "down" and z_component < -threshold:
            filtered.append(grasp)
        elif direction == "up" and z_component > threshold:
            filtered.append(grasp)
    return filtered


def filter_by_position(grasp_collection,
                       x_range=None,
                       y_range=None,
                       z_range=None):
    """Keep grasps whose ``ac_pos`` falls within specified coordinate ranges.

    Parameters
    ----------
    grasp_collection : GraspCollection
    x_range, y_range, z_range : tuple of (min, max) or None
        If None, no filtering on that axis.

    Returns
    -------
    GraspCollection
    """
    filtered = GraspCollection()
    for grasp in grasp_collection:
        p = grasp.ac_pos
        if x_range is not None and not (x_range[0] <= p[0] <= x_range[1]):
            continue
        if y_range is not None and not (y_range[0] <= p[1] <= y_range[1]):
            continue
        if z_range is not None and not (z_range[0] <= p[2] <= z_range[1]):
            continue
        filtered.append(grasp)
    return filtered


def filter_by_jaw_width(grasp_collection,
                        min_width=None,
                        max_width=None):
    """Keep grasps within a jaw-width range.

    Parameters
    ----------
    grasp_collection : GraspCollection
    min_width, max_width : float or None

    Returns
    -------
    GraspCollection
    """
    filtered = GraspCollection()
    for grasp in grasp_collection:
        w = grasp.ee_values
        if min_width is not None and w < min_width:
            continue
        if max_width is not None and w > max_width:
            continue
        filtered.append(grasp)
    return filtered


def filter_custom(grasp_collection, predicate: Callable):
    """Keep grasps that satisfy an arbitrary predicate function.

    Parameters
    ----------
    grasp_collection : GraspCollection
    predicate : callable
        ``predicate(grasp) -> bool``.  Returns True to keep.

    Returns
    -------
    GraspCollection
    """
    filtered = GraspCollection()
    for grasp in grasp_collection:
        if predicate(grasp):
            filtered.append(grasp)
    return filtered


# ======================================================================
# Demo
# ======================================================================
def main():
    """Run the grasp filtering demo."""
    from sealp.examples.grasp.planning import plan_grasps, visualize_grasps

    # ------------------------------------------------------------------
    # 1. Setup
    # ------------------------------------------------------------------
    base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0))
    mgm.gen_frame(ax_length=0.3).attach_to(base)

    # ------------------------------------------------------------------
    # 2. Create object and plan grasps (or load from disk)
    # ------------------------------------------------------------------
    obj_cmodel = mcm.gen_box(xyz_lengths=np.array([0.06, 0.04, 0.03]),
                             pos=np.array([0, 0, 0.015]))
    obj_cmodel.rgba = np.array([0.6, 0.5, 0.4, 1.0])

    out_dir = os.path.join(os.path.dirname(__file__), "_output")
    pickle_path = os.path.join(out_dir, "demo_box_grasps.pickle")

    if os.path.isfile(pickle_path):
        print(f"Loading grasps from {pickle_path}...")
        grasp_collection = GraspCollection.load_from_disk(
            file_name=pickle_path)
        gripper = pg.PiperGripper()
    else:
        print("Planning grasps (no cached file found)...")
        grasp_collection, gripper = plan_grasps(obj_cmodel, max_samples=50)
        os.makedirs(out_dir, exist_ok=True)
        grasp_collection.save_to_disk(file_name=pickle_path)

    print(f"Total grasps: {len(grasp_collection)}")

    # ------------------------------------------------------------------
    # 3. Apply filters
    # ------------------------------------------------------------------
    # Keep only top-down grasps (gripper z-axis points down)
    filtered = filter_by_orientation(grasp_collection, axis_idx=2,
                                     direction="down", threshold=0.0)
    print(f"After orientation filter (z-down): {len(filtered)}")

    # Keep grasps within a height range
    filtered = filter_by_position(filtered, z_range=(0.005, 0.05))
    print(f"After position filter (z in [0.005, 0.05]): {len(filtered)}")

    # ------------------------------------------------------------------
    # 4. Save filtered grasps
    # ------------------------------------------------------------------
    filtered_path = os.path.join(out_dir, "demo_box_filtered_grasps.pickle")
    filtered.save_to_disk(file_name=filtered_path)
    print(f"Saved {len(filtered)} filtered grasps to {filtered_path}")

    # ------------------------------------------------------------------
    # 5. Visualize filtered grasps (green) on top of original (red faint)
    # ------------------------------------------------------------------
    obj_cmodel.attach_to(base)

    # Original grasps — faint red
    for i, grasp in enumerate(grasp_collection):
        if i >= 20:
            break
        gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat,
                                grasp.ee_values)
        gripper.gen_meshmodel(alpha=0.15).attach_to(base)

    # Filtered grasps — solid
    for i, grasp in enumerate(filtered):
        if i >= 20:
            break
        gripper.grip_at_by_pose(grasp.ac_pos, grasp.ac_rotmat,
                                grasp.ee_values)
        gripper.gen_meshmodel(alpha=0.8).attach_to(base)

    print(f"\nShowing original (faint) vs filtered (solid).")
    print("Press ESC to close.")
    base.run()


if __name__ == "__main__":
    main()
