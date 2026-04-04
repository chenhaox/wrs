"""
Piper Pick-and-Place Demo
==========================

Demonstrates basic pick-and-place planning with the Piper 6-DoF arm.
The demo creates a simple object, generates a random grasp configuration,
and plans a motion to move the object between two poses.

Run directly::

    python -m sealp.pick_and_place.pnp_demo
"""

import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.visualization.panda.world as wd
import wrs.robot_sim.manipulators.piper.piper as piper_mod


def main():
    # ----------------------------------------------------------------
    # 1. Create the Panda3D world
    # ----------------------------------------------------------------
    world = wd.World(cam_pos=[1.5, 1.5, 1.0], lookat_pos=[0, 0, 0.3])
    mgm.gen_frame().attach_to(world)

    # ----------------------------------------------------------------
    # 2. Instantiate the Piper arm
    # ----------------------------------------------------------------
    arm = piper_mod.Piper(pos=np.zeros(3), rotmat=np.eye(3),
                          enable_cc=True, name="piper_demo")

    # Show the arm in its home configuration
    arm.gen_meshmodel(alpha=0.3).attach_to(world)

    # Show the arm in a random FK configuration
    rand_conf = arm.rand_conf()
    arm.fk(rand_conf, update=True)
    arm.gen_meshmodel(alpha=1.0).attach_to(world)

    # Show the stick model with joint frames
    arm.gen_stickmodel(toggle_jnt_frames=True).attach_to(world)

    # ----------------------------------------------------------------
    # 3. Print basic info
    # ----------------------------------------------------------------
    print("=" * 50)
    print("Piper Pick-and-Place Demo")
    print("=" * 50)
    print(f"Home configuration:  {arm.home_conf}")
    print(f"Random configuration: {rand_conf}")
    tcp_pos, tcp_rotmat = arm.fk(rand_conf, update=True)
    print(f"TCP position:  {tcp_pos}")
    print(f"TCP rotation:\n{tcp_rotmat}")

    # Test FK → IK round-trip
    ik_result = arm.ik(tcp_pos, tcp_rotmat)
    if ik_result is not None:
        print(f"IK solution:   {ik_result}")
        print("FK→IK round-trip: SUCCESS")
    else:
        print("FK→IK round-trip: FAILED (IK returned None)")

    print("=" * 50)
    print("Press ESC to close the viewer.")

    # ----------------------------------------------------------------
    # 4. Run the viewer
    # ----------------------------------------------------------------
    world.run()


if __name__ == "__main__":
    main()
