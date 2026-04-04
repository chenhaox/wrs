"""
Created on 2025/10/2 
Author: Hao Chen (chen960216@gmail.com)

Piper Arm Definition for WRS Simulator
-------------------------------------

This module defines a six‑degree‑of‑freedom (6‑DoF) model of the AgileX
PiPER robotic arm for use within the WRS simulator.  It closely follows
the structure of the ``Realman`` example supplied by the WRS framework,
but extracts geometric and kinematic parameters from the official PiPER
URDF specification.  Each joint’s local position, orientation,
actuation axis and motion limits are derived from the URDF file
``piper_description_v100_camera.urdf``.  Meshes
for the individual links are loaded from the same package and colours
are assigned to improve visualisation.

The resulting ``Piper`` class inherits from
``wrs.robot_sim.manipulators.manipulator_interface.ManipulatorInterface``
and can be instantiated and attached to a WRS ``World`` just like any
other manipulator.  For numerical inverse kinematics a ``trac_ik`` solver
may optionally be used when the library is available.

Links and joints defined:

1. **arm_base** → **link1** (joint1)
   * Translation: ``(0, 0, 0.123)``
   * Rotation: Euler angles ``(0, 0, −1.5708)``
   * Axis: ``(0,0,1)``
   * Range: ±150° (≈±2.618 rad)

2. **link1** → **link2** (joint2)
   * Translation: ``(0, 0, 0)`` m
   * Rotation: Euler angles ``(1.5708, −0.034907, −1.5708)`` rad
   * Axis: ``(0, 0, 1)``
   * Range: ``[0, π]`` rad

3. **link2** → **link3** (joint3)
   * Translation: ``(0.28358, 0.028726, 0)`` m
   * Rotation: Euler angles ``(0, 0, 0.06604341)`` rad
   * Axis: ``(0, 0, 1)``
   * Range: ``[−2.697, 0]`` rad

4. **link3** → **link4** (joint4)
   * Translation: ``(−0.24221, 0.068514, 0)`` m
   * Rotation: Euler angles ``(−1.5708, 0, 1.3826)`` rad
   * Axis: ``(0, 0, 1)``
   * Range: ±1.832 rad  

5. **link4** → **link5** (joint5)
   * Translation: ``(0, 0, 0)`` m
   * Rotation: Euler angles ``(1.5708, 0, 0)`` rad
   * Axis: ``(0, 0, 1)``
   * Range: ±1.22 rad   

6. **link5** → **link6** (joint6)
   * Translation: ``(0, 0.091, 0.0014165)`` m
   * Rotation: Euler angles ``(−1.5708, −1.5708, 0)`` rad
   * Axis: ``(0, 0, 1)``
   * Range: ±π rad    

Two prismatic joints (joint7 and joint8) and a camera link are defined
in the URDF but are omitted here because the WRS ``ManipulatorInterface``
currently supports only revolute joints.  Nevertheless, the attachment
point for tooling (TCP) can easily be adjusted via ``loc_tcp_pos``
and ``loc_tcp_rotmat``.
"""

import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm

# Attempt to import the Trac IK solver.  If unavailable, numerical IK
# provided by the joint linkage controller (JLC) will be used instead.
try:
    from trac_ik import TracIK

    is_trac_ik = True
    print("Trac IK module loaded successfully")
except Exception as e:
    print(f"Trac IK module not loaded: {e}")
    is_trac_ik = False


class Piper(mi.ManipulatorInterface):
    """Model of the AgileX PiPER 6‑DoF arm for the WRS simulator."""

    def __init__(self, pos: np.ndarray = np.zeros(3), rotmat: np.ndarray = np.eye(3),
                 ik_solver: str = 'd', name: str = 'PiperArm', enable_cc: bool = False):
        """
        Initialise the PiPER arm.

        :param pos: World position of the arm base.
        :param rotmat: World orientation of the arm base.
        :param ik_solver: Either 'd' (default WRS numerical IK) or 'a'/'j'
                          to select alternative solvers.  When the
                          trac_ik library is available the TracIK solver
                          will be used.
        :param name: Identifier for this manipulator.
        :param enable_cc: Enable self‑collision checking if true.
        """
        super().__init__(pos=pos, rotmat=rotmat, home_conf=np.zeros(6),
                         name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)

        # define a uniform colour for all links
        rgba = np.array([0.79216, 0.81961, 0.93333, 1])

        # anchor (arm_base)
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base_link.STL"))
        # no rotation offset for the base link
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0, 0, 0)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rgba

        # --- Joint 1 (arm_base -> link1) ---
        self.jlc.jnts[0].loc_pos = np.array([0.0, 0.0, 0.123])
        # rotate about Z by −90° (−1.5708 rad)
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, -1.5708)
        self.jlc.jnts[0].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[0].motion_range = np.array([-2.618, 2.618])
        # assign the mesh for link1; orient it according to the URDF collision
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link1.STL"))
        self.jlc.jnts[0].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        # link1 in URDF has a rotation of +90° about Z for the collision mesh
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 1.5708)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([0.133, 0.165, 0.224, 1])

        # --- Joint 2 (link1 -> link2) ---
        self.jlc.jnts[1].loc_pos = np.array([0.0, 0.0, 0.0])
        # rotation sequence from URDF: roll=1.5708, pitch=-0.034907, yaw=-1.5708
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.5708, -0.034907, -1.5708)
        self.jlc.jnts[1].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[1].motion_range = np.array([0.0, 3.14])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link2.STL"))
        self.jlc.jnts[1].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        # link2 has a small yaw offset of +0.1 rad for the collision mesh【338922380148493†L96-L100】
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.1)
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([0.776,0.851,0.941, 1])

        # --- Joint 3 (link2 -> link3) ---
        self.jlc.jnts[2].loc_pos = np.array([0.28358, 0.028726, 0.0])
        # a small yaw rotation (≈3.78°)【338922380148493†L141-L147】
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.06604341)
        self.jlc.jnts[2].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[2].motion_range = np.array([-2.697, 0.0])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link3.STL"))
        self.jlc.jnts[2].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        # link3 has a yaw offset of −1.75 rad on the collision mesh【338922380148493†L134-L137】
        self.jlc.jnts[2].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, -1.75)
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([0.329,0.459,0.737, 1])

        # --- Joint 4 (link3 -> link4) ---
        self.jlc.jnts[3].loc_pos = np.array([-0.24221, 0.068514, 0.0])
        # rotation: roll=-90°, yaw≈1.3826 rad  
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(-1.5708, 0.0, 1.3826)
        self.jlc.jnts[3].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[3].motion_range = np.array([-1.832, 1.832])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link4.STL"))
        self.jlc.jnts[3].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        # link4 has no additional rotation
        self.jlc.jnts[3].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, 0.0)
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([0.231,0.329,0.529, 1])

        # --- Joint 5 (link4 -> link5) ---
        self.jlc.jnts[4].loc_pos = np.array([0.0, 0.0, 0.0])
        # rotation: roll=90°   
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(1.5708, 0.0, 0.0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[4].motion_range = np.array([-1.22, 1.22])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "link5.STL"))
        self.jlc.jnts[4].lnk.loc_pos = np.array([0.0, 0.0, 0.0])
        # link5 collision mesh is rotated by −π about Z【338922380148493†L210-L213】
        self.jlc.jnts[4].lnk.loc_rotmat = rm.rotmat_from_euler(0.0, 0.0, -3.14)
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([0.067,0.169,0.341, 1])

        # --- Joint 6 (link5 -> link6) ---
        self.jlc.jnts[5].loc_pos = np.array([0.0, 0.091, 0.0014165])
        # rotation: roll=−90°, pitch=−90°    
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(-1.5708, -1.5708, 0.0)
        self.jlc.jnts[5].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[5].motion_range = np.array([-3.14, 3.14])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "camera_v3.dae"))
        # In the URDF the camera (link6) mesh has an offset
        self.jlc.jnts[5].lnk.loc_pos = np.array([-0.002, -0.008, 0.0])
        self.jlc.jnts[5].lnk.loc_rotmat = rm.rotmat_from_euler(-1.57, 0.0, 0.0)
        self.jlc.jnts[5].lnk.cmodel.rgba = np.array([0.031,0.090,0.196, 1])

        # Finalise the joint linkage chain and set up IK solver
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)

        # Define the tool centre point (TCP) at the end of link6.  By
        # default the TCP coincides with the end of link6 but it can be
        # adjusted by changing loc_tcp_pos and loc_tcp_rotmat after
        # instantiation.
        self.loc_tcp_pos = np.array([0.0, 0.0, 0.0])
        self.loc_tcp_rotmat = np.eye(3)

        # Configure the Trac IK solver when available
        if is_trac_ik:
            directory = os.path.abspath(os.path.dirname(__file__))
            urdf = os.path.join(directory, "piper_description_v100_camera.urdf")
            # base_link and link6 are defined in the URDF; use them for IK
            self._ik_solver = TracIK("arm_base", "link6", urdf)
        else:
            self._ik_solver = None

        # Set up collision checking (self‑collision) if requested
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self) -> None:
        """Configure pairs of links for self‑collision checking."""
        # Add each link to the collision checker and establish
        # conservative collision pairs (into and from lists).  The
        # selections below are similar to those used in the Realman
        # example.
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])  # base
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        from_list = [l3, l4, l5]
        into_list = [lb, l0, l1, ]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def ik(self, tgt_pos: np.ndarray, tgt_rotmat: np.ndarray,
           seed_jnt_values=None, option: str = "empty", toggle_dbg: bool = False):
        """
        Solve the inverse kinematics for the end‑effector.

        When the Trac IK library is available this method delegates to
        ``TracIK.ik``; otherwise it falls back on the numerical IK
        implementation provided by the JLC.  An optional seed
        configuration may be provided to bias the solution.

        :param tgt_pos: Desired TCP position (3×1 array).
        :param tgt_rotmat: Desired TCP rotation matrix (3×3).
        :param seed_jnt_values: Optional initial joint values (6×1 array).
        :param option: Additional option string (unused here).
        :param toggle_dbg: If true prints debugging information.
        :return: Joint configuration achieving the target pose or None
        """
        # Transform target pose into the wrist coordinate frame
        tgt_rotmat = tgt_rotmat @ self.loc_tcp_rotmat.T
        tgt_pos = tgt_pos - tgt_rotmat @ self.loc_tcp_pos

        if is_trac_ik and self._ik_solver is not None:
            # convert to the base_link frame for Trac IK
            anchor_inv_homomat = np.linalg.inv(rm.homomat_from_posrot(
                self.jlc.anchor.pos, self.jlc.anchor.rotmat))
            tgt_homomat = anchor_inv_homomat.dot(rm.homomat_from_posrot(tgt_pos, tgt_rotmat))
            tgt_pos, tgt_rotmat = tgt_homomat[:3, 3], tgt_homomat[:3, :3]
            seed_jnt_values = self.home_conf if seed_jnt_values is None else seed_jnt_values.copy()
            return self._ik_solver.ik(tgt_pos, tgt_rotmat, seed_jnt_values=seed_jnt_values)
        else:
            # fall back to numerical IK provided by the JLC
            return self.jlc.ik(tgt_pos=tgt_pos,
                               tgt_rotmat=tgt_rotmat,
                               seed_jnt_values=seed_jnt_values,
                               toggle_dbg=toggle_dbg)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    # Visual test for the Piper arm model.
    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    arm = Piper(enable_cc=True)
    # arm.fix_to(pos=np.array([0.1, 0.024, 0.3133]),
    #            rotmat=rm.rotmat_from_euler(0.3, 0.8, 1.0))
    arm.fk(arm.rand_conf(), update=True)
    arm.gen_meshmodel().attach_to(base)
    # arm.show_cdprim()
    base.run()
    # generate random joint configurations and test FK/IK consistency
    # for _ in range(10):
    #     rand_conf = arm.rand_conf()
    #     pos, rotmat = arm.fk(rand_conf, update=True)
    #     jnt = arm.ik(pos, rotmat)
    #     print(jnt)
