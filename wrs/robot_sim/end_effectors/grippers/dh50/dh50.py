import os

import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi


class Dh50(gpi.GripperInterface):
    """DH50 two-finger parallel gripper updated to the current WRS API."""

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT,
                 name="dh50"):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        coupling_offset_pos = np.asarray(coupling_offset_pos, dtype=float)
        coupling_offset_rotmat = np.asarray(coupling_offset_rotmat, dtype=float)
        self.coupling.loc_flange_pose_list[0] = (coupling_offset_pos, coupling_offset_rotmat)
        if np.linalg.norm(coupling_offset_pos) > 1e-9:
            self.coupling.lnk_list[0].cmodel = mcm.gen_stick(spos=np.zeros(3),
                                                             epos=coupling_offset_pos,
                                                             type="rect",
                                                             radius=.035,
                                                             rgb=np.array([.2, .2, .2]),
                                                             alpha=1,
                                                             n_sec=24)
        self.jaw_range = np.array([0.0, .05])
        self.palm = rkjlc.rkjl.Anchor(name=name + "_palm",
                                      pos=self.coupling.gl_flange_pose_list[0][0],
                                      rotmat=self.coupling.gl_flange_pose_list[0][1])
        self.palm.lnk_list[0].name = name + "_base"
        self.palm.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0, 0, np.pi / 2)
        self.palm.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "base.STL"),
            name=name + "_base",
            cdmesh_type=self.cdmesh_type,
            cdprim_type=mcm.const.CDPrimType.AABB,
            ex_radius=.002)
        self.palm.lnk_list[0].cmodel.rgba = np.array([.2, .2, .2, 1])
        self.lft_jlc = rkjlc.JLChain(pos=self.palm.pos, rotmat=self.palm.rotmat, n_dof=1, name=name + "_lft")
        self.lft_jlc.jnts[0].change_type(rkjlc.const.JntType.PRISMATIC,
                                          motion_range=np.array([0.0, self.jaw_range[1] / 2.0]))
        self.lft_jlc.jnts[0].loc_pos = np.array([.00683, -.01315, .1065])
        self.lft_jlc.jnts[0].loc_motion_ax = np.array([1.0, 0.0, 0.0])
        self.lft_jlc.jnts[0].lnk.name = name + "_lft_finger"
        self.lft_jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "lf.STL"),
            name=name + "_lft_finger",
            cdmesh_type=self.cdmesh_type,
            cdprim_type=mcm.const.CDPrimType.AABB,
            ex_radius=.002)
        self.lft_jlc.jnts[0].lnk.cmodel.rgba = np.array([.5, .5, .5, 1])
        self.rgt_jlc = rkjlc.JLChain(pos=self.palm.pos, rotmat=self.palm.rotmat, n_dof=1, name=name + "_rgt")
        self.rgt_jlc.jnts[0].change_type(rkjlc.const.JntType.PRISMATIC,
                                          motion_range=np.array([0.0, self.jaw_range[1] / 2.0]))
        self.rgt_jlc.jnts[0].loc_pos = np.array([-.00683, .01315, .1065])
        self.rgt_jlc.jnts[0].loc_motion_ax = np.array([-1.0, 0.0, 0.0])
        self.rgt_jlc.jnts[0].lnk.name = name + "_rgt_finger"
        self.rgt_jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, np.pi)
        self.rgt_jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "rg.STL"),
            name=name + "_rgt_finger",
            cdmesh_type=self.cdmesh_type,
            cdprim_type=mcm.const.CDPrimType.AABB,
            ex_radius=.002)
        self.rgt_jlc.jnts[0].lnk.cmodel.rgba = np.array([.5, .5, .5, 1])
        self.lft_jlc.finalize()
        self.rgt_jlc.finalize()
        self.loc_acting_center_pos = coupling_offset_pos + coupling_offset_rotmat @ np.array([0, 0, .139])
        self.loc_acting_center_rotmat = coupling_offset_rotmat
        self.cdelements = (self.palm.lnk_list[0],
                           self.lft_jlc.jnts[0].lnk,
                           self.rgt_jlc.jnts[0].lnk)

    def fix_to(self, pos, rotmat, jaw_width=None):
        self._pos = np.asarray(pos, dtype=float)
        self._rotmat = np.asarray(rotmat, dtype=float)
        if jaw_width is not None:
            self.change_jaw_width(jaw_width=jaw_width)
        self.coupling.pos = self._pos
        self.coupling.rotmat = self._rotmat
        self.palm.pos = self.coupling.gl_flange_pose_list[0][0]
        self.palm.rotmat = self.coupling.gl_flange_pose_list[0][1]
        self.lft_jlc.fix_to(self.palm.pos, self.palm.rotmat)
        self.rgt_jlc.fix_to(self.palm.pos, self.palm.rotmat)
        self.update_oiee()

    def get_jaw_width(self):
        return self.lft_jlc.jnts[0].motion_value * 2.0

    @gpi.ei.EEInterface.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        side_jaw_width = jaw_width / 2.0
        if self.jaw_range[0] / 2.0 <= side_jaw_width <= self.jaw_range[1] / 2.0:
            self.jaw_width = jaw_width
            self.lft_jlc.goto_given_conf(jnt_values=np.array([side_jaw_width]))
            self.rgt_jlc.goto_given_conf(jnt_values=np.array([side_jaw_width]))
        else:
            raise ValueError(f"The jaw_width parameter is out of range: {jaw_width}")

    def jaw_to(self, jaw_width):
        self.change_jaw_width(jaw_width=jaw_width)

    def open(self):
        self.change_jaw_width(self.jaw_range[1])

    def close(self):
        self.change_jaw_width(self.jaw_range[0])

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False):
        m_col = mmc.ModelCollection(name=self.name + "_stickmodel")
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.palm.gen_stickmodel(toggle_root_frame=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        self.lft_jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                    toggle_flange_frame=False).attach_to(m_col)
        self.rgt_jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                    toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name + "_meshmodel")
        self.coupling.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_flange_frame=False,
                                    toggle_root_frame=False,
                                    toggle_cdmesh=toggle_cdmesh,
                                    toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.palm.gen_meshmodel(rgb=rgb,
                                alpha=alpha,
                                toggle_root_frame=toggle_jnt_frames,
                                toggle_flange_frame=False,
                                toggle_cdmesh=toggle_cdmesh,
                                toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.lft_jlc.gen_meshmodel(rgb=rgb,
                                   alpha=alpha,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_flange_frame=False,
                                   toggle_cdmesh=toggle_cdmesh,
                                   toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.rgt_jlc.gen_meshmodel(rgb=rgb,
                                   alpha=alpha,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_flange_frame=False,
                                   toggle_cdmesh=toggle_cdmesh,
                                   toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        self._gen_oiee_meshmodel(m_col, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh)
        return m_col


if __name__ == "__main__":
    from wrs import wd, mgm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    gripper = Dh50()
    gripper.change_jaw_width(.05)
    gripper.gen_meshmodel(toggle_tcp_frame=True, toggle_cdprim=True).attach_to(base)
    base.run()
