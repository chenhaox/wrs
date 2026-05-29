import os

import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.manipulators.manipulator_interface as mi


class UR7E(mi.ManipulatorInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 home_conf=None,
                 name='ur7e',
                 enable_cc=False,
                 ik_solver=None,
                 homeconf=None):
        if homeconf is not None:
            home_conf = homeconf
        if home_conf is None:
            home_conf = np.zeros(6)
        home_conf = np.asarray(home_conf, dtype=float)
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        self._setup_chain(current_file_dir=current_file_dir)
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)
        self.loc_tcp_pos = np.zeros(3)
        self.loc_tcp_rotmat = np.eye(3)
        if self.cc is not None:
            self.setup_cc()

    @staticmethod
    def _make_cmodel(current_file_dir, mesh_name, name, rgba):
        cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", mesh_name),
            name=name,
            cdprim_type=mcm.const.CDPrimType.SURFACE_BALLS,
            ex_radius=.01)
        cmodel.rgba = np.array(rgba)
        return cmodel

    def _setup_chain(self, current_file_dir):
        # anchor/base
        self.jlc.anchor.lnk_list[0].name = "base"
        self.jlc.anchor.lnk_list[0].mass = 2.0
        self.jlc.anchor.lnk_list[0].cmodel = self._make_cmodel(
            current_file_dir, "base.dae", "ur7e_base", [.5, .5, .5, .3])

        # first joint and shoulder link. This real-forpos variant keeps the
        # extra base-frame rotation used by the original file.
        self.jlc.jnts[0].loc_pos = np.array([0, 0, 0.163])
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(.0, 0, np.pi)
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-np.pi * 2, np.pi * 2])
        self.jlc.jnts[0].lnk.name = "shoulder"
        self.jlc.jnts[0].lnk.com = np.array([.0, -.02, .0])
        self.jlc.jnts[0].lnk.mass = 1.95
        self.jlc.jnts[0].lnk.cmodel = self._make_cmodel(
            current_file_dir, "shoulder.dae", "ur7e_shoulder", [.1, .3, .5, .3])

        # second joint and upper arm link
        self.jlc.jnts[1].loc_pos = np.array([0, 0.138, 0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(.0, np.pi / 2.0, .0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[1].motion_range = np.array([-np.pi * 2, np.pi * 2])
        self.jlc.jnts[1].lnk.name = "upperarm"
        self.jlc.jnts[1].lnk.com = np.array([.13, 0, .1157])
        self.jlc.jnts[1].lnk.mass = 3.42
        self.jlc.jnts[1].lnk.cmodel = self._make_cmodel(
            current_file_dir, "upperarm.dae", "ur7e_upperarm", [.7, .7, .7, .3])

        # third joint and forearm link
        self.jlc.jnts[2].loc_pos = np.array([0, -.131, .425])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[2].motion_range = np.array([-np.pi * 2, np.pi * 2])
        self.jlc.jnts[2].lnk.name = "forearm"
        self.jlc.jnts[2].lnk.com = np.array([.05, .0, .0238])
        self.jlc.jnts[2].lnk.mass = 1.437
        self.jlc.jnts[2].lnk.cmodel = self._make_cmodel(
            current_file_dir, "forearm.dae", "ur7e_forearm", [.35, .35, .35, .3])

        # fourth joint and wrist1 link
        self.jlc.jnts[3].loc_pos = np.array([.0, .0, 0.392])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(.0, np.pi / 2.0, 0)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[3].motion_range = np.array([-np.pi * 2, np.pi * 2])
        self.jlc.jnts[3].lnk.name = "wrist1"
        self.jlc.jnts[3].lnk.com = np.array([.0, .0, 0.01])
        self.jlc.jnts[3].lnk.mass = 0.871
        self.jlc.jnts[3].lnk.cmodel = self._make_cmodel(
            current_file_dir, "wrist1.dae", "ur7e_wrist1", [.7, .7, .7, .3])

        # fifth joint and wrist2 link
        self.jlc.jnts[4].loc_pos = np.array([0, .127, 0])
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-np.pi * 2, np.pi * 2])
        self.jlc.jnts[4].lnk.name = "wrist2"
        self.jlc.jnts[4].lnk.com = np.array([.0, .0, 0.01])
        self.jlc.jnts[4].lnk.mass = 0.8
        self.jlc.jnts[4].lnk.cmodel = self._make_cmodel(
            current_file_dir, "wrist2.dae", "ur7e_wrist2", [.1, .3, .5, .3])

        # sixth joint and wrist3 link
        self.jlc.jnts[5].loc_pos = np.array([0, 0, .100])
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[5].motion_range = np.array([-np.pi * 2, np.pi * 2])
        self.jlc.jnts[5].lnk.name = "wrist3"
        self.jlc.jnts[5].lnk.com = np.array([.0, .0, -0.02])
        self.jlc.jnts[5].lnk.mass = 0.8
        self.jlc.jnts[5].lnk.cmodel = self._make_cmodel(
            current_file_dir, "wrist3.dae", "ur7e_wrist3", [.5, .5, .5, .3])

        self.jlc.set_flange(loc_flange_pos=np.array([0, .100, 0]),
                            loc_flange_rotmat=rm.rotmat_from_euler(-np.pi / 2.0, 0, 0))

    def setup_cc(self):
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        self.cc.set_cdpair_by_ids([l3, l4, l5], [lb, l0])
        self.cc.set_cdpair_by_ids([l5], [l1, l2])


if __name__ == '__main__':
    from wrs import wd, mgm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    arm = UR7E(enable_cc=True)
    arm.goto_given_conf(np.array([np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi, 0, 0]))
    arm.gen_meshmodel(toggle_flange_frame=True, toggle_jnt_frames=True, alpha=.7).attach_to(base)
    arm.show_cdprim()
    print("TCP:", arm.gl_tcp_pos, arm.gl_tcp_rotmat)
    print("Collided:", arm.is_collided())
    base.run()
