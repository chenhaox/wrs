import os
from typing import Literal

import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jl as rkjl
import wrs.robot_sim._kinematics.collision_checker as cc
import wrs.robot_sim.manipulators.ur7e.ur7e as ur7e_manipulator
import wrs.robot_sim.robots.robot_interface as ri

try:
    from trac_ik import TracIK
    TRACIK_IMPORT_ERROR = None
except ImportError as exc:
    TracIK = None
    TRACIK_IMPORT_ERROR = exc

try:
    import pyikfast
    PYIKFAST_IMPORT_ERROR = None
except ImportError as exc:
    pyikfast = None
    PYIKFAST_IMPORT_ERROR = exc


class StaticFixture:
    """A fixed collision/display link mounted relative to a robot root pose."""

    def __init__(self,
                 mesh_path,
                 name,
                 loc_pos=np.zeros(3),
                 loc_rotmat=np.eye(3),
                 rgba=np.array([.35, .35, .35, 1.0]),
                 cdprim_type=mcm.const.CDPrimType.AABB,
                 ex_radius=.002):
        self.name = name
        self.loc_pos = np.asarray(loc_pos, dtype=float)
        self.loc_rotmat = np.asarray(loc_rotmat, dtype=float)
        self.anchor = rkjl.Anchor(name=name)
        self.anchor.lnk_list[0].name = name
        self.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=mesh_path,
            name=name,
            cdprim_type=cdprim_type,
            ex_radius=ex_radius)
        self.anchor.lnk_list[0].cmodel.rgba = np.asarray(rgba)

    @property
    def lnk(self):
        return self.anchor.lnk_list[0]

    def fix_to(self, root_pos, root_rotmat):
        self.anchor.fix_to(pos=root_pos + root_rotmat @ self.loc_pos,
                           rotmat=root_rotmat @ self.loc_rotmat)

    def gen_meshmodel(self, rgb=None, alpha=None, toggle_cdprim=False, toggle_cdmesh=False):
        return self.anchor.gen_meshmodel(name=self.name + "_meshmodel",
                                         rgb=rgb,
                                         alpha=alpha,
                                         toggle_cdprim=toggle_cdprim,
                                         toggle_cdmesh=toggle_cdmesh)

    def gen_stickmodel(self):
        return self.anchor.gen_stickmodel(name=self.name + "_stickmodel",
                                          toggle_root_frame=False,
                                          toggle_flange_frame=False)


class UR7EBase(ri.RobotInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 name="ur7e",
                 enable_cc=True,
                 arm_home_conf=None,
                 arm_loc_pos=np.zeros(3),
                 arm_loc_rotmat=np.eye(3),
                 fixture_specs=None,
                 hnd_cls=None,
                 hnd_kwargs=None,
                 hnd_loc_rotmat=np.eye(3),
                 ik_solver=None):
        pos = np.asarray(pos, dtype=float)
        rotmat = np.asarray(rotmat, dtype=float)
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.arm_loc_pos = np.asarray(arm_loc_pos, dtype=float)
        self.arm_loc_rotmat = np.asarray(arm_loc_rotmat, dtype=float)
        if arm_home_conf is None:
            arm_home_conf = np.zeros(6)
        self.oih_infos = []
        self.ik_backend_name = None if ik_solver is None else str(ik_solver).strip().lower()
        self._prefer_tracik = self.ik_backend_name in ("t", "tracik", "pytracik")
        self._prefer_ikfast = self.ik_backend_name in ("fast", "ikfast", "pyikfast")
        arm_ik_solver = "n" if self._prefer_tracik or self._prefer_ikfast else ik_solver
        self.iksolver_cache = {}
        self.fixture_list = []
        self.manipulator_dict = {}
        self.hnd_dict = {}
        self.hnd_loc_rotmat = np.asarray(hnd_loc_rotmat, dtype=float)
        current_dir = os.path.dirname(__file__)
        if fixture_specs is not None:
            for spec in fixture_specs:
                mesh_path = spec["mesh_path"]
                if not os.path.isabs(mesh_path):
                    mesh_path = os.path.join(current_dir, "meshes", mesh_path)
                fixture = StaticFixture(
                    mesh_path=mesh_path,
                    name=spec.get("name", os.path.splitext(os.path.basename(mesh_path))[0]),
                    loc_pos=spec.get("loc_pos", np.zeros(3)),
                    loc_rotmat=spec.get("loc_rotmat", np.eye(3)),
                    rgba=spec.get("rgba", np.array([.35, .35, .35, 1.0])),
                    cdprim_type=spec.get("cdprim_type", mcm.const.CDPrimType.AABB),
                    ex_radius=spec.get("ex_radius", .002))
                fixture.fix_to(self.pos, self.rotmat)
                self.fixture_list.append(fixture)
        arm_pos, arm_rotmat = self._compute_arm_root_pose()
        self.arm = ur7e_manipulator.UR7E(pos=arm_pos,
                                         rotmat=arm_rotmat,
                                         home_conf=np.asarray(arm_home_conf, dtype=float),
                                         name=name + "_arm",
                                         enable_cc=False,
                                         ik_solver=arm_ik_solver)
        self.manipulator = self.arm
        self.hnd = None
        if hnd_cls is not None:
            if hnd_kwargs is None:
                hnd_kwargs = {}
            hnd_kwargs = hnd_kwargs.copy()
            hnd_kwargs.setdefault("name", name + "_hnd")
            hnd_pos, hnd_rotmat = self._compute_hnd_root_pose()
            self.hnd = hnd_cls(pos=hnd_pos, rotmat=hnd_rotmat, **hnd_kwargs)
            self.arm.loc_tcp_pos = self.hnd_loc_rotmat @ self.hnd.loc_acting_center_pos
            self.arm.loc_tcp_rotmat = self.hnd_loc_rotmat @ self.hnd.loc_acting_center_rotmat
        self._delegator = self.arm
        self.manipulator_dict["arm"] = self.arm
        self.manipulator_dict["hnd"] = self.arm
        if self.hnd is not None:
            self.hnd_dict["hnd"] = self.hnd
            self.hnd_dict["arm"] = self.hnd
        if self.cc is not None:
            self.setup_cc()

    def _compute_arm_root_pose(self):
        return self.pos + self.rotmat @ self.arm_loc_pos, self.rotmat @ self.arm_loc_rotmat

    def _compute_hnd_root_pose(self):
        return self.arm.gl_flange_pos, self.arm.gl_flange_rotmat @ self.hnd_loc_rotmat

    def update_end_effector(self, ee_values=None):
        if self.hnd is not None:
            if ee_values is not None:
                self.hnd.change_ee_values(ee_values=ee_values)
            hnd_pos, hnd_rotmat = self._compute_hnd_root_pose()
            self.hnd.fix_to(pos=hnd_pos, rotmat=hnd_rotmat)

    @property
    def gl_tcp_pos(self):
        return self.arm.gl_tcp_pos

    @property
    def gl_tcp_rotmat(self):
        return self.arm.gl_tcp_rotmat

    @property
    def home_conf(self):
        return self.arm.home_conf

    @property
    def jnt_ranges(self):
        return self.arm.jnt_ranges

    @property
    def end_effector(self):
        return self.hnd

    @property
    def oiee_list(self):
        if self.hnd is None:
            return []
        return self.hnd.oiee_list

    def setup_cc(self):
        fixture_ids = [self.cc.add_cce(fixture.lnk) for fixture in self.fixture_list]
        ee_cces = []
        if self.hnd is not None:
            for cdlnk in self.hnd.cdelements:
                ee_cces.append(self.cc.add_cce(cdlnk))
        mlb = self.cc.add_cce(self.arm.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.arm.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.arm.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.arm.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.arm.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.arm.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.arm.jlc.jnts[5].lnk)
        self.cc.set_cdpair_by_ids(ee_cces + [ml3, ml4, ml5], [mlb, ml0])
        self.cc.set_cdpair_by_ids([ml3, ml4, ml5], [ml1])
        self.cc.set_cdpair_by_ids([ml5], [ml1, ml2])
        if ee_cces:
            self.cc.set_cdpair_by_ids(ee_cces, [ml1, ml2])
        if fixture_ids:
            self.cc.set_cdpair_by_ids(ee_cces + [ml1, ml2, ml3, ml4, ml5], fixture_ids)
        self.cc.enable_extcd_by_id_list(ee_cces + [ml0, ml1, ml2, ml3, ml4, ml5], type="from")
        self.cc.enable_innercd_by_id_list([mlb, ml0, ml1, ml2], type="into")
        self.cc.dynamic_ext_list = ee_cces[1:]

    def reset_cc(self):
        self.cc = cc.CollisionChecker("collision_checker")
        self.setup_cc()

    def backup_state(self):
        self.arm.backup_state()
        if self.hnd is not None:
            self.hnd.backup_state()

    def restore_state(self):
        self.arm.restore_state()
        self.update_end_effector()
        if self.hnd is not None:
            self.hnd.restore_state()

    def fix_to(self, pos, rotmat):
        self._pos = np.asarray(pos, dtype=float)
        self._rotmat = np.asarray(rotmat, dtype=float)
        for fixture in self.fixture_list:
            fixture.fix_to(self.pos, self.rotmat)
        arm_pos, arm_rotmat = self._compute_arm_root_pose()
        self.arm.fix_to(pos=arm_pos, rotmat=arm_rotmat)
        self.update_end_effector()
        self._update_oih()

    def goto_given_conf(self, jnt_values=None, **kwargs):
        if jnt_values is None:
            jnt_values = kwargs.get("jnt_values", None)
        if jnt_values is None:
            raise ValueError("jnt_values must be provided.")
        result = self.arm.goto_given_conf(jnt_values=np.asarray(jnt_values, dtype=float))
        self.update_end_effector()
        self._update_oih()
        return result

    def goto_home_conf(self):
        return self.goto_given_conf(self.arm.home_conf)

    def fk(self, component_name="arm", jnt_values=np.zeros(6)):
        if component_name not in self.manipulator_dict:
            raise ValueError("The given component name is not supported.")
        return self.goto_given_conf(jnt_values)

    def ik(self, tgt_pos, tgt_rotmat, seed_jnt_values=None, toggle_dbg=False):
        if self._prefer_tracik:
            return self.tracik(tgt_pos=tgt_pos,
                               tgt_rotmat=tgt_rotmat,
                               seed_jnt_values=seed_jnt_values)
        if self._prefer_ikfast:
            conf_list = self.ik_all(tgt_pos=tgt_pos,
                                    tgt_rotmat=tgt_rotmat,
                                    seed_jnt_values=seed_jnt_values,
                                    toggle_dbg=toggle_dbg)
            if not conf_list:
                return None
            return conf_list[0]
        return self.arm.ik(tgt_pos=tgt_pos,
                           tgt_rotmat=tgt_rotmat,
                           seed_jnt_values=seed_jnt_values,
                           toggle_dbg=toggle_dbg)

    def ik_all(self, tgt_pos, tgt_rotmat, seed_jnt_values=None, toggle_dbg=False):
        if self._prefer_ikfast:
            return self.pyikfast(tgt_pos=tgt_pos,
                                 tgt_rotmat=tgt_rotmat,
                                 seed_jnt_values=seed_jnt_values)
        if self._prefer_tracik:
            conf = self.tracik(tgt_pos=tgt_pos,
                               tgt_rotmat=tgt_rotmat,
                               seed_jnt_values=seed_jnt_values)
            return [] if conf is None else [np.asarray(conf, dtype=float)]
        result = self.arm.ik(tgt_pos=tgt_pos,
                             tgt_rotmat=tgt_rotmat,
                             seed_jnt_values=seed_jnt_values,
                             option="multiple",
                             toggle_dbg=toggle_dbg)
        return self._normalize_ik_solution_list(result, seed_jnt_values=seed_jnt_values)

    def get_jnt_values(self, component_name="arm"):
        if component_name not in self.manipulator_dict:
            raise ValueError("The given component name is not supported.")
        return self.arm.get_jnt_values()

    def rand_conf(self, component_name="arm"):
        if component_name not in self.manipulator_dict:
            raise ValueError("The given component name is not supported.")
        return self.arm.rand_conf()

    def are_jnts_in_ranges(self, jnt_values=None, **kwargs):
        if jnt_values is None:
            jnt_values = kwargs.get("jnt_values", None)
        if jnt_values is None:
            raise ValueError("jnt_values must be provided.")
        return self.arm.are_jnts_in_ranges(jnt_values=np.asarray(jnt_values, dtype=float))

    def get_ee_values(self):
        if self.hnd is not None:
            return self.hnd.get_ee_values()
        return None

    def change_ee_values(self, ee_values):
        if self.hnd is not None:
            self.hnd.change_ee_values(ee_values=ee_values)

    def jaw_to(self, hand_name="hnd", jawwidth=0.0):
        if self.hnd is not None:
            self.hnd.change_jaw_width(jaw_width=jawwidth)
        self.jaw_width = jawwidth

    def hndclose(self):
        self.jaw_to(jawwidth=0.0)

    def hndopen(self):
        if self.hnd is not None:
            self.jaw_to(jawwidth=self.hnd.jaw_range[1])
        else:
            self.jaw_to(jawwidth=0.076)

    def hold(self, hnd_name, objcm, jawwidth=None):
        if self.hnd is not None:
            rel_pos, rel_rotmat = self.arm.cvt_gl_pose_to_tcp(objcm.pos, objcm.rotmat)
            oiee = self.hnd.hold(obj_cmodel=objcm, jaw_width=jawwidth)
            if self.cc is not None:
                uuid = self.cc.add_cce(oiee)
                self.cc.enable_extcd_by_id_list(id_list=[uuid], type="from")
                self.cc.enable_innercd_by_id_list(id_list=[uuid], type="from")
                self.cc.dynamic_ext_list.append(uuid)
            return rel_pos, rel_rotmat
        rel_pos, rel_rotmat = self.arm.cvt_gl_pose_to_tcp(objcm.pos, objcm.rotmat)
        self.oih_infos.append({
            "collision_model": objcm,
            "rel_pos": rel_pos,
            "rel_rotmat": rel_rotmat,
            "gl_pos": objcm.pos,
            "gl_rotmat": objcm.rotmat,
        })
        return rel_pos, rel_rotmat

    def release(self, hnd_name, objcm, jawwidth=None):
        if self.hnd is not None:
            oiee = self.hnd.release(obj_cmodel=objcm, jaw_width=jawwidth)
            if oiee is not None and self.cc is not None:
                self.cc.remove_cce(oiee)
            return
        for obj_info in list(self.oih_infos):
            if obj_info["collision_model"] is objcm:
                self.oih_infos.remove(obj_info)
                break

    def _update_oih(self):
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_pose_in_tcp_to_gl(obj_info["rel_pos"], obj_info["rel_rotmat"])
            obj_info["gl_pos"] = gl_pos
            obj_info["gl_rotmat"] = gl_rotmat
            obj_info["collision_model"].pose = (gl_pos, gl_rotmat)

    def get_oih_list(self):
        if self.hnd is not None:
            self.hnd.update_oiee()
            return [oiee.cmodel for oiee in self.hnd.oiee_list]
        self._update_oih()
        return [obj_info["collision_model"] for obj_info in self.oih_infos]

    def get_tgt_pose_in_rbt(self, tgt_pos, tgt_rotmat):
        return rm.rel_pose(self.arm.pos, self.arm.rotmat, tgt_pos, tgt_rotmat)

    def get_tgt_flange_pose_in_arm_base(self, tgt_tcp_pos, tgt_tcp_rotmat):
        tgt_flange_rotmat = tgt_tcp_rotmat @ self.arm.loc_tcp_rotmat.T
        tgt_flange_pos = tgt_tcp_pos - tgt_tcp_rotmat @ self.arm.loc_tcp_pos
        return self.get_tgt_pose_in_rbt(tgt_flange_pos, tgt_flange_rotmat)

    def tracik(self,
               urdf_path: str = os.path.join(os.path.dirname(__file__), "urdf", "ur7e.urdf"),
               base_link_name: str = "base_link",
               tip_link_name: str = "wrist_3_link",
               tgt_pos=np.zeros(3),
               tgt_rotmat=np.eye(3),
               seed_jnt_values=None,
               solver_type: Literal["Speed", "Distance", "Manip1", "Manip2"] = "Distance"):
        if TracIK is None:
            raise ImportError("trac_ik is not installed or failed to load in this environment.") from TRACIK_IMPORT_ERROR
        rel_pos, rel_rotmat = self.get_tgt_flange_pose_in_arm_base(tgt_pos, tgt_rotmat)
        key = (urdf_path, base_link_name, tip_link_name, solver_type)
        if key not in self.iksolver_cache:
            self.iksolver_cache[key] = TracIK(base_link_name=base_link_name,
                                              tip_link_name=tip_link_name,
                                              urdf_path=urdf_path,
                                              solver_type=solver_type,
                                              timeout=.01)
        if seed_jnt_values is None:
            seed_jnt_values = self.arm.home_conf
        return self.iksolver_cache[key].ik(rel_pos, rel_rotmat, seed_jnt_values)

    def pyikfast(self, tgt_pos, tgt_rotmat, seed_jnt_values=None):
        if pyikfast is None:
            raise ImportError("pyikfast is not installed or failed to load in this environment.") from PYIKFAST_IMPORT_ERROR
        rel_pos, rel_rotmat = self.get_tgt_flange_pose_in_arm_base(tgt_pos, tgt_rotmat)
        result = pyikfast.inverse(rel_pos.tolist(), rel_rotmat.reshape(-1).tolist())
        return self._normalize_ik_solution_list(result, seed_jnt_values=seed_jnt_values)

    def _normalize_ik_solution_list(self, result, seed_jnt_values=None, duplicate_tol=1e-5):
        if result is None:
            return []
        if isinstance(result, (int, float, np.integer, np.floating)) and result == 0:
            return []
        try:
            if len(result) == 0:
                return []
        except TypeError:
            return []
        seed = self.arm.home_conf if seed_jnt_values is None else np.asarray(seed_jnt_values, dtype=float)
        raw_array = np.asarray(result, dtype=float)
        if raw_array.ndim == 1:
            if raw_array.size != self.arm.n_dof:
                return []
            raw_list = [raw_array]
        elif raw_array.ndim == 2:
            raw_list = [row for row in raw_array if row.size == self.arm.n_dof]
        else:
            raw_list = []
            for item in result:
                item_array = np.asarray(item, dtype=float)
                if item_array.ndim == 1 and item_array.size == self.arm.n_dof:
                    raw_list.append(item_array)
        conf_list = []
        for raw_conf in raw_list:
            conf = self._nearest_joint_equivalent(raw_conf, seed)
            if not self.are_jnts_in_ranges(conf):
                continue
            if any(np.linalg.norm(conf - existing_conf) <= duplicate_tol for existing_conf in conf_list):
                continue
            conf_list.append(conf)
        conf_list.sort(key=lambda conf: np.linalg.norm(conf - seed))
        return conf_list

    def _nearest_joint_equivalent(self, jnt_values, reference):
        jnt_values = np.asarray(jnt_values, dtype=float).copy()
        reference = np.asarray(reference, dtype=float)
        jnt_ranges = self.arm.jnt_ranges
        for i, value in enumerate(jnt_values):
            candidates = value + 2.0 * np.pi * np.arange(-2, 3)
            in_range = candidates[(candidates >= jnt_ranges[i, 0]) & (candidates <= jnt_ranges[i, 1])]
            if len(in_range) > 0:
                candidates = in_range
            jnt_values[i] = candidates[np.argmin(np.abs(candidates - reference[i]))]
        return jnt_values

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name="ur7e_robot_stickmodel",
                       **kwargs):
        m_col = mmc.ModelCollection(name=name)
        for fixture in self.fixture_list:
            fixture.gen_stickmodel().attach_to(m_col)
        self.arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                toggle_jnt_frames=toggle_jnt_frames,
                                toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        if self.hnd is not None:
            self.hnd.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                    toggle_jnt_frames=toggle_jnt_frames).attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=True,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name="ur7e_robot_meshmodel",
                      **kwargs):
        m_col = mmc.ModelCollection(name=name)
        for fixture in self.fixture_list:
            fixture.gen_meshmodel(rgb=rgb,
                                  alpha=alpha,
                                  toggle_cdprim=toggle_cdprim,
                                  toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        self.arm.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_tcp_frame=toggle_tcp_frame,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_flange_frame=toggle_flange_frame,
                               toggle_cdprim=toggle_cdprim,
                               toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        if self.hnd is not None:
            self.hnd.gen_meshmodel(rgb=rgb,
                                   alpha=alpha,
                                   toggle_tcp_frame=toggle_tcp_frame,
                                   toggle_jnt_frames=toggle_jnt_frames,
                                   toggle_cdprim=toggle_cdprim,
                                   toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col


class UR7EDualBase(ri.RobotInterface):
    """Two UR7E manipulators treated as one robot for whole-body collision/planning."""

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 name="dual_ur7e",
                 enable_cc=True,
                 arm_specs=None,
                 fixture_specs=None,
                 active_arm_name="upper_arm",
                 ik_solver=None):
        pos = np.asarray(pos, dtype=float)
        rotmat = np.asarray(rotmat, dtype=float)
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        if arm_specs is None:
            raise ValueError("arm_specs must define at least one UR7E arm.")
        self.fixture_list = []
        self.arm_list = []
        self.arm_dict = {}
        self.manipulator_dict = {}
        self.hnd_dict = {}
        self.oih_infos = []
        self.active_arm_name = active_arm_name
        current_dir = os.path.dirname(__file__)
        if fixture_specs is not None:
            for spec in fixture_specs:
                mesh_path = spec["mesh_path"]
                if not os.path.isabs(mesh_path):
                    mesh_path = os.path.join(current_dir, "meshes", mesh_path)
                fixture = StaticFixture(
                    mesh_path=mesh_path,
                    name=spec.get("name", os.path.splitext(os.path.basename(mesh_path))[0]),
                    loc_pos=spec.get("loc_pos", np.zeros(3)),
                    loc_rotmat=spec.get("loc_rotmat", np.eye(3)),
                    rgba=spec.get("rgba", np.array([.35, .35, .35, 1.0])),
                    cdprim_type=spec.get("cdprim_type", mcm.const.CDPrimType.AABB),
                    ex_radius=spec.get("ex_radius", .002))
                fixture.fix_to(self.pos, self.rotmat)
                self.fixture_list.append(fixture)
        for i, spec in enumerate(arm_specs):
            arm_name = spec.get("name", f"arm_{i}")
            loc_pos = np.asarray(spec.get("loc_pos", np.zeros(3)), dtype=float)
            loc_rotmat = np.asarray(spec.get("loc_rotmat", np.eye(3)), dtype=float)
            home_conf = np.asarray(spec.get("home_conf", np.zeros(6)), dtype=float)
            arm_pos, arm_rotmat = self._compute_child_pose(loc_pos, loc_rotmat)
            arm = ur7e_manipulator.UR7E(pos=arm_pos,
                                        rotmat=arm_rotmat,
                                        home_conf=home_conf,
                                        name=name + "_" + arm_name,
                                        enable_cc=False,
                                        ik_solver=ik_solver)
            arm.loc_pos = loc_pos
            arm.loc_rotmat = loc_rotmat
            self.arm_list.append(arm)
            self.arm_dict[arm_name] = arm
            self.manipulator_dict[arm_name] = arm
            setattr(self, arm_name, arm)
        self.arm = self.arm_dict.get(self.active_arm_name, self.arm_list[0])
        self.manipulator = self.arm
        self.hnd = None
        self._delegator = None
        if "arm" not in self.manipulator_dict:
            self.manipulator_dict["arm"] = self.arm
        if self.cc is not None:
            self.setup_cc()

    @property
    def n_dof(self):
        return sum(arm.n_dof for arm in self.arm_list)

    @property
    def home_conf(self):
        return np.concatenate([arm.home_conf for arm in self.arm_list])

    @property
    def jnt_ranges(self):
        return np.vstack([arm.jnt_ranges for arm in self.arm_list])

    @property
    def gl_tcp_pos(self):
        return self.arm.gl_tcp_pos

    @property
    def gl_tcp_rotmat(self):
        return self.arm.gl_tcp_rotmat

    @property
    def end_effector(self):
        return None

    @property
    def oiee_list(self):
        return self.get_oih_list()

    def _compute_child_pose(self, loc_pos, loc_rotmat):
        return self.pos + self.rotmat @ loc_pos, self.rotmat @ loc_rotmat

    def _get_arm(self, component_name=None):
        if component_name is None or component_name == "arm":
            return self.arm
        if component_name not in self.arm_dict:
            raise ValueError("The given component name is not supported.")
        return self.arm_dict[component_name]

    def _split_conf(self, jnt_values):
        jnt_values = np.asarray(jnt_values, dtype=float)
        if len(jnt_values) != self.n_dof:
            raise ValueError("The given joint values do not match total n_dof.")
        conf_list = []
        start = 0
        for arm in self.arm_list:
            end = start + arm.n_dof
            conf_list.append(jnt_values[start:end])
            start = end
        return conf_list

    @staticmethod
    def _add_arm_cce(collision_checker, arm):
        mlb = collision_checker.add_cce(arm.jlc.anchor.lnk_list[0])
        ml0 = collision_checker.add_cce(arm.jlc.jnts[0].lnk)
        ml1 = collision_checker.add_cce(arm.jlc.jnts[1].lnk)
        ml2 = collision_checker.add_cce(arm.jlc.jnts[2].lnk)
        ml3 = collision_checker.add_cce(arm.jlc.jnts[3].lnk)
        ml4 = collision_checker.add_cce(arm.jlc.jnts[4].lnk)
        ml5 = collision_checker.add_cce(arm.jlc.jnts[5].lnk)
        return {
            "base": mlb,
            "all": [mlb, ml0, ml1, ml2, ml3, ml4, ml5],
            "links": [ml0, ml1, ml2, ml3, ml4, ml5],
            "moving": [ml1, ml2, ml3, ml4, ml5],
            "distal": [ml3, ml4, ml5],
            "wrist": [ml5],
            "proximal": [mlb, ml0],
            "mid": [ml1, ml2],
        }

    def setup_cc(self):
        fixture_ids = [self.cc.add_cce(fixture.lnk) for fixture in self.fixture_list]
        arm_cce_list = [self._add_arm_cce(self.cc, arm) for arm in self.arm_list]
        for arm_cce in arm_cce_list:
            self.cc.set_cdpair_by_ids(arm_cce["distal"], arm_cce["proximal"])
            self.cc.set_cdpair_by_ids(arm_cce["wrist"], arm_cce["mid"])
            if fixture_ids:
                self.cc.set_cdpair_by_ids(arm_cce["moving"], fixture_ids)
            self.cc.enable_extcd_by_id_list(arm_cce["links"], type="from")
            self.cc.enable_innercd_by_id_list(arm_cce["proximal"] + arm_cce["mid"], type="into")
        for i, arm_cce_from in enumerate(arm_cce_list):
            for arm_cce_into in arm_cce_list[:i]:
                self.cc.set_cdpair_by_ids(arm_cce_from["moving"], arm_cce_into["moving"])

    def reset_cc(self):
        self.cc = cc.CollisionChecker("collision_checker")
        self.setup_cc()

    def backup_state(self):
        for arm in self.arm_list:
            arm.backup_state()

    def restore_state(self):
        for arm in reversed(self.arm_list):
            arm.restore_state()

    def fix_to(self, pos, rotmat):
        self._pos = np.asarray(pos, dtype=float)
        self._rotmat = np.asarray(rotmat, dtype=float)
        for fixture in self.fixture_list:
            fixture.fix_to(self.pos, self.rotmat)
        for arm in self.arm_list:
            arm_pos, arm_rotmat = self._compute_child_pose(arm.loc_pos, arm.loc_rotmat)
            arm.fix_to(pos=arm_pos, rotmat=arm_rotmat, jnt_values=arm.get_jnt_values())
        self._update_oih()

    def goto_given_conf(self, jnt_values):
        for arm, conf in zip(self.arm_list, self._split_conf(jnt_values)):
            arm.goto_given_conf(jnt_values=conf)
        self._update_oih()

    def goto_home_conf(self):
        return self.goto_given_conf(self.home_conf)

    def fk(self, component_name="both", jnt_values=None):
        if jnt_values is None and not isinstance(component_name, str):
            jnt_values = component_name
            component_name = "both"
        if component_name == "both":
            return self.goto_given_conf(jnt_values)
        return self._get_arm(component_name).goto_given_conf(jnt_values=np.asarray(jnt_values, dtype=float))

    def ik(self, tgt_pos, tgt_rotmat, component_name=None, seed_jnt_values=None, toggle_dbg=False):
        return self._get_arm(component_name).ik(tgt_pos=tgt_pos,
                                                tgt_rotmat=tgt_rotmat,
                                                seed_jnt_values=seed_jnt_values,
                                                toggle_dbg=toggle_dbg)

    def get_jnt_values(self, component_name="both"):
        if component_name == "both":
            return np.concatenate([arm.get_jnt_values() for arm in self.arm_list])
        return self._get_arm(component_name).get_jnt_values()

    def rand_conf(self, component_name="both"):
        if component_name == "both":
            return np.concatenate([arm.rand_conf() for arm in self.arm_list])
        return self._get_arm(component_name).rand_conf()

    def are_jnts_in_ranges(self, jnt_values):
        return all(arm.are_jnts_in_ranges(conf) for arm, conf in zip(self.arm_list, self._split_conf(jnt_values)))

    def get_ee_values(self):
        return None

    def change_ee_values(self, ee_values):
        pass

    def jaw_to(self, hand_name="hnd", jawwidth=0.0):
        self.jaw_width = jawwidth

    def hndclose(self):
        self.jaw_to(jawwidth=0.0)

    def hndopen(self):
        self.jaw_to(jawwidth=0.076)

    def hold(self, hnd_name, objcm, jawwidth=None, component_name=None):
        arm = self._get_arm(component_name)
        rel_pos, rel_rotmat = arm.cvt_gl_pose_to_tcp(objcm.pos, objcm.rotmat)
        self.oih_infos.append({
            "collision_model": objcm,
            "arm_name": component_name or self.active_arm_name,
            "rel_pos": rel_pos,
            "rel_rotmat": rel_rotmat,
            "gl_pos": objcm.pos,
            "gl_rotmat": objcm.rotmat,
        })
        return rel_pos, rel_rotmat

    def release(self, hnd_name, objcm, jawwidth=None):
        for obj_info in list(self.oih_infos):
            if obj_info["collision_model"] is objcm:
                self.oih_infos.remove(obj_info)
                break

    def _update_oih(self):
        for obj_info in self.oih_infos:
            arm = self._get_arm(obj_info["arm_name"])
            gl_pos, gl_rotmat = arm.cvt_pose_in_tcp_to_gl(obj_info["rel_pos"], obj_info["rel_rotmat"])
            obj_info["gl_pos"] = gl_pos
            obj_info["gl_rotmat"] = gl_rotmat
            obj_info["collision_model"].pose = (gl_pos, gl_rotmat)

    def get_oih_list(self):
        self._update_oih()
        return [obj_info["collision_model"] for obj_info in self.oih_infos]

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name="dual_ur7e_robot_stickmodel",
                       **kwargs):
        m_col = mmc.ModelCollection(name=name)
        for fixture in self.fixture_list:
            fixture.gen_stickmodel().attach_to(m_col)
        for arm in self.arm_list:
            arm.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=True,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name="dual_ur7e_robot_meshmodel",
                      **kwargs):
        m_col = mmc.ModelCollection(name=name)
        for fixture in self.fixture_list:
            fixture.gen_meshmodel(rgb=rgb,
                                  alpha=alpha,
                                  toggle_cdprim=toggle_cdprim,
                                  toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        for arm in self.arm_list:
            arm.gen_meshmodel(rgb=rgb,
                              alpha=alpha,
                              toggle_tcp_frame=toggle_tcp_frame,
                              toggle_jnt_frames=toggle_jnt_frames,
                              toggle_flange_frame=toggle_flange_frame,
                              toggle_cdprim=toggle_cdprim,
                              toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col
