import numpy as np

from trac_ik import TracIK as _TracIK

from wrs.robot_sim.robots.ur7e.dual_ur7e_dh50 import DualUR7EDH50

from yanpu_pnp import config as cfgutils
from yanpu_pnp import ik_backends


def build_robot(cfg, enable_cc=True, ik_backend=None):
    backend = ik_backends.resolve_backend(base_cfg=cfg, override=ik_backend)
    robot = DualUR7EDH50(enable_cc=enable_cc,
                         ik_solver=ik_backends.robot_solver_name(backend),
                         **cfgutils.rack_kwargs(cfg))
    robot.lft_arm.goto_given_conf(np.asarray(cfg["rack"]["home_conf"]["lft_arm"], dtype=float))
    robot.rgt_arm.goto_given_conf(np.asarray(cfg["rack"]["home_conf"]["rgt_arm"], dtype=float))
    robot.lft_arm.hndopen()
    robot.rgt_arm.hndopen()
    return robot


def print_ik_status(robot):
    print(f"TracIK top-level import: {bool(_TracIK)} ({_TracIK.__module__}.{_TracIK.__name__}).")
    for arm_name, arm in robot.arm_dict.items():
        cache_size = len(getattr(arm, "iksolver_cache", {}))
        print(f"{arm.name} ({arm_name}): backend={getattr(arm, 'ik_backend_name', None)}, "
              f"prefer_tracik={getattr(arm, '_prefer_tracik', False)}, "
              f"prefer_ikfast={getattr(arm, '_prefer_ikfast', False)}, "
              f"solver_cache={cache_size} before first IK.")


def print_tracik_status(robot):
    print_ik_status(robot)
