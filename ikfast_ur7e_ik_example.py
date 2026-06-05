import numpy as np

import pyikfast
import wrs.basis.robot_math as rm
from wrs.robot_sim.manipulators.ur7e.ur7e import UR7E


np.set_printoptions(precision=9, suppress=True)


def rot_error(src_rot, tgt_rot):
    return np.linalg.norm(rm.delta_w_between_rotmat(src_rot, tgt_rot))


def py_fk(q):
    pos, rot = pyikfast.forward(np.asarray(q).tolist())
    return np.asarray(pos), np.asarray(rot).reshape(3, 3)


def joint_error(q, target_q):
    delta = (np.asarray(q) - np.asarray(target_q) + np.pi) % (2.0 * np.pi) - np.pi
    return np.linalg.norm(delta)


def main():
    arm = UR7E(ik_solver="n")

    rng = np.random.default_rng(0)
    source_q = rng.uniform(-1.2, 1.2, size=6)
    target_pos, target_rot = arm.fk(source_q)

    py_solutions = [
        np.asarray(q)
        for q in pyikfast.inverse(target_pos.tolist(), target_rot.reshape(-1).tolist())
    ]
    wrs_solution = arm.ik(target_pos, target_rot, seed_jnt_values=source_q)

    print("random source q:")
    print(source_q)
    print()
    print("target:")
    print(target_pos)
    print(target_rot)
    print()

    print("pyikfast IK solutions:")
    print(len(py_solutions))
    for i, q in enumerate(py_solutions):
        pos, rot = py_fk(q)
        print(f"{i}: q={q}")
        print(f"   joint error to source q={joint_error(q, source_q)}")
        print(f"   FK pos error={np.linalg.norm(pos - target_pos)}")
        print(f"   FK rot error={rot_error(rot, target_rot)}")
    print()

    print("WRS UR7E IK solution:")
    print(wrs_solution)
    if wrs_solution is not None:
        pos, rot = arm.fk(wrs_solution)
        print(f"joint error to source q={joint_error(wrs_solution, source_q)}")
        print(f"FK pos error={np.linalg.norm(pos - target_pos)}")
        print(f"FK rot error={rot_error(rot, target_rot)}")


if __name__ == "__main__":
    main()
