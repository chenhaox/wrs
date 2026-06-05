import numpy as np

import pyikfast
import wrs.basis.robot_math as rm
from wrs.robot_sim.manipulators.ur7e.ur7e import UR7E
from wrs.robot_sim.manipulators.ur7e.ur7e_real_forpos import UR7E as UR7ERealForpos


TARGET_TRANSLATION = np.array([0.5, 0.5, 0.5], dtype=float)
TARGET_ROTATION = np.eye(3)
RZ_PI = np.diag([-1.0, -1.0, 1.0])


def rotmat_from_flat(values):
    return np.asarray(values, dtype=float).reshape(3, 3)


def rot_err_rad(src, tgt):
    return float(np.linalg.norm(rm.delta_w_between_rotmat(src, tgt)))


def joint_delta_norm(lhs, rhs):
    delta = (np.asarray(lhs) - np.asarray(rhs) + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.linalg.norm(delta))


def unique_joint_solutions(solutions, atol=1e-5):
    unique = []
    for solution in solutions:
        solution = (np.asarray(solution, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
        if not any(joint_delta_norm(solution, existing) < atol for existing in unique):
            unique.append(solution)
    return unique


def collect_wrs_ik_solutions(arm, tgt_pos, tgt_rotmat, extra_seeds=()):
    seeds = [np.zeros(6), *extra_seeds]
    for i in range(100):
        seeds.append(np.random.default_rng(i).uniform(-np.pi, np.pi, size=6))

    solutions = []
    for seed in seeds:
        solution = arm.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, seed_jnt_values=seed)
        if solution is not None:
            solutions.append(solution)
    return unique_joint_solutions(solutions)


def print_solution_summary(title, solutions, arm, tgt_pos, tgt_rotmat):
    print(title)
    print("-" * len(title))
    if not solutions:
        print("no solutions")
        print()
        return
    for i, solution in enumerate(solutions):
        pos, rotmat = arm.fk(solution)
        print(f"{i:02d} q_rad={np.array2string(solution, precision=9)}")
        print(f"   q_deg={np.array2string(np.degrees(solution), precision=6)}")
        print(f"   pos_err_m={np.linalg.norm(pos - tgt_pos):.9e}")
        print(f"   rot_err_rad={rot_err_rad(rotmat, tgt_rotmat):.9e}")
    print()


def main():
    np.set_printoptions(precision=9, suppress=False)

    arm = UR7E(ik_solver="n")
    real_forpos_arm = UR7ERealForpos(ik_solver=None)

    py_solutions = [
        np.asarray(solution, dtype=float)
        for solution in pyikfast.inverse(TARGET_TRANSLATION.tolist(), TARGET_ROTATION.reshape(-1).tolist())
    ]

    print("Target used by ikfast_tst.py")
    print("---------------------------")
    print(f"translation={TARGET_TRANSLATION}")
    print(f"rotation=\n{TARGET_ROTATION}")
    print()

    print("pyikfast inverse solutions forwarded through pyikfast and WRS UR7E")
    print("------------------------------------------------------------------")
    print(f"pyikfast solution count: {len(py_solutions)}")
    for i, solution in enumerate(py_solutions):
        py_pos, py_rot_flat = pyikfast.forward(solution.tolist())
        py_pos = np.asarray(py_pos, dtype=float)
        py_rotmat = rotmat_from_flat(py_rot_flat)
        wrs_pos, wrs_rotmat = arm.fk(solution)
        real_pos, real_rotmat = real_forpos_arm.fk(solution)

        print(f"{i:02d} q_rad={np.array2string(solution, precision=9)}")
        print(f"   q_deg={np.array2string(np.degrees(solution), precision=6)}")
        print(f"   py FK target pos_err_m={np.linalg.norm(py_pos - TARGET_TRANSLATION):.9e}")
        print(f"   py FK target rot_err_rad={rot_err_rad(py_rotmat, TARGET_ROTATION):.9e}")
        print(f"   active UR7E target pos_err_m={np.linalg.norm(wrs_pos - TARGET_TRANSLATION):.9e}")
        print(f"   active UR7E target rot_err_rad={rot_err_rad(wrs_rotmat, TARGET_ROTATION):.9e}")
        print(f"   active UR7E vs Rz(pi)*py FK pos_err_m={np.linalg.norm(wrs_pos - RZ_PI @ py_pos):.9e}")
        print(f"   active UR7E vs Rz(pi)*py FK rot_err_rad={rot_err_rad(wrs_rotmat, RZ_PI @ py_rotmat):.9e}")
        print(f"   real_forpos UR7E vs py FK pos_err_m={np.linalg.norm(real_pos - py_pos):.9e}")
        print(f"   real_forpos UR7E vs py FK rot_err_rad={rot_err_rad(real_rotmat, py_rotmat):.9e}")
    print()

    wrs_solutions = collect_wrs_ik_solutions(arm, TARGET_TRANSLATION, TARGET_ROTATION, py_solutions)
    print_solution_summary(
        "WRS active UR7E numerical IK for the same raw target",
        wrs_solutions,
        arm,
        TARGET_TRANSLATION,
        TARGET_ROTATION,
    )

    equivalent_wrs_pos = RZ_PI @ TARGET_TRANSLATION
    equivalent_wrs_rot = RZ_PI @ TARGET_ROTATION
    equivalent_wrs_solutions = collect_wrs_ik_solutions(arm, equivalent_wrs_pos, equivalent_wrs_rot, py_solutions)
    print_solution_summary(
        "WRS active UR7E numerical IK for Rz(pi)-transformed pyikfast target",
        equivalent_wrs_solutions,
        arm,
        equivalent_wrs_pos,
        equivalent_wrs_rot,
    )

    consistency_q = np.array([0.3, -1.0, 1.2, -0.8, 0.5, -0.2], dtype=float)
    consistency_pos, consistency_rot_flat = pyikfast.forward(consistency_q.tolist())
    consistency_solutions = pyikfast.inverse(consistency_pos, consistency_rot_flat)
    best_consistency_error = min(
        (joint_delta_norm(consistency_q, solution) for solution in consistency_solutions),
        default=float("nan"),
    )

    print("pyikfast internal FK -> IK consistency check")
    print("-------------------------------------------")
    print(f"original q_rad={np.array2string(consistency_q, precision=9)}")
    print(f"returned solution count={len(consistency_solutions)}")
    print(f"best joint-space error modulo 2pi={best_consistency_error:.9e}")
    if consistency_solutions:
        best_solution = min(consistency_solutions, key=lambda solution: joint_delta_norm(consistency_q, solution))
        print(f"best returned q_rad={np.array2string(np.asarray(best_solution), precision=9)}")


if __name__ == "__main__":
    main()
