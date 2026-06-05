import numpy as np

import pyikfast
from wrs.robot_sim.manipulators.ur7e.ur7e import UR7E


np.set_printoptions(precision=9, suppress=True)


def main():
    q = np.array([0.3, -1.0, 1.2, -0.8, 0.5, -0.2], dtype=float)

    py_pos, py_rot = pyikfast.forward(q.tolist())
    py_pos = np.asarray(py_pos)
    py_rot = np.asarray(py_rot).reshape(3, 3)

    arm = UR7E()
    wrs_pos, wrs_rot = arm.fk(q)


    print("q:")
    print(q)
    print()

    print("pyikfast FK:")
    print(py_pos)
    print(py_rot)
    print()

    print("WRS UR7E FK:")
    print(wrs_pos)
    print(wrs_rot)
    print()

    print("raw FK position diff:")
    print(np.linalg.norm(wrs_pos - py_pos))
    print("raw FK rotation matrix diff:")
    print(np.linalg.norm(wrs_rot - py_rot))
    print()

    print("after Rz(pi) base transform:")
    print("position diff:")
    print(np.linalg.norm(wrs_pos - py_pos))
    print("rotation matrix diff:")
    print(np.linalg.norm(wrs_rot - py_rot))


if __name__ == "__main__":
    main()
