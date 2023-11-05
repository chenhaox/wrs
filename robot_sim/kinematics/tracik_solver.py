import numpy as np
import multiprocessing as mp
import basis.robot_math as rm
import scipy.optimize as sopt
import robot_sim.kinematics.constant as rkc


def _fk(anchor, joints, tcp_joint_id, tcp_loc_homomat, joint_values, toggle_jacobian):
    """
    joints = jlc.joints
    author: weiwei
    date: 20231105
    """
    n_dof = len(joints)
    homomat = anchor.homomat
    j_pos = np.zeros((n_dof, 3))
    j_axis = np.zeros((n_dof, 3))
    for i in range(tcp_joint_id + 1):
        j_axis[i, :] = homomat[:3, :3] @ joints[i].loc_motion_axis
        if joints[i].type == rkc.JointType.REVOLUTE:
            j_pos[i, :] = homomat[:3, 3] + homomat[:3, :3] @ joints[i].loc_pos
        homomat = homomat @ joints[i].get_motion_homomat(motion_value=joint_values[i])
    tcp_gl_homomat = homomat @ tcp_loc_homomat
    tcp_gl_pos = tcp_gl_homomat[:3, 3]
    tcp_gl_rotmat = tcp_gl_homomat[:3, :3]
    if toggle_jacobian:
        j_mat = np.zeros((6, n_dof))
        for i in range(tcp_joint_id + 1):
            if joints[i].type == rkc.JointType.REVOLUTE:
                vec_jnt2tcp = tcp_gl_pos - j_pos[i, :]
                j_mat[:3, i] = np.cross(j_axis[i, :], vec_jnt2tcp)
                j_mat[3:6, i] = j_axis[i, :]
            if joints[i].type == rkc.JointType.PRISMATIC:
                j_mat[:3, i] = j_axis[i, :]
        return tcp_gl_pos, tcp_gl_rotmat, j_mat
    else:
        return tcp_gl_pos, tcp_gl_rotmat


def _get_joint_ranges(joints):
    """
    get jntsrnage
    :return: [[jnt1min, jnt1max], [jnt2min, jnt2max], ...]
    date: 20180602, 20200704osaka
    author: weiwei
    """
    jnt_limits = []
    for i in range(len(joints)):
        jnt_limits.append(joints[i].motion_range)
    return np.asarray(jnt_limits)


class NumIKSolverProc(mp.Process):
    def __init__(self, anchor, joints, tcp_joint_id, tcp_loc_homomat, wln_ratio, param_queue, state_queue,
                 result_queue):
        super(NumIKSolverProc, self).__init__()
        self._param_queue = param_queue
        self._state_queue = state_queue
        self._result_queue = result_queue
        # nik related preparation
        self.n_dof = len(joints)
        self.anchor = anchor
        self.joints = joints
        self.tcp_joint_id = tcp_joint_id
        self.tcp_loc_homomat = tcp_loc_homomat
        self.max_link_length = self._get_max_link_length()
        self.clamp_pos_err = 2 * self.max_link_length
        self.clamp_rot_err = np.pi / 3
        self.jnt_wt_ratio = wln_ratio
        # maximum reach
        self.max_rng = 10.0
        # # extract min max for quick access
        self.joint_ranges = _get_joint_ranges(joints)
        self.min_jnt_vals = self.joint_ranges[:, 0]
        self.max_jnt_vals = self.joint_ranges[:, 1]
        self.jnt_rngs = self.max_jnt_vals - self.min_jnt_vals
        self.jnt_rngs_mid = (self.max_jnt_vals + self.min_jnt_vals) / 2
        self.min_jnt_threshold = self.min_jnt_vals + self.jnt_rngs * self.jnt_wt_ratio
        self.max_jnt_threshold = self.max_jnt_vals - self.jnt_rngs * self.jnt_wt_ratio

    def _get_max_link_length(self):
        max_len = 0
        for i in range(1, self.n_dof):
            if self.joints[i].type == rkc.JointType.REVOLUTE:
                tmp_vec = self.joints[i].gl_pos_q - self.joints[i - 1].gl_pos_q
                tmp_len = np.linalg.norm(tmp_vec)
                if tmp_len > max_len:
                    max_len = tmp_len
        return max_len

    def _jnt_wt_mat(self, jnt_values):
        """
        get the joint weight mat
        :param jnt_values:
        :return: W, W^(1/2)
        author: weiwei
        date: 20201126
        """
        jnt_wt = np.ones(self.n_dof)
        # min damping interval
        selection = jnt_values < self.min_jnt_threshold
        normalized_diff = ((jnt_values - self.min_jnt_vals) / (self.min_jnt_threshold - self.min_jnt_vals))[selection]
        jnt_wt[selection] = -2 * np.power(normalized_diff, 3) + 3 * np.power(normalized_diff, 2)
        # max damping interval
        selection = jnt_values > self.max_jnt_threshold
        normalized_diff = ((self.max_jnt_vals - jnt_values) / (self.max_jnt_vals - self.max_jnt_threshold))[selection]
        jnt_wt[selection] = -2 * np.power(normalized_diff, 3) + 3 * np.power(normalized_diff, 2)
        jnt_wt[jnt_values >= self.max_jnt_vals] = 0
        jnt_wt[jnt_values <= self.min_jnt_vals] = 0
        return np.diag(jnt_wt), np.diag(np.sqrt(jnt_wt))

    def _clamp_tcp_err(self, tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec):
        clamped_tcp_vec = np.copy(tcp_err_vec)
        if tcp_pos_err_val >= self.clamp_pos_err:
            clamped_tcp_vec[:3] = self.clamp_pos_err * tcp_err_vec[:3] / tcp_pos_err_val
        if tcp_rot_err_val >= self.clamp_rot_err:
            clamped_tcp_vec[3:6] = self.clamp_rot_err * tcp_err_vec[3:6] / tcp_rot_err_val
        return clamped_tcp_vec

    def run(self):
        while True:
            tgt_pos, tgt_rotmat, seed_jnt_vals, max_n_iter = self._param_queue.get()
            # print("numik starting")
            iter_jnt_vals = seed_jnt_vals.copy()
            counter = 0
            while self._result_queue.empty():
                tcp_gl_pos, tcp_gl_rotmat, j_mat = _fk(self.anchor,
                                                       self.joints,
                                                       self.tcp_joint_id,
                                                       self.tcp_loc_homomat,
                                                       joint_values=iter_jnt_vals,
                                                       toggle_jacobian=True)
                tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_posrot(src_pos=tcp_gl_pos,
                                                                                       src_rotmat=tcp_gl_rotmat,
                                                                                       tgt_pos=tgt_pos,
                                                                                       tgt_rotmat=tgt_rotmat)
                if tcp_pos_err_val < 1e-4 and tcp_rot_err_val < 1e-3:
                    # print("num got result")
                    self._result_queue.put(iter_jnt_vals)
                    break
                clamped_err_vec = self._clamp_tcp_err(tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec)
                wln, wln_sqrt = self._jnt_wt_mat(iter_jnt_vals)
                # weighted clamping
                k_phi = 0.1
                phi_q = ((2 * iter_jnt_vals - self.jnt_rngs_mid) / self.jnt_rngs) * k_phi
                clamping = -(np.identity(wln.shape[0]) - wln) @ phi_q
                # pinv with weighted clamping
                delta_jnt_values = clamping + wln_sqrt @ np.linalg.pinv(j_mat @ wln_sqrt, rcond=1e-4) @ (
                        clamped_err_vec - j_mat @ clamping)
                iter_jnt_vals = iter_jnt_vals + delta_jnt_values
                if counter > max_n_iter:
                    # print("numik failed")
                    break
                counter += 1
            # self._state_queue.put(1)


class OptIKSolverProc(mp.Process):
    def __init__(self, anchor, joints, tcp_joint_id, tcp_loc_homomat, param_queue, state_queue, result_queue):
        super(OptIKSolverProc, self).__init__()
        self._param_queue = param_queue
        self._result_queue = result_queue
        self._state_queue = state_queue
        self.anchor = anchor
        self.joints = joints
        self.joint_ranges = _get_joint_ranges(joints)
        self.tcp_joint_id = tcp_joint_id
        self.tcp_loc_homomat = tcp_loc_homomat

    def run(self):  # OptIKSolver.sqpss
        """
        sqpss is faster than sqp
        :return:
        author: weiwei
        date: 20231101
        """

        def _objective(x, tgt_pos, tgt_rotmat):
            tcp_gl_pos, tcp_gl_rotmat = _fk(self.anchor,
                                            self.joints,
                                            self.tcp_joint_id,
                                            self.tcp_loc_homomat,
                                            joint_values=x,
                                            toggle_jacobian=False)
            tcp_pos_err_val, tcp_rot_err_val, tcp_err_vec = rm.diff_between_posrot(src_pos=tcp_gl_pos,
                                                                                   src_rotmat=tcp_gl_rotmat,
                                                                                   tgt_pos=tgt_pos,
                                                                                   tgt_rotmat=tgt_rotmat)
            return tcp_err_vec.dot(tcp_err_vec)

        def _call_back(x):
            if not self._result_queue.empty():
                raise StopIteration

        while True:
            tgt_pos, tgt_rotmat, seed_jnt_vals, max_n_iter = self._param_queue.get()
            # print("optik starting")
            options = {'ftol': 1e-6,
                       'eps': 1e-12,
                       'maxiter': max_n_iter}
            try:
                result = sopt.minimize(fun=_objective,
                                       args=(tgt_pos, tgt_rotmat),
                                       x0=seed_jnt_vals,
                                       method='SLSQP',
                                       bounds=self.joint_ranges,
                                       options=options,
                                       callback=_call_back)
            except StopIteration:
                # self._state_queue.put(1)
                continue
            if result.success and result.fun < 1e-4:
                # print("opt got result")
                self._result_queue.put(result.x)
            else:
                self._result_queue.put(None)
                # print("optik failed")
            # self._state_queue.put(1)


class TracIKSolver(object):
    """
    author: weiwei
    date: 20231102
    """

    def __init__(self, jlc, wln_ratio=.05):
        self.jlc = jlc
        self._nik_param_queue = mp.Queue()
        self._oik_param_queue = mp.Queue()
        self._nik_state_queue = mp.Queue()
        self._oik_state_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self.nik_solver_proc = NumIKSolverProc(self.jlc.anchor,
                                               self.jlc.joints,
                                               self.jlc.tcp_joint_id,
                                               self.jlc.tcp_loc_homomat,
                                               wln_ratio,
                                               self._nik_param_queue,
                                               self._nik_state_queue,
                                               self._result_queue)
        self.oik_solver_proc = OptIKSolverProc(self.jlc.anchor,
                                               self.jlc.joints,
                                               self.jlc.tcp_joint_id,
                                               self.jlc.tcp_loc_homomat,
                                               self._oik_param_queue,
                                               self._oik_state_queue,
                                               self._result_queue)
        self.nik_solver_proc.start()
        self.oik_solver_proc.start()
        tcp_gl_pos, tcp_gl_rotmat = self.jlc.get_gl_tcp()
        # run once to avoid long waiting time in the beginning
        self._oik_param_queue.put((tcp_gl_pos, tcp_gl_rotmat, self.jlc.get_joint_values(), 10))
        self._result_queue.get()

    def ik(self, tgt_pos, tgt_rotmat, seed_jnt_vals=None, max_n_iter=100):
        if seed_jnt_vals is None:
            seed_jnt_vals = self.jlc.get_joint_values()
        self._nik_param_queue.put((tgt_pos, tgt_rotmat, seed_jnt_vals, max_n_iter))
        self._oik_param_queue.put((tgt_pos, tgt_rotmat, seed_jnt_vals, max_n_iter))
        result = self._result_queue.get()
        print(result)
        return result
        # # oik_state = self._oik_state_queue.get()
        # # nik_state = self._nik_state_queue.get()
        # print(self._result_queue.empty())
        # if not self._result_queue.empty():
        #     print("done")
        #     result = self._result_queue.get()
        #     return result
        # else:
        #     return None
