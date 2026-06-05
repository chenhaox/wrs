import heapq
import itertools
import time
from dataclasses import dataclass

import numpy as np

import wrs.basis.robot_math as rm
import wrs.modeling.model_collection as mmc
import wrs.motion.probabilistic.rrt_connect as rrtc


def _set_arm_conf(robot, arm_name, conf):
    manipulator = robot.manipulator_dict[arm_name]
    manipulator.goto_given_conf(jnt_values=np.asarray(conf, dtype=float))
    if hasattr(robot, "_update_oih"):
        robot._update_oih()


def _set_arm_conf_dict(robot, conf_dict):
    for arm_name, conf in conf_dict.items():
        _set_arm_conf(robot, arm_name, conf)


def _interpolate_conf(start_conf, goal_conf, granularity):
    length, direction = rm.unit_vector(goal_conf - start_conf, toggle_length=True)
    if length < 1e-9:
        return [np.asarray(start_conf, dtype=float)]
    n_steps = max(int(np.ceil(length / granularity)), 1)
    return [start_conf + direction * length * i / n_steps for i in range(n_steps + 1)]


def _as_unit_axis(axis):
    axis = np.asarray([0.0, 0.0, 1.0] if axis is None else axis, dtype=float)
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-9:
        raise ValueError("tcp_z_axis_world must be a non-zero vector.")
    return axis / axis_length


def _is_tcp_z_axis_aligned(manipulator, world_axis, min_dot):
    tcp_rotmat = getattr(manipulator, "gl_tcp_rotmat", None)
    if tcp_rotmat is None:
        return False
    tcp_z_axis = np.asarray(tcp_rotmat, dtype=float)[:, 2]
    tcp_z_axis_length = np.linalg.norm(tcp_z_axis)
    if tcp_z_axis_length < 1e-9:
        return False
    return float(np.dot(tcp_z_axis / tcp_z_axis_length, world_axis)) >= min_dot


def _make_tcp_z_axis_constraint(max_angle, world_axis):
    if max_angle is None:
        return None
    world_axis = _as_unit_axis(world_axis)
    min_dot = float(np.cos(max_angle))

    def constraint(robot, _conf):
        manipulator = getattr(robot, "manipulator", robot)
        return _is_tcp_z_axis_aligned(manipulator, world_axis, min_dot)

    return constraint


class _ArmPlanningAdapter:
    """Expose one arm of a multi-arm robot as a small RRT-compatible robot."""

    def __init__(self, robot, arm_name, fixed_conf_dict):
        self.robot = robot
        self.name = robot.name + "_" + arm_name + "_planner"
        self.arm_name = arm_name
        self.fixed_conf_dict = fixed_conf_dict
        self.manipulator = robot.manipulator_dict[arm_name]
        self._delegator = self.manipulator

    @property
    def home_conf(self):
        return self.manipulator.home_conf

    @property
    def jnt_ranges(self):
        return self.manipulator.jnt_ranges

    @property
    def oiee_list(self):
        return []

    def backup_state(self):
        return self.robot.backup_state()

    def restore_state(self):
        return self.robot.restore_state()

    def are_jnts_in_ranges(self, jnt_values):
        return self.manipulator.are_jnts_in_ranges(jnt_values=np.asarray(jnt_values, dtype=float))

    def goto_given_conf(self, jnt_values):
        _set_arm_conf_dict(self.robot, self.fixed_conf_dict)
        _set_arm_conf(self.robot, self.arm_name, jnt_values)

    def get_jnt_values(self):
        return self.manipulator.get_jnt_values()

    def rand_conf(self):
        return self.manipulator.rand_conf()

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False, toggle_dbg=False):
        return self.robot.is_collided(obstacle_list=obstacle_list,
                                      other_robot_list=other_robot_list,
                                      toggle_contacts=toggle_contacts,
                                      toggle_dbg=toggle_dbg)

    def get_ee_values(self):
        return None

    def gen_meshmodel(self, *args, **kwargs):
        return mmc.ModelCollection(name=self.name + "_meshmodel")


class MultiArmMotionData:
    """A compact synchronized trajectory for multiple named arms."""

    def __init__(self, robot, arm_names, conf_list):
        self.robot = robot
        self.arm_names = tuple(arm_names)
        self.conf_list = conf_list

    @property
    def jv_list(self):
        return [np.concatenate([conf_dict[name] for name in self.arm_names]) for conf_dict in self.conf_list]

    def arm_conf_list(self, arm_name):
        return [conf_dict[arm_name] for conf_dict in self.conf_list]

    def apply(self, index):
        _set_arm_conf_dict(self.robot, self.conf_list[index])

    def __len__(self):
        return len(self.conf_list)

    def __iter__(self):
        return iter(self.conf_list)

    def __getitem__(self, index):
        return self.conf_list[index]


@dataclass
class _ArmPlanRequest:
    name: str
    start_conf: np.ndarray
    goal_conf: np.ndarray


class MultiArmRRTConnect:
    """Plan n single-arm RRT paths and coordinate them with prioritized SIPP-style waits."""

    def __init__(self, robot):
        self.robot = robot
        self.arm_requests = []
        self.arm_path_dict = {}
        self.schedule_state_list = []
        self.last_debug_info = {}

    def add_arm(self, name, start_conf, goal_conf):
        if name not in self.robot.manipulator_dict:
            raise ValueError(f"Unknown arm name: {name}")
        self.arm_requests.append(_ArmPlanRequest(name=name,
                                                 start_conf=np.asarray(start_conf, dtype=float),
                                                 goal_conf=np.asarray(goal_conf, dtype=float)))

    def clear(self):
        self.arm_requests = []
        self.arm_path_dict = {}
        self.schedule_state_list = []
        self.last_debug_info = {}

    def _make_start_conf_dict(self):
        return {request.name: request.start_conf for request in self.arm_requests}

    @staticmethod
    def _conf_text(conf):
        return np.array2string(np.asarray(conf, dtype=float), precision=3, suppress_small=True)

    @staticmethod
    def _copy_conf_dict(conf_dict):
        return {name: np.asarray(conf, dtype=float).copy() for name, conf in conf_dict.items()}

    def _make_sipp_conf_dict(self,
                             arm_name,
                             path,
                             path_id,
                             time_id,
                             scheduled_conf_list_dict,
                             start_conf_dict):
        conf_dict = {}
        for scheduled_arm_name, conf_list in scheduled_conf_list_dict.items():
            conf_dict[scheduled_arm_name] = self._conf_at_time(conf_list, time_id)
        conf_dict[arm_name] = path[min(path_id, len(path) - 1)]
        for start_arm_name, start_conf in start_conf_dict.items():
            if start_arm_name not in conf_dict:
                conf_dict[start_arm_name] = start_conf
        return self._copy_conf_dict(conf_dict)

    @staticmethod
    def _collision_summary(robot, obstacle_list, other_robot_list):
        try:
            result = robot.is_collided(obstacle_list=obstacle_list,
                                       other_robot_list=other_robot_list,
                                       toggle_contacts=True)
        except TypeError:
            result = robot.is_collided(obstacle_list=obstacle_list,
                                       other_robot_list=other_robot_list)
        if isinstance(result, tuple):
            is_collided, contacts = result
            return bool(is_collided), len(contacts)
        return bool(result), None

    def _check_direct_path(self,
                           adapter,
                           start_conf,
                           goal_conf,
                           obstacle_list,
                           other_robot_list,
                           granularity,
                           collect_all,
                           tcp_z_axis_world,
                           tcp_z_axis_min_dot):
        path = _interpolate_conf(start_conf, goal_conf, granularity)
        bad_indices = []
        for i, conf in enumerate(path):
            adapter.goto_given_conf(conf)
            if tcp_z_axis_min_dot is not None and not _is_tcp_z_axis_aligned(
                    adapter.manipulator, tcp_z_axis_world, tcp_z_axis_min_dot):
                bad_indices.append(i)
                if not collect_all:
                    break
            elif adapter.is_collided(obstacle_list=obstacle_list, other_robot_list=other_robot_list):
                bad_indices.append(i)
                if not collect_all:
                    break
        return len(bad_indices) == 0, path, bad_indices

    def _plan_single_arm(self,
                         request,
                         fixed_conf_dict,
                         obstacle_list,
                         other_robot_list,
                         ext_dist,
                         max_time,
                         max_n_iter,
                         smoothing_n_iter,
                         tcp_z_axis_world,
                         tcp_z_axis_min_dot,
                         conf_constraint_fn,
                         toggle_dbg):
        fixed_conf_dict = {name: conf for name, conf in fixed_conf_dict.items() if name != request.name}
        adapter = _ArmPlanningAdapter(self.robot, request.name, fixed_conf_dict)
        if toggle_dbg:
            print(f"MultiArmRRTConnect: planning {request.name}; "
                  f"joint_delta={np.linalg.norm(request.goal_conf - request.start_conf):.3f}, "
                  f"fixed_arms={list(fixed_conf_dict.keys())}, max_time={max_time}.")
            print(f"  start={self._conf_text(request.start_conf)}")
            print(f"  goal ={self._conf_text(request.goal_conf)}")
        direct_valid, direct_path, bad_indices = self._check_direct_path(adapter,
                                                                         request.start_conf,
                                                                         request.goal_conf,
                                                                         obstacle_list,
                                                                         other_robot_list,
                                                                         granularity=ext_dist,
                                                                         collect_all=toggle_dbg,
                                                                         tcp_z_axis_world=tcp_z_axis_world,
                                                                         tcp_z_axis_min_dot=tcp_z_axis_min_dot)
        if direct_valid:
            if toggle_dbg:
                print(f"MultiArmRRTConnect: {request.name} direct joint path is valid "
                      f"({len(direct_path)} states).")
            return direct_path
        if toggle_dbg:
            print(f"MultiArmRRTConnect: {request.name} direct joint path is blocked/invalid; "
                  f"{len(bad_indices)}/{len(direct_path)} sampled states rejected, "
                  f"first_bad_indices={bad_indices[:8]}.")
        planner = rrtc.RRTConnect(adapter)
        planner.rbt = adapter
        tic = time.perf_counter()
        mot_data = planner.plan(start_conf=request.start_conf,
                                goal_conf=request.goal_conf,
                                obstacle_list=obstacle_list,
                                other_robot_list=other_robot_list,
                                ext_dist=ext_dist,
                                max_n_iter=max_n_iter,
                                max_time=max_time,
                                smoothing_n_iter=smoothing_n_iter,
                                toggle_dbg=toggle_dbg,
                                conf_constraint_fn=conf_constraint_fn)
        elapsed = time.perf_counter() - tic
        if mot_data is None:
            if toggle_dbg:
                print(f"MultiArmRRTConnect: RRT failed for {request.name} after {elapsed:.2f}s.")
            return None
        if toggle_dbg:
            print(f"MultiArmRRTConnect: RRT planned {request.name} in {elapsed:.2f}s "
                  f"with {len(mot_data.jv_list)} states.")
        return mot_data.jv_list

    def _candidate_next_states(self, state, goal_state, max_moving_arms):
        movable_axes = [i for i, (value, goal) in enumerate(zip(state, goal_state)) if value < goal]
        for flags in itertools.product([0, 1], repeat=len(movable_axes)):
            if not any(flags):
                continue
            if max_moving_arms is not None and sum(flags) > max_moving_arms:
                continue
            next_state = list(state)
            for axis, flag in zip(movable_axes, flags):
                if flag:
                    next_state[axis] += 1
            yield tuple(next_state)

    def _conf_dict_at_state(self, arm_names, path_list, state):
        return {name: path[index] for name, path, index in zip(arm_names, path_list, state)}

    def _is_schedule_edge_valid(self,
                                arm_names,
                                path_list,
                                state,
                                next_state,
                                obstacle_list,
                                other_robot_list,
                                granularity):
        prev_conf_dict = self._conf_dict_at_state(arm_names, path_list, state)
        next_conf_dict = self._conf_dict_at_state(arm_names, path_list, next_state)
        max_steps = 1
        for name in arm_names:
            length = np.linalg.norm(next_conf_dict[name] - prev_conf_dict[name])
            max_steps = max(max_steps, int(np.ceil(length / granularity)))
        for step in range(1, max_steps + 1):
            ratio = step / max_steps
            conf_dict = {}
            for name in arm_names:
                conf_dict[name] = prev_conf_dict[name] + (next_conf_dict[name] - prev_conf_dict[name]) * ratio
            _set_arm_conf_dict(self.robot, conf_dict)
            if self.robot.is_collided(obstacle_list=obstacle_list,
                                      other_robot_list=other_robot_list):
                return False
        return True

    def _edge_cost(self, arm_names, path_list, state, next_state):
        cost = 0.0
        for name, path, prev_i, next_i in zip(arm_names, path_list, state, next_state):
            if prev_i != next_i:
                cost = max(cost, np.linalg.norm(path[next_i] - path[prev_i]))
        return max(cost, 1e-6)

    def _heuristic(self, path_list, state, goal_state):
        return sum(goal - value for value, goal in zip(state, goal_state))

    def _coordinate_paths(self,
                          arm_names,
                          path_list,
                          obstacle_list,
                          other_robot_list,
                          granularity,
                          max_moving_arms,
                          max_time,
                          start_time):
        start_state = tuple(0 for _ in path_list)
        goal_state = tuple(len(path) - 1 for path in path_list)
        queue = []
        counter = itertools.count()
        heapq.heappush(queue, (0.0, next(counter), start_state))
        came_from = {start_state: None}
        cost_so_far = {start_state: 0.0}
        valid_edge_cache = {}
        while queue:
            if max_time is not None and time.perf_counter() - start_time > max_time:
                return None
            _, _, state = heapq.heappop(queue)
            if state == goal_state:
                break
            for next_state in self._candidate_next_states(state, goal_state, max_moving_arms):
                edge_key = (state, next_state)
                if edge_key not in valid_edge_cache:
                    valid_edge_cache[edge_key] = self._is_schedule_edge_valid(
                        arm_names=arm_names,
                        path_list=path_list,
                        state=state,
                        next_state=next_state,
                        obstacle_list=obstacle_list,
                        other_robot_list=other_robot_list,
                        granularity=granularity)
                if not valid_edge_cache[edge_key]:
                    continue
                new_cost = cost_so_far[state] + self._edge_cost(arm_names, path_list, state, next_state)
                if next_state not in cost_so_far or new_cost < cost_so_far[next_state]:
                    cost_so_far[next_state] = new_cost
                    priority = new_cost + self._heuristic(path_list, next_state, goal_state)
                    heapq.heappush(queue, (priority, next(counter), next_state))
                    came_from[next_state] = state
        if goal_state not in came_from:
            return None
        state_list = []
        state = goal_state
        while state is not None:
            state_list.append(state)
            state = came_from[state]
        state_list.reverse()
        return state_list

    @staticmethod
    def _conf_at_time(conf_list, time_id):
        return conf_list[min(time_id, len(conf_list) - 1)]

    @staticmethod
    def _is_schedule_moving(conf_list, time_id):
        current_conf = MultiArmRRTConnect._conf_at_time(conf_list, time_id)
        next_conf = MultiArmRRTConnect._conf_at_time(conf_list, time_id + 1)
        return np.linalg.norm(next_conf - current_conf) > 1e-9

    def _is_sipp_transition_valid(self,
                                  arm_name,
                                  path,
                                  path_id,
                                  next_path_id,
                                  time_id,
                                  scheduled_conf_list_dict,
                                  start_conf_dict,
                                  obstacle_list,
                                  other_robot_list,
                                  granularity,
                                  max_moving_arms,
                                  moving_tcp_clearance,
                                  tcp_z_axis_world,
                                  tcp_z_axis_min_dot,
                                  return_reason=False):
        def result(valid, reason="ok"):
            return (valid, reason) if return_reason else valid

        current_arm_moving = next_path_id != path_id
        scheduled_moving_dict = {
            scheduled_arm_name: self._is_schedule_moving(conf_list, time_id)
            for scheduled_arm_name, conf_list in scheduled_conf_list_dict.items()
        }
        if max_moving_arms is not None:
            moving_arm_count = int(current_arm_moving)
            moving_arm_count += sum(scheduled_moving_dict.values())
            if moving_arm_count > max_moving_arms:
                return result(False, "moving_arm_limit")
        current_conf = path[path_id]
        next_conf = path[next_path_id]
        max_steps = max(1, int(np.ceil(np.linalg.norm(next_conf - current_conf) / granularity)))
        for conf_list in scheduled_conf_list_dict.values():
            scheduled_current = self._conf_at_time(conf_list, time_id)
            scheduled_next = self._conf_at_time(conf_list, time_id + 1)
            max_steps = max(max_steps, int(np.ceil(np.linalg.norm(scheduled_next - scheduled_current) / granularity)))
        for step in range(1, max_steps + 1):
            ratio = step / max_steps
            conf_dict = {}
            for scheduled_arm_name, conf_list in scheduled_conf_list_dict.items():
                scheduled_current = self._conf_at_time(conf_list, time_id)
                scheduled_next = self._conf_at_time(conf_list, time_id + 1)
                conf_dict[scheduled_arm_name] = scheduled_current + (scheduled_next - scheduled_current) * ratio
            conf_dict[arm_name] = current_conf + (next_conf - current_conf) * ratio
            for start_arm_name, start_conf in start_conf_dict.items():
                if start_arm_name not in conf_dict:
                    conf_dict[start_arm_name] = start_conf
            _set_arm_conf_dict(self.robot, conf_dict)
            if tcp_z_axis_min_dot is not None:
                for checked_arm_name in conf_dict:
                    if not _is_tcp_z_axis_aligned(self.robot.manipulator_dict[checked_arm_name],
                                                  tcp_z_axis_world,
                                                  tcp_z_axis_min_dot):
                        return result(False, f"tcp_z_axis@{checked_arm_name}@substep{step}/{max_steps}")
            if moving_tcp_clearance is not None and current_arm_moving:
                current_tcp_pos = self.robot.manipulator_dict[arm_name].gl_tcp_pos
                for scheduled_arm_name, scheduled_is_moving in scheduled_moving_dict.items():
                    if not scheduled_is_moving:
                        continue
                    scheduled_tcp_pos = self.robot.manipulator_dict[scheduled_arm_name].gl_tcp_pos
                    if np.linalg.norm(current_tcp_pos - scheduled_tcp_pos) < moving_tcp_clearance:
                        return result(False, "moving_tcp_clearance")
            if self.robot.is_collided(obstacle_list=obstacle_list, other_robot_list=other_robot_list):
                return result(False, f"collision@substep{step}/{max_steps}")
        return result(True)

    def _plan_sipp_arm_schedule(self,
                                arm_name,
                                path,
                                scheduled_conf_list_dict,
                                start_conf_dict,
                                obstacle_list,
                                other_robot_list,
                                granularity,
                                max_moving_arms,
                                moving_tcp_clearance,
                                tcp_z_axis_world,
                                tcp_z_axis_min_dot,
                                max_wait_steps,
                                max_time,
                                start_time,
                                toggle_dbg=False):
        start_state = (0, 0)
        goal_path_id = len(path) - 1
        scheduled_horizon = max([len(conf_list) - 1 for conf_list in scheduled_conf_list_dict.values()] + [0])
        horizon = scheduled_horizon + goal_path_id + max_wait_steps
        expanded_count = 0
        accepted_count = 0
        reject_counts = {}
        first_rejections = []
        queue = []
        counter = itertools.count()
        heapq.heappush(queue, (goal_path_id, next(counter), start_state))
        came_from = {start_state: None}
        cost_so_far = {start_state: 0}
        last_expanded_state = start_state
        while queue:
            if max_time is not None and time.perf_counter() - start_time > max_time:
                timeout_state = queue[0][2] if queue else last_expanded_state
                timeout_path_id, timeout_time_id = timeout_state
                self.last_debug_info.update({
                    "stage": "sipp_timeout",
                    "failed_arm": arm_name,
                    "sipp_state": timeout_state,
                    "sipp_conf_dict": self._make_sipp_conf_dict(arm_name=arm_name,
                                                                 path=path,
                                                                 path_id=timeout_path_id,
                                                                 time_id=timeout_time_id,
                                                                 scheduled_conf_list_dict=scheduled_conf_list_dict,
                                                                 start_conf_dict=start_conf_dict),
                    "sipp_scheduled_arms": list(scheduled_conf_list_dict.keys()),
                    "sipp_scheduled_horizon": scheduled_horizon,
                    "sipp_horizon": horizon,
                    "sipp_expanded_count": expanded_count,
                    "sipp_accepted_count": accepted_count,
                    "sipp_reject_counts": reject_counts.copy(),
                    "sipp_first_rejections": first_rejections.copy(),
                })
                if toggle_dbg:
                    elapsed = time.perf_counter() - start_time
                    print(f"MultiArmRRTConnect: SIPP timeout for {arm_name}; "
                          f"elapsed_total={elapsed:.2f}s, max_time={max_time}, "
                          f"expanded={expanded_count}, accepted={accepted_count}, rejects={reject_counts}.")
                return None, None
            _, _, state = heapq.heappop(queue)
            expanded_count += 1
            last_expanded_state = state
            path_id, time_id = state
            if path_id == goal_path_id and time_id >= scheduled_horizon:
                index_list = []
                while state is not None:
                    index_list.append(state[0])
                    state = came_from[state]
                index_list.reverse()
                if toggle_dbg:
                    print(f"MultiArmRRTConnect: SIPP scheduled {arm_name}; "
                          f"path_states={len(path)}, scheduled_states={len(index_list)}, "
                          f"expanded={expanded_count}, accepted={accepted_count}, rejects={reject_counts}.")
                return [path[index] for index in index_list], index_list
            if time_id >= horizon:
                continue
            next_path_ids = [path_id]
            if path_id < goal_path_id:
                next_path_ids.append(path_id + 1)
            for next_path_id in next_path_ids:
                next_state = (next_path_id, time_id + 1)
                if next_state in came_from:
                    continue
                is_valid, reason = self._is_sipp_transition_valid(arm_name=arm_name,
                                                                  path=path,
                                                                  path_id=path_id,
                                                                  next_path_id=next_path_id,
                                                                  time_id=time_id,
                                                                  scheduled_conf_list_dict=scheduled_conf_list_dict,
                                                                  start_conf_dict=start_conf_dict,
                                                                  obstacle_list=obstacle_list,
                                                                  other_robot_list=other_robot_list,
                                                                  granularity=granularity,
                                                                  max_moving_arms=max_moving_arms,
                                                                  moving_tcp_clearance=moving_tcp_clearance,
                                                                  tcp_z_axis_world=tcp_z_axis_world,
                                                                  tcp_z_axis_min_dot=tcp_z_axis_min_dot,
                                                                  return_reason=True)
                if not is_valid:
                    reject_counts[reason] = reject_counts.get(reason, 0) + 1
                    if toggle_dbg and len(first_rejections) < 8:
                        first_rejections.append((state, next_state, reason))
                    continue
                cost_so_far[next_state] = cost_so_far[state] + 1
                heuristic = goal_path_id - next_path_id
                heapq.heappush(queue, (cost_so_far[next_state] + heuristic, next(counter), next_state))
                came_from[next_state] = state
                accepted_count += 1
        if toggle_dbg:
            print(f"MultiArmRRTConnect: SIPP failed for {arm_name}; "
                  f"path_states={len(path)}, scheduled_arms={list(scheduled_conf_list_dict.keys())}, "
                  f"scheduled_horizon={scheduled_horizon}, horizon={horizon}, "
                  f"expanded={expanded_count}, accepted={accepted_count}, rejects={reject_counts}.")
            if first_rejections:
                print(f"MultiArmRRTConnect: first SIPP rejections for {arm_name}: {first_rejections}.")
        failed_path_id, failed_time_id = last_expanded_state
        self.last_debug_info.update({
            "stage": "sipp_failed",
            "failed_arm": arm_name,
            "sipp_state": last_expanded_state,
            "sipp_conf_dict": self._make_sipp_conf_dict(arm_name=arm_name,
                                                         path=path,
                                                         path_id=failed_path_id,
                                                         time_id=failed_time_id,
                                                         scheduled_conf_list_dict=scheduled_conf_list_dict,
                                                         start_conf_dict=start_conf_dict),
            "sipp_scheduled_arms": list(scheduled_conf_list_dict.keys()),
            "sipp_scheduled_horizon": scheduled_horizon,
            "sipp_horizon": horizon,
            "sipp_expanded_count": expanded_count,
            "sipp_accepted_count": accepted_count,
            "sipp_reject_counts": reject_counts.copy(),
            "sipp_first_rejections": first_rejections.copy(),
        })
        return None, None

    def _coordinate_paths_sipp(self,
                               arm_names,
                               path_list,
                               obstacle_list,
                               other_robot_list,
                               granularity,
                               max_moving_arms,
                               moving_tcp_clearance,
                               tcp_z_axis_world,
                               tcp_z_axis_min_dot,
                               max_wait_steps,
                               max_time,
                               start_time,
                               toggle_dbg=False):
        start_conf_dict = {arm_name: path[0] for arm_name, path in zip(arm_names, path_list)}
        scheduled_conf_list_dict = {}
        scheduled_index_list_dict = {}
        for arm_name, path in zip(arm_names, path_list):
            conf_list, index_list = self._plan_sipp_arm_schedule(
                arm_name=arm_name,
                path=path,
                scheduled_conf_list_dict=scheduled_conf_list_dict,
                start_conf_dict=start_conf_dict,
                obstacle_list=obstacle_list,
                other_robot_list=other_robot_list,
                granularity=granularity,
                max_moving_arms=max_moving_arms,
                moving_tcp_clearance=moving_tcp_clearance,
                tcp_z_axis_world=tcp_z_axis_world,
                tcp_z_axis_min_dot=tcp_z_axis_min_dot,
                max_wait_steps=max_wait_steps,
                max_time=max_time,
                start_time=start_time,
                toggle_dbg=toggle_dbg)
            if conf_list is None:
                return None, None
            scheduled_conf_list_dict[arm_name] = conf_list
            scheduled_index_list_dict[arm_name] = index_list
        schedule_len = max(len(conf_list) for conf_list in scheduled_conf_list_dict.values())
        conf_list = []
        state_list = []
        for time_id in range(schedule_len):
            conf_list.append({
                arm_name: self._conf_at_time(scheduled_conf_list_dict[arm_name], time_id)
                for arm_name in arm_names
            })
            state_list.append(tuple(
                scheduled_index_list_dict[arm_name][min(time_id, len(scheduled_index_list_dict[arm_name]) - 1)]
                for arm_name in arm_names))
        return conf_list, state_list

    def plan(self,
             obstacle_list=None,
             other_robot_list=None,
             ext_dist=.2,
             max_n_iter=10000,
             max_time=30.0,
             per_arm_max_time=None,
             smoothing_n_iter=50,
             coordination_ext_dist=None,
             max_moving_arms=None,
             moving_tcp_clearance=None,
             tcp_z_axis_max_angle=None,
             tcp_z_axis_world=None,
             max_wait_steps=100,
             toggle_dbg=False):
        if not self.arm_requests:
            raise ValueError("No arm requests were added.")
        if obstacle_list is None:
            obstacle_list = []
        start_time = time.perf_counter()
        coordination_ext_dist = ext_dist if coordination_ext_dist is None else coordination_ext_dist
        tcp_z_axis_world = _as_unit_axis(tcp_z_axis_world)
        tcp_z_axis_min_dot = None if tcp_z_axis_max_angle is None else float(np.cos(tcp_z_axis_max_angle))
        conf_constraint_fn = _make_tcp_z_axis_constraint(tcp_z_axis_max_angle, tcp_z_axis_world)
        self.robot.backup_state()
        try:
            start_conf_dict = self._make_start_conf_dict()
            self.last_debug_info = {
                "stage": "start",
                "start_conf_dict": self._copy_conf_dict(start_conf_dict),
                "goal_conf_dict": {
                    request.name: np.asarray(request.goal_conf, dtype=float).copy()
                    for request in self.arm_requests
                },
                "arm_order": [request.name for request in self.arm_requests],
                "params": {
                    "ext_dist": ext_dist,
                    "coordination_ext_dist": coordination_ext_dist,
                    "max_time": max_time,
                    "per_arm_max_time": per_arm_max_time,
                    "max_n_iter": max_n_iter,
                    "smoothing_n_iter": smoothing_n_iter,
                    "max_wait_steps": max_wait_steps,
                    "max_moving_arms": max_moving_arms,
                    "moving_tcp_clearance": moving_tcp_clearance,
                    "tcp_z_axis_max_angle": tcp_z_axis_max_angle,
                    "tcp_z_axis_world": tcp_z_axis_world.copy(),
                },
            }
            if toggle_dbg:
                tcp_z_axis_text = None if tcp_z_axis_max_angle is None else np.degrees(tcp_z_axis_max_angle)
                print("MultiArmRRTConnect: plan parameters "
                      f"arms={[request.name for request in self.arm_requests]}, ext_dist={ext_dist}, "
                      f"coordination_ext_dist={coordination_ext_dist}, max_time={max_time}, "
                      f"per_arm_max_time={per_arm_max_time}, max_n_iter={max_n_iter}, "
                      f"smoothing_n_iter={smoothing_n_iter}, max_wait_steps={max_wait_steps}, "
                      f"max_moving_arms={max_moving_arms}, moving_tcp_clearance={moving_tcp_clearance}, "
                      f"tcp_z_axis_max_angle_deg={tcp_z_axis_text}.")
            _set_arm_conf_dict(self.robot, start_conf_dict)
            if self.robot.is_collided(obstacle_list=obstacle_list, other_robot_list=other_robot_list):
                if toggle_dbg:
                    _, contact_count = self._collision_summary(self.robot, obstacle_list, other_robot_list)
                    print("MultiArmRRTConnect: start configuration is in collision; "
                          f"contact_count={contact_count}.")
                return None
            arm_names = [request.name for request in self.arm_requests]
            path_list = []
            self.arm_path_dict = {}
            for request in self.arm_requests:
                if max_time is None:
                    remaining_time = per_arm_max_time
                else:
                    remaining_time = max(max_time - (time.perf_counter() - start_time), 0.0)
                    if per_arm_max_time is not None:
                        remaining_time = min(remaining_time, per_arm_max_time)
                if toggle_dbg:
                    print(f"MultiArmRRTConnect: remaining time for {request.name}: {remaining_time}.")
                path = self._plan_single_arm(request=request,
                                             fixed_conf_dict=start_conf_dict,
                                             obstacle_list=obstacle_list,
                                             other_robot_list=other_robot_list,
                                             ext_dist=ext_dist,
                                             max_time=remaining_time,
                                             max_n_iter=max_n_iter,
                                             smoothing_n_iter=smoothing_n_iter,
                                             tcp_z_axis_world=tcp_z_axis_world,
                                             tcp_z_axis_min_dot=tcp_z_axis_min_dot,
                                             conf_constraint_fn=conf_constraint_fn,
                                             toggle_dbg=toggle_dbg)
                if path is None:
                    if toggle_dbg:
                        print(f"MultiArmRRTConnect: failed to plan arm {request.name}.")
                    self.last_debug_info.update({
                        "stage": "single_arm_rrt_failed",
                        "failed_arm": request.name,
                    })
                    return None
                path = [np.asarray(conf, dtype=float) for conf in path]
                self.arm_path_dict[request.name] = path
                path_list.append(path)
                self.last_debug_info["path_dict"] = {
                    name: [np.asarray(conf, dtype=float).copy() for conf in arm_path]
                    for name, arm_path in self.arm_path_dict.items()
                }
            if toggle_dbg:
                print("MultiArmRRTConnect: single-arm path lengths "
                      f"{dict(zip(arm_names, [len(path) for path in path_list]))}.")
                elapsed_before_coordination = time.perf_counter() - start_time
                remaining_total_time = None if max_time is None else max(max_time - elapsed_before_coordination, 0.0)
                print("MultiArmRRTConnect: entering SIPP coordination; "
                      f"elapsed_total={elapsed_before_coordination:.2f}s, "
                      f"remaining_total_time={remaining_total_time}.")
            self.last_debug_info.update({
                "stage": "coordination",
                "arm_names": tuple(arm_names),
                "path_dict": {
                    name: [np.asarray(conf, dtype=float).copy() for conf in path]
                    for name, path in zip(arm_names, path_list)
                },
                "path_lengths": dict(zip(arm_names, [len(path) for path in path_list])),
                "elapsed_before_coordination": time.perf_counter() - start_time,
            })
            conf_list, state_list = self._coordinate_paths_sipp(arm_names=arm_names,
                                                                path_list=path_list,
                                                                obstacle_list=obstacle_list,
                                                                other_robot_list=other_robot_list,
                                                                granularity=coordination_ext_dist,
                                                                max_moving_arms=max_moving_arms,
                                                                moving_tcp_clearance=moving_tcp_clearance,
                                                                tcp_z_axis_world=tcp_z_axis_world,
                                                                tcp_z_axis_min_dot=tcp_z_axis_min_dot,
                                                                max_wait_steps=max_wait_steps,
                                                                max_time=max_time,
                                                                start_time=start_time,
                                                                toggle_dbg=toggle_dbg)
            if conf_list is None:
                if toggle_dbg:
                    print("MultiArmRRTConnect: failed to coordinate arm paths.")
                return None
            self.schedule_state_list = state_list
            self.last_debug_info.update({
                "stage": "success",
                "schedule_state_list": list(state_list),
                "conf_list": [self._copy_conf_dict(conf_dict) for conf_dict in conf_list],
            })
            if toggle_dbg:
                print(f"MultiArmRRTConnect: coordinated {len(conf_list)} synchronized states.")
            return MultiArmMotionData(robot=self.robot, arm_names=arm_names, conf_list=conf_list)
        finally:
            self.robot.restore_state()
