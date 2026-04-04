"""
Piper Pick-and-Place Wrapper
=============================

Convenience wrapper around the WRS ``PickPlacePlanner`` that
pre-configures a Piper 6-DoF arm and provides high-level
``pick_and_place`` functionality.

Usage
-----
>>> from sealp.pick_and_place.piper_pnp import PiperPickAndPlace
>>> pnp = PiperPickAndPlace()
>>> pnp.pick_and_place(obj_cmodel, grasp_collection, goal_pose_list)
"""

import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.manipulators.piper.piper as piper_mod
import wrs.manipulation.pick_place as pp


class PiperPickAndPlace:
    """High-level pick-and-place controller backed by a Piper arm.

    Parameters
    ----------
    pos : np.ndarray
        Base position of the Piper arm in world coordinates.
    rotmat : np.ndarray
        Base orientation of the Piper arm (3×3 rotation matrix).
    enable_cc : bool
        Enable self-collision checking on the arm.
    name : str
        Identifier for this robot instance.
    """

    def __init__(self,
                 pos: np.ndarray = np.zeros(3),
                 rotmat: np.ndarray = np.eye(3),
                 enable_cc: bool = True,
                 name: str = "piper_pnp"):
        # Instantiate the Piper arm
        self.robot = piper_mod.Piper(pos=pos, rotmat=rotmat,
                                     name=name, enable_cc=enable_cc)
        # Build the pick-and-place planner
        self._planner = pp.PickPlacePlanner(robot=self.robot)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def planner(self) -> pp.PickPlacePlanner:
        """Return the underlying ``PickPlacePlanner``."""
        return self._planner

    @property
    def arm(self) -> piper_mod.Piper:
        """Return the underlying ``Piper`` arm."""
        return self.robot

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------
    def pick_and_place(self,
                       obj_cmodel: mcm.CollisionModel,
                       grasp_collection,
                       goal_pose_list: list,
                       start_jnt_values=None,
                       end_jnt_values=None,
                       pick_approach_direction=None,
                       pick_approach_distance: float = 0.07,
                       pick_depart_direction=None,
                       pick_depart_distance: float = 0.07,
                       place_approach_direction_list=None,
                       place_approach_distance_list=None,
                       place_depart_direction_list=None,
                       place_depart_distance_list=None,
                       obstacle_list=None,
                       use_rrt: bool = True,
                       toggle_dbg: bool = False):
        """Plan a full pick-and-place motion.

        Delegates to ``PickPlacePlanner.gen_pick_and_place`` with
        sensible defaults for the Piper arm.

        Parameters
        ----------
        obj_cmodel : mcm.CollisionModel
            The object to be picked.
        grasp_collection
            A ``GraspCollection`` instance with pre-annotated grasps.
        goal_pose_list : list of (pos, rotmat)
            Target poses for the object.
        start_jnt_values : np.ndarray or None
            Starting joint configuration.  ``None`` → current config.
        end_jnt_values : np.ndarray or None
            Final joint configuration.  ``None`` → current config.
        obstacle_list : list or None
            Collision models to avoid.
        use_rrt : bool
            Use RRT for transit motion.
        toggle_dbg : bool
            Print debugging info and render intermediate states.

        Returns
        -------
        MotionData or None
            The planned motion, or ``None`` if planning fails.
        """
        n_goals = len(goal_pose_list)
        if place_approach_direction_list is None:
            place_approach_direction_list = [-rm.const.z_ax] * n_goals
        if place_approach_distance_list is None:
            place_approach_distance_list = [0.07] * n_goals
        if place_depart_direction_list is None:
            place_depart_direction_list = [rm.const.z_ax] * n_goals
        if place_depart_distance_list is None:
            place_depart_distance_list = [0.07] * n_goals
        if obstacle_list is None:
            obstacle_list = []

        return self._planner.gen_pick_and_place(
            obj_cmodel=obj_cmodel,
            grasp_collection=grasp_collection,
            goal_pose_list=goal_pose_list,
            start_jnt_values=start_jnt_values,
            end_jnt_values=end_jnt_values,
            pick_approach_direction=pick_approach_direction,
            pick_approach_distance=pick_approach_distance,
            pick_depart_direction=pick_depart_direction,
            pick_depart_distance=pick_depart_distance,
            place_approach_direction_list=place_approach_direction_list,
            place_approach_distance_list=place_approach_distance_list,
            place_depart_direction_list=place_depart_direction_list,
            place_depart_distance_list=place_depart_distance_list,
            obstacle_list=obstacle_list,
            use_rrt=use_rrt,
            toggle_dbg=toggle_dbg,
        )

    def pick_and_moveto(self,
                        obj_cmodel: mcm.CollisionModel,
                        grasp,
                        moveto_pose_list: list,
                        start_jnt_values=None,
                        pick_approach_direction=None,
                        pick_approach_distance: float = 0.07,
                        pick_depart_direction=None,
                        pick_depart_distance: float = 0.07,
                        obstacle_list=None,
                        use_rrt: bool = True,
                        toggle_dbg: bool = False):
        """Plan a pick followed by movement to one or more poses.

        Parameters
        ----------
        obj_cmodel : mcm.CollisionModel
            The object to be picked.
        grasp
            A single grasp from a ``GraspCollection``.
        moveto_pose_list : list of (pos, rotmat)
            Target poses to visit.
        obstacle_list : list or None
            Collision models to avoid.

        Returns
        -------
        MotionData or None
        """
        n = len(moveto_pose_list)
        if obstacle_list is None:
            obstacle_list = []

        return self._planner.gen_pick_and_moveto(
            obj_cmodel=obj_cmodel,
            grasp=grasp,
            moveto_pose_list=moveto_pose_list,
            moveto_approach_direction_list=[-rm.const.z_ax] * n,
            moveto_approach_distance_list=[0.07] * n,
            moveto_depart_direction_list=[rm.const.z_ax] * n,
            moveto_depart_distance_list=[0.07] * n,
            start_jnt_values=start_jnt_values,
            pick_approach_direction=pick_approach_direction,
            pick_approach_distance=pick_approach_distance,
            pick_depart_direction=pick_depart_direction,
            pick_depart_distance=pick_depart_distance,
            obstacle_list=obstacle_list,
            use_rrt=use_rrt,
            toggle_dbg=toggle_dbg,
        )

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def gen_meshmodel(self, **kwargs):
        """Generate a renderable mesh model of the robot."""
        return self.robot.gen_meshmodel(**kwargs)

    def gen_stickmodel(self, **kwargs):
        """Generate a stick-figure model of the robot."""
        return self.robot.gen_stickmodel(**kwargs)

    @staticmethod
    def animate(world, mot_data, interval: float = 0.01):
        """Animate a ``MotionData`` sequence in the given ``World``.

        Parameters
        ----------
        world : wd.World
            A Panda3D world instance.
        mot_data
            A ``MotionData`` returned by the planner.
        interval : float
            Delay between frames in seconds.
        """

        class _AnimData:
            def __init__(self, md):
                self.counter = 0
                self.mot_data = md

        anim = _AnimData(mot_data)

        def _update(anim_data, task):
            if anim_data.counter > 0:
                anim_data.mot_data.mesh_list[anim_data.counter - 1].detach()
            if anim_data.counter >= len(anim_data.mot_data):
                anim_data.counter = 0
            mesh = anim_data.mot_data.mesh_list[anim_data.counter]
            mesh.attach_to(world)
            if world.inputmgr.keymap['space']:
                anim_data.counter += 1
            return task.again

        world.taskMgr.doMethodLater(interval, _update, "piper_pnp_animate",
                                    extraArgs=[anim], appendTask=True)
