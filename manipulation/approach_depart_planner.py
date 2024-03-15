import os
import math
import copy
import pickle
import numpy as np
import basis.data_adapter as da
import modeling.collision_model as cm
import manipulation.utils as mpu
import motion.primitives.interplated as mpi
import motion.probabilistic.rrt_connect as rrtc
import robot_sim.robots.single_arm_robot_interface as sari


class ADPlanner(object):
    """
    AD = Approach_Depart
    NOTE: only accept robot as initiator
    """

    def __init__(self, sgl_arm_robot):
        """
        :param robot_s:
        author: weiwei, hao
        date: 20191122, 20210113
        """
        if not isinstance(sgl_arm_robot, sari.SglArmRobotInterface):
            raise ValueError("Only single arm robot can be used to initiate an InterplateMotion instance!")
        self.sgl_arm_robot = sgl_arm_robot
        self.rrtc_planner = rrtc.RRTConnect(self.sgl_arm_robot)
        self.im_planner = mpi.InterplatedMotion(self.sgl_arm_robot)

    def gen_linear_approach(self,
                            goal_tcp_pos,
                            goal_tcp_rotmat,
                            direction=None,
                            distance=.1,
                            jaw_width=None,
                            granularity=0.03,
                            obstacle_list=[]):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param direction: use the loc_z of goal_tcp_rotmat if None
        :param distance:
        :param jaw_width:
        :param granularity:
        :param obstacle_list:
        :param seed_jnt_values
        :return:
        author: weiwei
        date: 20210125
        """
        if direction is None:
            direction = goal_tcp_rotmat[:, 2]
        if distance is None:
            distance = .1
        conf_list = self.im_planner.gen_rel_linear_motion(goal_tcp_pos,
                                                          goal_tcp_rotmat,
                                                          direction,
                                                          distance,
                                                          obstacle_list=obstacle_list,
                                                          granularity=granularity,
                                                          type="sink")
        if conf_list is None:
            print("ADPlanner: Cannot generate linear approach!")
            return None
        else:
            result_data = mpu.ManipulationData(sgl_arm_robot=self.sgl_arm_robot)
            result_data.extend(conf_list)
            result_data.jaw_width_list[0] = jaw_width
            return result_data

    def gen_linear_depart(self,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          direction=None,  # np.array([0, 0, 1])
                          distance=.1,
                          jaw_width=.05,
                          granularity=0.03,
                          obstacle_list=[]):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param direction:
        :param distance:
        :param jaw_width:
        :param granularity:
        :param seed_jnt_values:
        :return: conf_list, jaw_width_list, objhomomat_list_list
        author: weiwei
        date: 20210125
        """
        if direction is None:
            direction = -start_tcp_rotmat[:, 2]
        if distance is None:
            distance = .1
        conf_list = self.im_planner.gen_rel_linear_motion(start_tcp_pos,
                                                          start_tcp_rotmat,
                                                          direction,
                                                          distance,
                                                          obstacle_list=obstacle_list,
                                                          granularity=granularity,
                                                          type='source')
        if conf_list is None:
            print("ADPlanner: Cannot generate linear depart!")
            return None
        else:
            result_data = mpu.ManipulationData(sgl_arm_robot=self.sgl_arm_robot)
            result_data.extend(conf_list)
            result_data.jaw_width_list[0] = jaw_width
            return result_data

    def gen_linear_approach_depart(self,
                                   goal_tcp_pos,
                                   goal_tcp_rotmat,
                                   approach_direction=None,  # np.array([0, 0, -1])
                                   approach_distance=.1,
                                   approach_jaw_width=.05,
                                   depart_direction=None,  # np.array([0, 0, 1])
                                   depart_distance=.1,
                                   depart_jaw_width=0,
                                   granularity=.03,
                                   obstacle_list=[]):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param hnd_name:
        :param approach_direction:
        :param approach_distance:
        :param approach_jaw_width:
        :param depart_direction:
        :param depart_distance:
        :param depart_jaw_width:
        :param granularity:
        :param obstacle_list:
        :return: approach_conf_list, depart_jaw_width_list
        author: weiwei, hao
        date: 20191122, 20200105, 20210113, 20210125
        """
        if approach_direction is None:
            approach_direction = goal_tcp_rotmat[:, 2]
        if approach_distance is None:
            approach_distance = .1
        approach_conf_list = self.im_planner.gen_rel_linear_motion(goal_tcp_pos,
                                                                   goal_tcp_rotmat,
                                                                   approach_direction,
                                                                   approach_distance,
                                                                   obstacle_list=obstacle_list,
                                                                   granularity=granularity,
                                                                   type="sink")
        if approach_conf_list is None:
            print("ADPlanner: Cannot generate the approach section of linear approach-depart!")
            return None
        else:
            if depart_direction is None:
                depart_direction = goal_tcp_rotmat[:, 2]
            if depart_distance is None:
                depart_distance = .1
            depart_conf_list = self.im_planner.gen_rel_linear_motion(goal_tcp_pos,
                                                                     goal_tcp_rotmat,
                                                                     depart_direction,
                                                                     depart_distance,
                                                                     obstacle_list=obstacle_list,
                                                                     granularity=granularity,
                                                                     type='source')
            if depart_conf_list is None:
                print("ADPlanner: Cannot generate the depart section of linear approach-depart!")
                return None
            else:
                result_data = mpu.ManipulationData(sgl_arm_robot=self.sgl_arm_robot)
                result_data.extend(approach_conf_list + depart_conf_list)
                result_data.jaw_width_list[0] = approach_jaw_width
                result_data.jaw_width_list[len(approach_conf_list)] = depart_jaw_width
                return result_data

    def gen_linear_approach_with_given_conf(self,
                                            goal_conf,
                                            direction=None,
                                            distance=.1,
                                            jaw_width=.05,
                                            granularity=0.03,
                                            obstacle_list=[]):
        goal_tcp_pos, goal_tcp_rotmat = self.sgl_arm_robot.fk(goal_conf)
        if direction is None:
            direction = goal_tcp_rotmat[:, 2]
        if distance is None:
            distance = .1
        conf_list = self.im_planner.gen_rel_linear_motion_with_given_conf(goal_conf=goal_conf,
                                                                          direction=direction,
                                                                          distance=distance,
                                                                          obstacle_list=obstacle_list,
                                                                          granularity=granularity,
                                                                          type="sink")
        if conf_list is None:
            print('ADPlanner: Cannot generate linear approach with given conf!')
            return None
        else:
            result_data = mpu.ManipulationData(sgl_arm_robot=self.sgl_arm_robot)
            result_data.extend(conf_list)
            result_data.jaw_width_list[0] = jaw_width
            return result_data

    def gen_linear_depart_with_given_conf(self,
                                          start_conf,
                                          direction=None,
                                          distance=.1,
                                          jaw_width=.05,
                                          granularity=0.03,
                                          obstacle_list=[]):
        start_tcp_pos, start_tcp_rotmat = self.sgl_arm_robot.fk(start_conf)
        if direction is None:
            direction = -start_tcp_rotmat[:, 2]
        if distance is None:
            distance = .1
        conf_list = self.im_planner.gen_rel_linear_motion_with_given_conf(goal_conf=start_conf,
                                                                          direction=direction,
                                                                          distance=distance,
                                                                          obstacle_list=obstacle_list,
                                                                          granularity=granularity,
                                                                          type="source")
        if conf_list is None:
            print("ADPlanner: Cannot generate linear approach with given conf!")
            return None
        else:
            result_data = mpu.ManipulationData(sgl_arm_robot=self.sgl_arm_robot)
            result_data.extend(conf_list)
            result_data.jaw_width_list[0] = jaw_width
            return result_data

    def gen_linear_approach_depart_with_given_conf(self,
                                                   goal_conf,
                                                   approach_direction=None,  # np.array([0, 0, -1])
                                                   approach_distance=.1,
                                                   approach_jaw_width=.05,
                                                   depart_direction=None,  # np.array([0, 0, 1])
                                                   depart_distance=.1,
                                                   depart_jaw_width=0,
                                                   granularity=.03,
                                                   obstacle_list=[]):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param hnd_name:
        :param approach_direction:
        :param approach_distance:
        :param approach_jaw_width:
        :param depart_direction:
        :param depart_distance:
        :param depart_jaw_width:
        :param granularity:
        :param obstacle_list:
        :return: approach_conf_list, depart_jaw_width_list
        author: weiwei, hao
        date: 20191122, 20200105, 20210113, 20210125
        """
        approach_data = self.gen_linear_approach_with_given_conf(goal_conf=goal_conf,
                                                                 direction=approach_direction,
                                                                 distance=approach_distance,
                                                                 jaw_width=approach_jaw_width,
                                                                 granularity=granularity,
                                                                 obstacle_list=obstacle_list)
        if approach_data is None:
            print("ADPlanner: Cannot generate the approach section of linear approach-depart with given conf!")
            return None
        else:
            depart_data = self.gen_linear_depart_with_given_conf(start_conf=goal_conf,
                                                                 direction=depart_direction,
                                                                 distance=depart_distance,
                                                                 jaw_width=depart_jaw_width,
                                                                 granularity=granularity,
                                                                 obstacle_list=obstacle_list)
            if depart_data is None:
                print("ADPlanner: Cannot generate the depart section of linear approach-depart with given conf!")
                return None
            else:
                return approach_data + depart_data

    def gen_approach_motion(self,
                            goal_tcp_pos,
                            goal_tcp_rotmat,
                            start_conf=None,
                            linear_direction=None,
                            linear_distance=.1,
                            jaw_width=.05,
                            granularity=.03,
                            obstacle_list=[],  # obstacles, will be checked by both rrt and linear
                            object_list=[]):  # target objects, will be checked by rrt, but not by linear
        linear_approach_data = self.gen_linear_approach(goal_tcp_pos=goal_tcp_pos,
                                                        goal_tcp_rotmat=goal_tcp_rotmat,
                                                        direction=linear_direction,
                                                        distance=linear_distance,
                                                        jaw_width=None,  # do not change jaw width
                                                        granularity=granularity,
                                                        obstacle_list=obstacle_list)
        if linear_approach_data is None:
            print("ADPlanner: Cannot gen approach linear!")
            return None
        else:
            if start_conf is None:
                return linear_approach_data
            else:
                # self.sgl_arm_robot.goto_given_conf(jnt_values=start_conf)
                # self.sgl_arm_robot.gen_meshmodel().attach_to(base)
                # object_list[0].attach_to(base)
                # object_list[0].show_cdprim()
                # # base.run()
                start2approach_conf_list = self.rrtc_planner.plan(start_conf=start_conf,
                                                                  goal_conf=linear_approach_data.conf_list[0],
                                                                  obstacle_list=obstacle_list + object_list,
                                                                  ext_dist=.1,
                                                                  max_time=100)
                if start2approach_conf_list is None:
                    print("ADPlanner: Cannot plan the motion from start_conf to the beginning of approach!")
                    return None
                else:
                    result_data = mpu.ManipulationData(sgl_arm_robot=self.sgl_arm_robot)
                    result_data.extend(start2approach_conf_list)
                    result_data.jaw_width_list[0] = jaw_width
                    return result_data + linear_approach_data

    def gen_depart_motion(self,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          end_conf=None,
                          linear_direction=None,
                          linear_distance=.1,
                          jaw_width=.05,
                          granularity=.03,
                          obstacle_list=[],  # obstacles, will be checked by both rrt and linear
                          object_list=[]):  # target objects, will be checked by rrt, but not by linear
        linear_depart_data = self.gen_linear_depart(start_tcp_pos,
                                                    start_tcp_rotmat,
                                                    linear_direction,
                                                    linear_distance,
                                                    jaw_width,
                                                    granularity,
                                                    obstacle_list)
        if linear_depart_data is None:
            print("ADPlanner: Cannot gen depart linear!")
            return None
        else:
            if end_conf is None:
                return linear_depart_data
            else:
                depart2end_conf_list = self.rrtc_planner.plan(start_conf=linear_depart_data.conf_list[-1],
                                                              goal_conf=end_conf,
                                                              obstacle_list=obstacle_list + object_list,
                                                              ext_dist=.1,
                                                              max_time=100)
                if depart2end_conf_list is None:
                    print("ADPlanner: Cannot plan depart motion!")
                    return None
                else:
                    result_data = mpu.ManipulationData(sgl_arm_robot=self.sgl_arm_robot)
                    result_data.extend(depart2end_conf_list)
                    return linear_depart_data + result_data

    def gen_approach_depart_motion(self,
                                   goal_tcp_pos,
                                   goal_tcp_rotmat,
                                   start_conf=None,
                                   end_conf=None,
                                   approach_direction=None,  # np.array([0, 0, -1])
                                   approach_distance=.1,
                                   approach_jaw_width=.05,
                                   depart_direction=None,  # np.array([0, 0, 1])
                                   depart_distance=.1,
                                   depart_jaw_width=0,
                                   granularity=.03,
                                   obstacle_list=[],  # obstacles, will be checked by both rrt and linear
                                   object_list=[]):  # target objects, will be checked by rrt, but not by linear
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param start_conf:
        :param end_conf:
        :param approach_direction:
        :param approach_distance:
        :param approach_jaw_width:
        :param depart_direction:
        :param depart_distance:
        :param depart_jaw_width:
        :param granularity:
        :param obstacle_list:
        :return:
        author: weiwei
        date: 20210113, 20210125
        """
        approach_data = self.gen_approach_motion(goal_tcp_pos=goal_tcp_pos,
                                                 goal_tcp_rotmat=goal_tcp_rotmat,
                                                 start_conf=start_conf,
                                                 linear_direction=approach_direction,
                                                 linear_distance=approach_distance,
                                                 jaw_width=approach_jaw_width,
                                                 granularity=granularity,
                                                 obstacle_list=obstacle_list,
                                                 object_list=object_list)
        if approach_data is None:
            print("ADPlanner: Cannot plan the approach section of approach-depart motion!")
            return None
        else:
            depart_data = self.gen_depart_motion_with_given_conf(start_conf=approach_data.conf_list[-1],
                                                                 end_conf=end_conf,
                                                                 linear_direction=depart_direction,
                                                                 linear_distance=depart_distance,
                                                                 jaw_width=depart_jaw_width,
                                                                 granularity=granularity,
                                                                 obstacle_list=obstacle_list,
                                                                 object_list=object_list)
            if depart_data is None:
                print("ADPlanner: Cannot plan the depart section of approach-depart motion!")
                return None
            else:
                return approach_data + depart_data

    def gen_depart_approach_motion_with_given_conf(self,
                                                   start_conf=None,
                                                   goal_conf=None,
                                                   depart_direction=None,  # np.array([0, 0, 1])
                                                   depart_distance=.1,
                                                   depart_jaw_width=0,
                                                   approach_direction=None,  # np.array([0, 0, -1])
                                                   approach_distance=.1,
                                                   approach_jaw_width=.05,
                                                   granularity=.03,
                                                   obstacle_list=[],
                                                   # obstacles, will be checked by both rrt and linear
                                                   object_list=[]):  # target objects, will be checked by rrt, but not by linear
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param start_conf:
        :param end_conf:
        :param approach_direction:
        :param approach_distance:
        :param approach_jaw_width:
        :param depart_direction:
        :param depart_distance:
        :param depart_jaw_width:
        :param granularity:
        :param obstacle_list:
        :return:
        author: weiwei
        date: 20210113, 20210125
        """
        linear_depart_data = self.gen_linear_depart_with_given_conf(start_conf=start_conf,
                                                                    direction=depart_direction,
                                                                    distance=depart_distance,
                                                                    jaw_width=depart_jaw_width,
                                                                    granularity=granularity,
                                                                    obstacle_list=obstacle_list)
        if linear_depart_data is None:
            print("ADPlanner: Cannot plan the linear depart section of depart-approach motion with given conf!")
            return None
        else:
            approach_data = self.gen_approach_motion_with_given_conf(goal_conf=goal_conf,
                                                                     start_conf=linear_depart_data.conf_list[-1],
                                                                     linear_direction=approach_direction,
                                                                     linear_distance=approach_distance,
                                                                     jaw_width=approach_jaw_width,
                                                                     granularity=granularity,
                                                                     obstacle_list=obstacle_list,
                                                                     object_list=object_list)
            if approach_data is None:
                print("ADPlanner: Cannot plan the approach section of depart-approach motion given conf!")
                return None
            else:
                return linear_depart_data + approach_data

    def gen_approach_motion_with_given_conf(self,
                                            goal_conf,
                                            start_conf=None,
                                            linear_direction=None,
                                            linear_distance=.1,
                                            jaw_width=.05,
                                            granularity=.03,
                                            obstacle_list=[],  # obstacles, will be checked by both rrt and linear
                                            object_list=[]):  # target objects, will be checked by rrt, but not by linear
        linear_approach_data = self.gen_linear_approach_with_given_conf(goal_conf=goal_conf,
                                                                        direction=linear_direction,
                                                                        distance=linear_distance,
                                                                        jaw_width=None,
                                                                        granularity=granularity,
                                                                        obstacle_list=obstacle_list)
        if linear_approach_data is None:
            print("ADPlanner: Cannot plan the linear approach section of approach with given conf!")
            return None
        else:
            if start_conf is None:
                return linear_approach_data
            else:
                startconf2approach_list = self.rrtc_planner.plan(start_conf=start_conf,
                                                                 goal_conf=linear_approach_data.conf_list[0],
                                                                 obstacle_list=obstacle_list + object_list,
                                                                 ext_dist=.1,
                                                                 max_time=100)
                if startconf2approach_list is None:
                    print("ADPlanner: Cannot plan the approach rrt motion section of approach with given conf!")
                    return None
                else:
                    result_data = mpu.ManipulationData(sgl_arm_robot=self.sgl_arm_robot)
                    result_data.extend(startconf2approach_list)
                    result_data.jaw_width_list[0] = jaw_width
                    return result_data + linear_approach_data

    def gen_depart_motion_with_given_conf(self,
                                          start_conf,
                                          end_conf=None,
                                          linear_direction=None,
                                          linear_distance=.1,
                                          jaw_width=.05,
                                          granularity=.03,
                                          obstacle_list=[],  # obstacles, will be checked by both rrt and linear
                                          object_list=[]):  # target objects, will be checked by rrt, but not by linear
        linear_depart_data = self.gen_linear_depart_with_given_conf(start_conf=start_conf,
                                                                    direction=linear_direction,
                                                                    distance=linear_distance,
                                                                    jaw_width=jaw_width,
                                                                    granularity=granularity,
                                                                    obstacle_list=obstacle_list)
        if linear_depart_data is None:
            print("ADPlanner: Cannot plan the linear depart section of depart with given conf!")
            return None
        else:
            if end_conf is None:
                return linear_depart_data
            else:
                depart2end_conf_list = self.rrtc_planner.plan(start_conf=linear_depart_data.conf_list[-1],
                                                              goal_conf=end_conf,
                                                              obstacle_list=obstacle_list + object_list,
                                                              ext_dist=.1,
                                                              max_time=100)
                if depart2end_conf_list is None:
                    print("ADPlanner: Cannot plan the depart rrt motion section of depart with given conf!")
                    return None
                else:
                    result_data = mpu.ManipulationData(sgl_arm_robot=self.sgl_arm_robot)
                    result_data.extend(depart2end_conf_list)
                    return linear_depart_data + result_data

    def gen_approach_depart_motion_with_given_conf(self,
                                                   goal_conf,
                                                   start_conf=None,
                                                   end_conf=None,
                                                   approach_direction=None,  # np.array([0, 0, -1])
                                                   approach_distance=.1,
                                                   approach_jaw_width=.05,
                                                   depart_direction=None,  # np.array([0, 0, 1])
                                                   depart_distance=.1,
                                                   depart_jaw_width=0,
                                                   granularity=.03,
                                                   obstacle_list=[],
                                                   # obstacles, will be checked by both rrt and linear
                                                   object_list=[]):  # target objects, will be checked by rrt, but not by linear
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param start_conf:
        :param end_conf:
        :param approach_direction:
        :param approach_distance:
        :param approach_jaw_width:
        :param depart_direction:
        :param depart_distance:
        :param depart_jaw_width:
        :param granularity:
        :param obstacle_list:
        :return:
        author: weiwei
        date: 20210113, 20210125
        """
        approach_data = self.gen_approach_motion_with_given_conf(goal_conf=goal_conf,
                                                                 start_conf=start_conf,
                                                                 linear_direction=approach_direction,
                                                                 linear_distance=approach_distance,
                                                                 jaw_width=approach_jaw_width,
                                                                 granularity=granularity,
                                                                 obstacle_list=obstacle_list,
                                                                 object_list=object_list)
        if approach_data is None:
            print("ADPlanner: Cannot plan the approach section of approach-depart motion with given conf!")
            return None
        else:
            depart_data = self.gen_depart_motion_with_given_conf(start_conf=approach_data.conf_list[-1],
                                                                 end_conf=end_conf,
                                                                 linear_direction=depart_direction,
                                                                 linear_distance=depart_distance,
                                                                 jaw_width=depart_jaw_width,
                                                                 granularity=granularity,
                                                                 obstacle_list=obstacle_list,
                                                                 object_list=object_list)
            if depart_data is None:
                print("ADPlanner: Cannot plan the depart section of approach-depart motion with given conf!")
                return None
            else:
                return approach_data + depart_data


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import robot_sim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)
    robot = ym.Yumi(enable_cc=True)
    goal_pos = np.array([.65, -.1, .3])
    goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    jnt_values = robot.rgt_arm.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
    # robot.rgt_arm.goto_given_conf(jnt_values=jnt_values)
    # robot.gen_meshmodel().attach_to(base)
    # base.run()

    sgl_arm = robot.rgt_arm
    adp = ADPlanner(sgl_arm)
    tic = time.time()
    # approach_motion_data = adp.gen_approach_motion(goal_pos,
    #                                                goal_rotmat,
    #                                                start_conf=robot.get_jnt_values(),
    #                                                linear_direction=np.array([0, 0, -1]),
    #                                                linear_distance=.1)
    # depart_motion_data = adp.gen_depart_motion(goal_pos,
    #                                            goal_rotmat,
    #                                            end_conf=robot.get_jnt_values(),
    #                                            linear_direction=np.array([0, 0, -1]),
    #                                            linear_distance=.1)
    # ad_motion_data = adp.gen_approach_depart_motion(goal_tcp_pos=goal_pos,
    #                                                 goal_tcp_rotmat=goal_rotmat,
    #                                                 start_conf=robot.get_jnt_values(),
    #                                                 end_conf=robot.get_jnt_values())
    ad_data = adp.gen_depart_approach_motion_with_given_conf(start_conf=jnt_values,
                                                                    goal_conf=jnt_values)


    class Data(object):
        def __init__(self, robot, arm, motion_data):
            self.robot_attached_list = []
            self.counter = 0
            # self.motion_data = approach_motion_data + depart_motion_data
            self.motion_data = motion_data
            self.robot = robot
            self.arm = arm


    anime_data = Data(robot, sgl_arm, ad_motion_data)


    def update(anime_data, task):
        if anime_data.counter >= len(anime_data.motion_data):
            if len(anime_data.robot_attached_list) != 0:
                for robot_attached in anime_data.robot_attached_list:
                    robot_attached.detach()
            anime_data.robot_attached_list.clear()
            anime_data.counter = 0
        if len(anime_data.robot_attached_list) > 1:
            for robot_attached in anime_data.robot_attached_list:
                robot_attached.detach()
        conf = anime_data.motion_data.conf_list[anime_data.counter]
        jaw_width = anime_data.motion_data.jaw_width_list[anime_data.counter]
        anime_data.arm.goto_given_conf(jnt_values=conf)
        if jaw_width is not None:
            anime_data.arm.change_jaw_width(jaw_width=jaw_width)
        robot_meshmodel = anime_data.robot.gen_meshmodel(toggle_cdprim=False, alpha=1)
        robot_meshmodel.attach_to(base)
        anime_data.robot_attached_list.append(robot_meshmodel)
        anime_data.counter += 1
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)
    base.run()

    # for i, jnt_values in enumerate(conf_list):
    #     robot.goto_given_conf(jnt_values)
    #     robot.change_jaw_width(jaw_width_list[i])
    #     robot.gen_meshmodel().attach_to(base)
    #     # robot.show_cdprim()
    # base.run()
