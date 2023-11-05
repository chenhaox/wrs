import math
import numpy as np
import robot_sim.kinematics.jlchain as jl
import robot_sim.robots.system_interface as ri

class XYBot(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='XYBot'):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        self.jlc = jl.JLChain(home_conf=np.zeros(2), name='XYBot')
        self.jlc.joints[1]['end_type'] = 'prismatic'
        self.jlc.joints[1]['loc_motionax'] = np.array([1, 0, 0])
        self.jlc.joints[1]['pos_in_loc_tcp'] = np.zeros(3)
        self.jlc.joints[1]['motion_rng'] = [-2.0, 15.0]
        self.jlc.joints[2]['end_type'] = 'prismatic'
        self.jlc.joints[2]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.joints[2]['pos_in_loc_tcp'] = np.zeros(3)
        self.jlc.joints[2]['motion_rng'] = [-2.0, 15.0]
        self.jlc.reinitialize()

    def fk(self, jnt_values=np.zeros(2)):
        self.jlc.fk(jnt_values)

    def rand_conf(self):
        return self.jlc.rand_conf()

    def get_jnt_values(self):
        return self.jlc.get_joint_values()

    def is_jnt_values_in_ranges(self, jnt_values):
        return self.jlc.are_joint_values_in_ranges(jnt_values)

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        for (obpos, size) in obstacle_list:
            dist = np.linalg.norm(np.asarray(obpos) - self.get_jnt_values())
            if dist <= size / 2.0:
                return True  # collision
        return False  # safe


class XYTBot(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='TwoWheelCarBot'):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        self.jlc = jl.JLChain(home_conf=np.zeros(3), name='XYBot')
        self.jlc.joints[1]['end_type'] = 'prismatic'
        self.jlc.joints[1]['loc_motionax'] = np.array([1, 0, 0])
        self.jlc.joints[1]['pos_in_loc_tcp'] = np.zeros(3)
        self.jlc.joints[1]['motion_rng'] = [-2.0, 15.0]
        self.jlc.joints[2]['end_type'] = 'prismatic'
        self.jlc.joints[2]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.joints[2]['pos_in_loc_tcp'] = np.zeros(3)
        self.jlc.joints[2]['motion_rng'] = [-2.0, 15.0]
        self.jlc.joints[3]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.joints[3]['pos_in_loc_tcp'] = np.zeros(3)
        self.jlc.joints[3]['motion_rng'] = [-math.pi, math.pi]
        self.jlc.reinitialize()

    def fk(self, component_name='all', jnt_values=np.zeros(3)):
        if component_name != 'all':
            raise ValueError("Only support hnd_name == 'all'!")
        self.jlc.fk(jnt_values)

    def rand_conf(self, component_name='all'):
        if component_name != 'all':
            raise ValueError("Only support hnd_name == 'all'!")
        return self.jlc.rand_conf()

    def get_jntvalues(self, component_name='all'):
        if component_name != 'all':
            raise ValueError("Only support hnd_name == 'all'!")
        return self.jlc.get_joint_values()

    def is_jnt_values_in_ranges(self, component_name, jnt_values):
        if component_name != 'all':
            raise ValueError("Only support hnd_name == 'all'!")
        return self.jlc.are_joint_values_in_ranges(jnt_values)

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        for (obpos, size) in obstacle_list:
            dist = np.linalg.norm(np.asarray(obpos) - self.get_jntvalues()[:2])
            if dist <= size / 2.0:
                return True  # collision
        return False  # safe