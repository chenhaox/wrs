import os  #文件路径/标准库
import math
import numpy as np
import modeling.model_collection as mc  #自定义模型集合简称mc
import modeling.collision_model as cm   #自定义碰撞检测模型简称cm
from panda3d.core import CollisionNode, CollisionBox, Point3
import robot_sim._kinematics.jlchain as jl   #自定义关节模块
import basis.robot_math as rm   #自定义机器人工具包
import robot_sim.end_effectors.gripper.gripper_interface as gp   #自定义末端夹爪类
# import trimesh

class Dh50(gp.GripperInterface):   #模型name

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), coupling_offset_pos=np.zeros(3),   #主要对其位置，旋转矩阵等定义，并引入碰撞检测类
                 coupling_offset_rotmat=np.eye(3), cdmesh_type='convex_hull', name='Dh50',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.coupling.jnts[1]['loc_pos'] = coupling_offset_pos   #耦合装置coupling，jints关节位置和旋转矩阵，连杆lnks碰撞检测模型
        self.coupling.jnts[1]['loc_rotmat'] = coupling_offset_rotmat
        self.coupling.lnks[0]['collision_model'] = cm.gen_stick(self.coupling.jnts[0]['loc_pos'],
                                                                self.coupling.jnts[1]['loc_pos'],
                                                                thickness=.07, rgba=[.2, .2, .2, 1],
                                                                sections=24)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']

        #dh50
        self.lft = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='Dh50')    #创建一个lft的属性类
        self.lft.lnks[0]['name'] = "base"   #第一个关节的名字叫 base，及dh50机构本身
        self.lft.lnks[0]['loc_pos'] = np.zeros(3)   #链的局部位置
        self.lft.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "Dh50_base.stl")   #文件路径
        # mesh = trimesh.load("meshes/Dh50_base.stl")
        # mesh.apply_scale(0.001)  # mm → m，目前sw中stl的保存单位仍为m，理论上能转成功，但是不行，mesh scale的代码目前无用
        self.lft.lnks[0]['rgba'] = [.3, .3, .3, 1]   #颜色，分别代表红绿蓝透明度，透明度1表示完全不透明

        #self.base = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='Dh50')    #创建一个base的属性类
        #self.base.lnks[0]['name'] = "base"   #第一个关节的名字叫 base，及dh50机构本身
        #self.base.lnks[0]['loc_pos'] = np.zeros(3)   #链的局部位置
        #self.base.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "Dh50_base.stl")   #文件路径
        #self.base.lnks[0]['rgba'] = [.2, .2, .2, 1]   #颜色，分别代表红绿蓝透明度，透明度1表示完全不透明

        #self.lft = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='left')   #创建一个lft的属性类
        self.lft.jnts[1]['loc_pos'] = np.array([0, -.015, .106])   #访问其第二个关节`,相对于定义[0,0,0]来说的！！！！！！！！！！！！！
        # gm.gen_frame(self.lft.jnts[1]['loc_pos']).attach_to(base)
        self.lft.jnts[1]['type'] = 'prismatic'   #该关节类型为平移关节
        self.lft.jnts[1]['motion_rng'] = [0, .030]   #关节活动范围为[0-0.030]单位m
        self.lft.jnts[1]['loc_motionax'] = np.array([0, -1, 0])   #该关节的运动轴为y，及在y方向上移动
        self.lft.lnks[1]['name'] = "lft_finger"
        self.lft.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "shell assembly_lft.stl")
        self.lft.lnks[1]['rgba'] = [.5, .5, .5, 1]

        # rgt
        self.rgt = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='right')   #创建一个rgt的属性类
        self.rgt.jnts[1]['loc_pos'] = np.array([0, .015, .106])
        self.rgt.jnts[1]['type'] = 'prismatic'
        self.rgt.jnts[1]['loc_motionax'] = np.array([0, -1, 0]) # 该关节的运动轴为y，及在y方向上移动
        self.rgt.lnks[1]['name'] = "rgt_finger"
        self.rgt.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "shell assembly_rgt.stl")
        self.rgt.lnks[1]['rgba'] = [.8, .8, .8, 1]   #充电线靠左，右边为rgt
        self.rgt.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, np.pi)  # 绕 Z 轴旋转 180°！！！！！！！！！
                                                                            # 通过欧拉角创建旋转对象 [α, β, γ]：绕各轴的旋转角度（单位默认为弧度，degrees=True 时可用角度制）。
                                                                            #degrees：若为 True，输入角度为度数（如 90）；若为 False（默认），输入为弧度（如 np.pi/2）
                                                                            # rotation = Rotation.from_euler('xyz', [α, β, γ], degrees=True)

        # jaw center
        self.jaw_center_pos = np.array([0, .0, 0.106]) + coupling_offset_pos     # 设置夹爪中心位置（Y轴偏移30mm）并加上耦合偏移量
        # reinitialize，用于重新计算左右夹爪的运动学参数和几何状态，通常在修改关节参数后必须调用
        self.lft.reinitialize()    # 重新初始化左侧夹爪
        self.rgt.reinitialize()    # 重新初始化右侧夹爪
        # collision detection
        self.all_cdelements = []     # 初始化碰撞检测元素列表
        self.enable_cc(toggle_cdprimit=enable_cc)   # 启用碰撞检测功能（根据参数决定是否使用基本碰撞体）
        # jaw width
        self.jawwidth_rng = [0, .06]


    @staticmethod
    def _finger_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(-.0035, 0.004, .025 + .003),
                                              x=.0035 + radius, y=0.0032 + radius, z=.025 + .003 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(.008, 0.028 - .002, -.011),
                                              x=.018 + radius, y=0.008 + radius, z=.011 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(-.005, 0.012 - .002, -.002 + .0025),
                                              x=.005 + radius, y=0.008 + radius, z=.002 + .0025 + radius)
        collision_node.addSolid(collision_primitive_c2)
        return collision_node

    @staticmethod
    def _hnd_base_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0, 0, .031),
                                              x=.036 + radius, y=0.038 + radius, z=.031 + radius)
        collision_node.addSolid(collision_primitive_c0)  # 0.62
        collision_primitive_c1 = CollisionBox(Point3(0, 0, .067),
                                              x=.036 + radius, y=0.027 + radius, z=.003 + radius)
        collision_node.addSolid(collision_primitive_c1)  # 0.06700000
        #
        collision_primitive_c2 = CollisionBox(Point3(.006, .049, .0485),
                                              x=.02 + radius, y=.02 + radius, z=.015 + radius)
        collision_node.addSolid(collision_primitive_c2)
        collision_primitive_c3 = CollisionBox(Point3(0, 0, .08),
                                              x=.013 + radius, y=0.013 + radius, z=.005 + radius)
        collision_node.addSolid(collision_primitive_c3)

        return collision_node

    def enable_cc(self, toggle_cdprimit):
        if toggle_cdprimit:
            super().enable_cc()
            # cdprimit
            self.cc.add_cdlnks(self.lft, [0])
            self.cc.add_cdlnks(self.rgt, [1])
            activelist = [self.lft.lnks[0],
                          self.rgt.lnks[1]]
            self.cc.set_active_cdlnks(activelist)
            self.all_cdelements = self.cc.all_cdelements
        else:
            self.all_cdelements = [self.lft.lnks[0],
                                   self.rgt.lnks[1]]
        # cdmesh
        for cdelement in self.all_cdelements:
            cdmesh = cdelement['collision_model'].copy()
            self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: motion_val, meter or radian
        """
        if  self.lft.jnts[1]['motion_rng'][0] <= motion_val <= self.lft.jnts[1]['motion_rng'][1]:
            self.lft.jnts[1]['motion_val'] = motion_val
            self.rgt.jnts[1]['motion_val'] = self.lft.jnts[1]['motion_val']
            self.lft.fk()
            self.rgt.fk()
        else:
            raise ValueError("The motion_val parameter is out of range!")

    def jaw_to(self, jaw_width):
        if jaw_width > self.jawwidth_rng[1]:   #self.jawwidth_rng = [最小宽度, 最大宽度]
            raise ValueError("The jaw_width parameter is out of range!")
        self.fk(motion_val=jaw_width / 2.0)

    def get_jawwidth(self):
        return self.lft.jnts[1]['motion_val'] * 2

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='lite6wrs_gripper_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.body.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                 tcp_loc_pos=tcp_loc_pos,
                                 tcp_loc_rotmat=tcp_loc_rotmat,
                                 toggle_tcpcs=False,
                                 toggle_jntscs=toggle_jntscs,
                                 toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.lft.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt.gen_stickmodel(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(stickmodel)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(stickmodel)

        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xc330gripper'):
        meshmodel = mc.ModelCollection(name=name)
        # self.coupling.gen_meshmodel(tcp_loc_pos=None,
        #                             tcp_loc_rotmat=None,
        #                             toggle_tcpcs=False,
        #                             toggle_jntscs=toggle_jntscs,
        #                             rgba=rgba).attach_to(meshmodel)
        self.lft.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.rgt.gen_meshmodel(tcp_loc_pos=None,
                               tcp_loc_rotmat=None,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        if toggle_tcpcs:
            jaw_center_gl_pos = self.rotmat.dot(self.jaw_center_pos) + self.pos
            jaw_center_gl_rotmat = self.rotmat.dot(self.jaw_center_rotmat)
            gm.gen_dashstick(spos=self.pos,
                             epos=jaw_center_gl_pos,
                             thickness=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(meshmodel)
            gm.gen_mycframe(pos=jaw_center_gl_pos, rotmat=jaw_center_gl_rotmat).attach_to(meshmodel)
        return meshmodel

    def open(self):
        '''
        gripper open
        '''
        self.jaw_to(.06)

    def close(self):
        '''
        gripper close
        '''
        self.jaw_to(0)


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
    gm.gen_frame().attach_to(base)
    # cm.CollisionModel("meshes/dual_realsense.stl", expand_radius=.001).attach_to(base)
    grpr = Dh50(enable_cc=True)
    # grpr.gen_meshmodel().attach_to(base)
    # grpr.open()
    # grpr.gen_meshmodel(toggle_tcpcs=True,toggle_jntscs=True).attach_to(base)
    grpr.close()
    grpr.gen_meshmodel(toggle_tcpcs=True,toggle_jntscs=True).attach_to(base)
    # grpr.jaw_to(0.01)
    # grpr.gen_meshmodel(toggle_tcpcs=True,toggle_jntscs=True).attach_to(base)
    base.run()