"""
Created on 2024/6/12 
Author: Hao Chen (chen960216@gmail.com)
"""
import numpy as np

if __name__ == '__main__':
    import robot_sim.end_effectors.gripper.robotiqhe.robotiqhe as rqh
    import robot_sim.robots.ur3e_dual.ur3e_dual as ur3ed
    import grasping.planning.antipodal as gpa
    import visualization.panda.world as wd
    import regrasp.utils.file_sys as fs
    import modeling.geometric_model as gm
    import modeling.collision_model as cm
    from regrasp.handover import Handover
    from regrasp.utils.visualize import visualize_grasps

    base = wd.World(cam_pos=[3, 0, 4], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    obj_name = 'lshape.stl'
    if not isinstance(obj_name, str) and not obj_name.endswith('.stl') and not obj_name.endswith('.dae'):
        raise Exception(
            f"{obj_name} is not a valid object name! Please check the name of the object! It should end with .stl or .dae!")
    # examine if the obj_name is in the model folder
    obj_path = fs.workdir_model.joinpath(obj_name)
    if not obj_path.exists():
        raise Exception(f"{obj_path} does not exist! Please check the name of the object!")
    obj_cm = cm.CollisionModel(str(obj_path))
    gripper_s = rqh.RobotiqHE()
    rgt_hnd = gripper_s.copy()
    lft_hnd = gripper_s.copy()

    rbt = ur3ed.UR3EDual(enable_cc=True)

    obj_name_no_ext = obj_name.split('.')[0]
    grasp_info_list = gpa.load_pickle_file(obj_name_no_ext, str(fs.workdir_data), file_name='robotiqhe_grasps.pickle')

    hmstr = Handover(obj_cm=obj_cm,
                     rgt_hnd=rgt_hnd,
                     lft_hnd=lft_hnd,
                     grasp_info_list_rgt=grasp_info_list,
                     grasp_info_list_lft=grasp_info_list,
                     rbt=rbt)
    hmstr._gen_fp_list([np.array([0.700, 0, 1.300])], toggle_debug=True)