"""
Created on 2024/8/7 
Author: Hao Chen (chen960216@gmail.com)
"""
if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    from regrasp.utils import *

    base = wd.World(cam_pos=[3, 0, 4], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)

    THIS_DIR = Path(__file__).resolve().parent
    DATA_PATH = THIS_DIR.parent.joinpath('data/data_handover_hndovrinfo.pickle')
    hndovr_data = load_handover_data(DATA_PATH)

    visualize_grasps_fp(grasp_info_fid_gp=hndovr_data.grasp_list_fp_rgt,
                        fp_list=hndovr_data.floating_pose_list,
                        obj=get_obj_cm(hndovr_data.obj_name),
                        gripper=get_gripper_by_name('robotiqhe'),
                        base=base, )
    base.run()
