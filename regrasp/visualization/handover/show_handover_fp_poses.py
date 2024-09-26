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

    THIS_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = THIS_DIR.parent.joinpath('data/data_handover_hndovrinfo.pickle')
    hndovr_data = load_handover_data(DATA_PATH)

    visualize_handover(grasp_pairs_fp=hndovr_data.grasp_pairs_fp,
                       fp_list=hndovr_data.floating_pose_list,
                       ikjnt_fp_lft=hndovr_data.ik_jnts_fp_gid_lft,
                       ikjnt_fp_rgt=hndovr_data.ik_jnts_fp_gid_rgt,
                       rbt=get_rbt_by_name('ur3e'),
                       obj=get_obj_cm_by_name(hndovr_data.obj_name),
                       base=base, )
    base.run()
