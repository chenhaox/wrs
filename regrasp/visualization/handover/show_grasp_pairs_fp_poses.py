"""
Created on 2024/8/9 
Author: Hao Chen (chen960216@gmail.com)
"""
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

    get_obj_cm_by_name(hndovr_data.obj_name).attach_to(base)
    grasp_pairs = []
    for g_r, g_l in hndovr_data.identity_grasp_pairs:
        grasp_pairs.append((hndovr_data.grasp_list_rgt[g_r], hndovr_data.grasp_list_lft[g_l]))
    visualize_grasp_pairs(grasp_pairs=grasp_pairs,
                          rgt_gripper=get_gripper_by_name('robotiqhe'),
                          lft_gripper=get_gripper_by_name('robotiqhe'),
                          base=base, )
    base.run()
