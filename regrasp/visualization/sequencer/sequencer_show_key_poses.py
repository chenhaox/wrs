"""
Created on 9/25/2024
Author: Hao Chen (chen960216@gmail.com)
"""
"""
Created on 9/9/2024 
Author: Hao Chen (chen960216@gmail.com)
"""
if __name__ == "__main__":
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    from regrasp.utils import *
    from regrasp.sequencer import Sequencer
    import basis.robot_math as rm

    base = wd.World(cam_pos=[3, 0, 4], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    THIS_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = THIS_DIR.parent.joinpath('data/data_handover_hndovrinfo.pickle')
    hndovr_data = load_handover_data(DATA_PATH)
    # load robots and objects
    robot = get_rbt_by_name("ur3e")
    obj = get_obj_cm_by_name(hndovr_data.obj_name)

    # robot.gen_meshmodel().attach_to(base)
    # obj at initial pose
    obj_init = obj.copy()
    init_pose = rm.homomat_from_posrot(np.array([0.6, .3, .78]), np.eye(3))
    obj_init.set_homomat(init_pose)
    obj_init.set_rgba([0.2, 0.6, 0.8, 1.0])
    obj_init.attach_to(base)
    # obj at goal pose
    obj_goal = obj.copy()
    goal_pose = rm.homomat_from_posrot(np.array([0.5, .3, .78]), np.eye(3))
    obj_goal.set_homomat(goal_pose)
    obj_goal.set_rgba([0.0, 0.8, 0.4, .5])
    obj_goal.attach_to(base)

    # obstacle
    DATA_PATH = THIS_DIR.parent.joinpath("model/bunnysim.stl")
    obs_obj = cm.CollisionModel(str(DATA_PATH))
    obs_obj_pose = rm.homomat_from_posrot(np.array([0.6, .3, .78]), np.eye(3))
    obs_obj.set_homomat(obs_obj_pose)
    # set obs_obj as a beautiful color
    obs_obj.attach_to(base)

    # add sequencer
    seq = Sequencer(robot,
                    handover_data=hndovr_data, )
    seq.add_start_goal(start_homomat=init_pose,
                       goal_homomat=goal_pose,
                       choice="startlftgoallft",
                       starttmpobstacle=[])
    seq.update_shortest_path()
    seq.plot_regrasp_graph()
    traj_data, extended_path_nid_list, path_nid_list = seq.get_motion_sequence(0, type="OO")
    print(path_nid_list)
    print(extended_path_nid_list)
    visualize_traj_data(traj_data, obj.copy(), robot, base, )
    base.run()
