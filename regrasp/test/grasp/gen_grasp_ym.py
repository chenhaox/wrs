""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20240218osaka

"""

if __name__ == '__main__':
    import math
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import modeling.collision_model as cm
    import grasping.planning.antipodal as gpa
    import regrasp.utils.file_sys as fs
    import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
    from regrasp.utils.visualize import visualize_grasps

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # object
    obj_name = 'bunnysim.stl'
    # examine obj_name
    if not isinstance(obj_name, str) and not obj_name.endswith('.stl') and not obj_name.endswith('.dae'):
        raise Exception(
            f"{obj_name} is not a valid object name! Please check the name of the object! It should end with .stl or .dae!")
    # examine if the obj_name is in the model folder
    obj_path = fs.workdir_model.joinpath(obj_name)
    if not obj_path.exists():
        raise Exception(f"{obj_path} does not exist! Please check the name of the object!")
    obj_cm = cm.CollisionModel(str(obj_path))
    obj_cm.set_rgba([.9, .75, .35, .3])
    obj_cm.attach_to(base)
    # hnd_s
    gripper_s = yg.YumiGripper()
    grasp_info_list = gpa.plan_grasps(gripper_s, obj_cm,
                                      angle_between_contact_normals=math.radians(177),
                                      openning_direction='loc_x',
                                      max_samples=15, min_dist_between_sampled_contact_points=.005,
                                      contact_offset=.005)
    obj_name_no_ext = obj_name.split('.')[0]
    gpa.write_pickle_file(obj_name_no_ext, grasp_info_list, root=str(fs.workdir_data), file_name='yumi_gripper.pickle')
    visualize_grasps(grasp_info_list, gripper_s, base)
    base.run()
