from yanpu_pnp import planner


class AnimationData:

    def __init__(self, robot, robot_mesh_list, frame_list, payload_dict, task_dict):
        self.robot = robot
        self.robot_mesh_list = robot_mesh_list
        self.frame_list = frame_list
        self.payload_dict = payload_dict
        self.task_dict = task_dict
        self.counter = 0
        self.current_robot_mesh = None
        self.end_hold_counter = 0


def precompute_robot_meshes(robot, frame_list):
    mesh_list = []
    robot.backup_state()
    for frame in frame_list:
        planner.apply_frame_state(robot, frame)
        mesh_list.append(robot.gen_meshmodel(alpha=.88, toggle_tcp_frame=True))
    robot.restore_state()
    return mesh_list


def make_update_task(base, animation_data):

    def update(panda_task):
        frame = animation_data.frame_list[animation_data.counter]
        if animation_data.current_robot_mesh is not None:
            animation_data.current_robot_mesh.detach()
        animation_data.current_robot_mesh = animation_data.robot_mesh_list[animation_data.counter]
        animation_data.current_robot_mesh.attach_to(base)

        planner.apply_frame_state(animation_data.robot, frame)
        planner.apply_payload_state(animation_data.robot,
                                    frame,
                                    animation_data.payload_dict,
                                    animation_data.task_dict)

        if animation_data.counter == len(animation_data.frame_list) - 1:
            animation_data.end_hold_counter += 1
            if animation_data.end_hold_counter >= 24:
                animation_data.counter = 0
                animation_data.end_hold_counter = 0
        else:
            animation_data.counter += 1
        return panda_task.again

    return update


def play(base, robot, frame_list, payload_dict, task_dict):
    robot_mesh_list = precompute_robot_meshes(robot, frame_list)
    animation_data = AnimationData(robot=robot,
                                   robot_mesh_list=robot_mesh_list,
                                   frame_list=frame_list,
                                   payload_dict=payload_dict,
                                   task_dict=task_dict)
    planner.apply_frame_state(robot, frame_list[0])
    planner.apply_payload_state(robot, frame_list[0], payload_dict, task_dict)
    base.taskMgr.doMethodLater(.05,
                               make_update_task(base, animation_data),
                               "yanpu_pnp_update",
                               appendTask=True)
    base.run()

