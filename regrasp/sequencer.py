import itertools
import math
from typing import List
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import motion.probabilistic.rrt_connect as rrtc
import modeling.collision_model as cm
from regrasp.utils import *
import basis.robot_math as rm
from robot_sim.robots.robot_interface import RobotInterface
import pickle
import copy

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.joinpath('data')


class Sequencer(object):
    def __init__(self,
                 rbt: RobotInterface,
                 handover_data: HandoverData,
                 obs_list: Optional[List[cm.CollisionModel]] = None,
                 gobacktoinitafterplanning=False,
                 toogle_graph_debug=False,
                 ret_dis=0.1,
                 debug=False,
                 inspector=None):
        if obs_list is None:
            obs_list = []
        self.rbt = rbt
        self.inspector = inspector
        self.handover_data = handover_data
        self.obstacle_list = obs_list
        self.gobacktoinitafterplanning = gobacktoinitafterplanning

        # start and goal
        self.start_rgt_node_ids = []  # start nodes rgt arm
        self.start_lft_node_ids = []  # start nodes lft arm
        self.goal_rgt_node_ids = []  # goal nodes rgt arm
        self.goal_lft_node_ids = []  # goal nodes lft arm
        self.shortest_paths = None  # shortest path

        # retract distance
        self.ret_dis = ret_dis  # retract  distance
        if not DATA_DIR.joinpath("regrasp_graph").exists():
            DATA_DIR.joinpath("regrasp_graph").mkdir(parents=True, exist_ok=True)
        # build the graph
        if DATA_DIR.joinpath("regrasp_graph", f"graph_data_{handover_data.obj_name}.pkl").exists():
            print(f"Regrasp graph for the object<{handover_data.obj_name}> exits.")
            self.regg = load_pickle(DATA_DIR.joinpath("regrasp_graph", f"graph_data_{handover_data.obj_name}.pkl"))
            print(f"Load regrasp graph successfully ....")
        else:
            print(f"Regrasp graph for the object<{handover_data.obj_name}> does not exist.")
            self.regg = nx.Graph()  # graph
            self.build_graph(armname="rgt_arm", handover_data=handover_data)  # add the handover grasps of the rgt arm
            self.build_graph(armname="lft_arm", handover_data=handover_data)  # add the handover grasps of the left arm
            self.bridge_graph(handover_data=handover_data)  # connect the handover grasps of the left arm and rgt arm
            # dump the graph
            dump_pickle(self.regg, DATA_DIR.joinpath("regrasp_graph", f"graph_data_{handover_data.obj_name}.pkl"))
            print(f"Dump regrasp graph successfully ....")
        self.reggbk = copy.deepcopy(self.regg)  # backup of the graph: restore the handover nodes
        # shortest_paths
        self.directshortestpaths_startrgtgoalrgt = []
        self.directshortestpaths_startrgtgoallft = []
        self.directshortestpaths_startlftgoalrgt = []
        self.directshortestpaths_startlftgoallft = []
        self.direct_shortest_paths = []
        self.debug = debug

    def reset(self):
        self.regg = copy.deepcopy(self.reggbk)
        # shortest_paths
        self.directshortestpaths_startrgtgoalrgt = []
        self.directshortestpaths_startrgtgoallft = []
        self.directshortestpaths_startlftgoalrgt = []
        self.directshortestpaths_startlftgoallft = []
        self.direct_shortest_paths = []

        self.start_rgt_node_ids = []
        self.start_lft_node_ids = []
        self.goal_rgt_node_ids = []
        self.goal_lft_node_ids = []
        self.shortest_paths = None

        self.start_homomat = None
        self.goal_homomat = None

    def build_graph(self, armname: Literal['rgt_arm', 'lft_arm'], handover_data: HandoverData):
        if armname == "rgt_arm":
            ik_feasible_jnts = handover_data.ik_jnts_fp_gid_rgt
            fpglist = handover_data.grasp_list_fp_rgt
            manipuablity = handover_data.manipulability_fp_gid_rgt
            cond = 'rgt'
        elif armname == "lft_arm":
            ik_feasible_jnts = handover_data.ik_jnts_fp_gid_lft
            fpglist = handover_data.grasp_list_fp_lft
            manipuablity = handover_data.manipulability_fp_gid_lft
            cond = 'lft'
        else:
            raise ValueError("Invalid armname")
        globalidsedges = {}  # the global id of the edges?
        # handpairList: possible handpairList
        for fpid in range(len(handover_data.floating_pose_list)):  # iterate the handover positions
            added_node_list = []
            if fpid not in handover_data.grasp_pairs_fp:  # {posid: possible handover pair [rgt hand grasp id, left hand grasp id]}
                continue
            for pair_id, (rgt_id, lft_id) in enumerate(
                    handover_data.grasp_pairs_fp[fpid]):  # i0 is the rgt hand grasp id
                if armname == "rgt_arm":
                    graspid = rgt_id
                else:
                    graspid = lft_id
                if graspid in added_node_list:
                    continue
                added_node_list.append(graspid)
                fp_grasp_rot4 = fpglist[fpid][graspid][2]  # grasp mat4
                fp_grasp_rot3 = fp_grasp_rot4[:3, :3]
                fp_gid = graspid  # floating pose grasp id
                approach_dir = - fpglist[fpid][graspid][3]  # negative of z (negative direction of the hand)
                fp_grasp_center = fpglist[fpid][graspid][1]  # center of the floating grasp
                fp_grasp_center_approach = fp_grasp_center + approach_dir * handover_data.retract_dist  # the place that is negative direction of along of the center of the floating grasp
                jaw_width = fpglist[fpid][graspid][0]  # jawidth
                fp_jnts = np.array([_[0] for _ in ik_feasible_jnts[fpid][graspid]])  # floating pose jnts
                fp_jnts_approach = np.array([_[1] for _ in ik_feasible_jnts[fpid][graspid]])
                fp_manipuablity = np.asarray([_[0] for _ in np.array(manipuablity[fpid][graspid])])
                fp_manipuablity_approach = np.asarray([_[1] for _ in np.array(manipuablity[fpid][graspid])])
                # manipulability
                # add into graph
                node_name = 'ho' + cond + str(fp_gid) + 'pos' + str(fpid)
                node_id = cond + str(fp_gid)
                self.regg.add_node(node_name, fgrcenter=fp_grasp_center,
                                   fgrcenterhanda=fp_grasp_center_approach, jawwidth=jaw_width,
                                   hndrotmat3np=fp_grasp_rot3,
                                   armjnts=fp_jnts,
                                   armjntshanda=fp_jnts_approach,
                                   floatingposegrippairind=pair_id,
                                   handoverposid=fpid,
                                   identity=node_id,
                                   fp_manipulability=fp_manipuablity,
                                   fp_manipuablity_approach=fp_manipuablity_approach,
                                   node_type=f'handover_{cond}'
                                   )
                if node_id not in globalidsedges:  # {armname+fpgid: }
                    globalidsedges[node_id] = []
                globalidsedges[node_id].append(node_name)
        for globalidedgesid in globalidsedges:
            if len(globalidsedges[globalidedgesid]) == 1:
                continue
            for edge in list(itertools.combinations(globalidsedges[globalidedgesid], 2)):
                self.regg.add_edge(*edge, weight=1, edgetype='transfer')

    def bridge_graph(self, handover_data: HandoverData):
        for posid, objrotmat4 in enumerate(handover_data.floating_pose_list):
            pass
            if posid not in handover_data.grasp_pairs_fp:
                continue
            for rgt_gid, lft_gid in handover_data.grasp_pairs_fp[posid]:
                rgtnode_name = 'horgt' + str(rgt_gid) + "pos" + str(posid)
                lftnode_name = 'holft' + str(lft_gid) + "pos" + str(posid)
                self.regg.add_edge(rgtnode_name, lftnode_name, weight=1, edgetype='handovertransit')
                # TODO Implement the Error Info Examination
                if self.inspector is not None:
                    self.inspector.add_error("handover", Error_info(
                        name='horgt' + str(rgt_gid) + "pos" + str(posid) + "---" + 'holft' + str(lft_gid) + "pos" + str(
                            posid),
                        lftarmjnts=self.regg.nodes['holft' + str(lft_gid) + "pos" + str(posid)]['armjnts'],
                        rgtarmjnts=self.regg.nodes['horgt' + str(rgt_gid) + "pos" + str(posid)]['armjnts'],
                        lftjawwidth=self.regg.nodes['holft' + str(lft_gid) + "pos" + str(posid)]['jawwidth'],
                        rgtjawwidth=self.regg.nodes['horgt' + str(rgt_gid) + "pos" + str(posid)]['jawwidth']), )

    def add_start_goal(self,
                       start_homomat: np.ndarray,
                       goal_homomat: np.ndarray,
                       choice: Literal["startrgtgoallft", "startrgtgoalrgt", "startlftgoalrgt", "startrgtgoallft"],
                       starttoolvec: Optional[np.ndarray] = None,
                       goaltoolvec: Optional[np.ndarray] = None,
                       possiblegrasp=None,
                       starttmpobstacle=[],
                       goaltmpobstacle=[]):
        """
        add start and goal to the grasph
        if start/goalgrasppose is not None, the only pose will be used
        the pose is defined by a numpy 4x4 homomatrix

        :param start_homomat: numpy matrix
        :param goal_homomat: numpy matrix
        :param choice in "startrgtgoallft" "startrgtgoalrgt" "startlftgoalrgt" "startrgtgoallft"
        :param startgraspgid:
        :param goalgraspgid:
        :param starttoolvec
        :param goaltoolvec there are three choices for the tool vecs: None indicates global z, [0,0,0] indicates no tool vec
        :return:

        author: weiwei
        date: 20180925
        """
        # self.grasp = possiblegrasp

        if starttoolvec is not None:
            starttoolVec3 = starttoolvec
        else:
            starttoolVec3 = None
        if goaltoolvec is not None:
            goaltoolVec3 = goaltoolvec
        else:
            goaltoolVec3 = None

        self.start_homomat = start_homomat
        self.goal_homomat = goal_homomat

        self.choice = choice
        startchoice = choice[:8]
        goalchoice = choice[8:]

        print("startgraspgid is None, all grasps are candidates")
        self._addend(start_homomat,
                     cond=startchoice,
                     worldframevec3=starttoolVec3,
                     tempobstacle=starttmpobstacle)
        print("goalgraspgid is None, all grasps are candidates")
        self._addend(goal_homomat,
                     cond=goalchoice,
                     worldframevec3=goaltoolVec3,
                     tempobstacle=goaltmpobstacle)
        # if self.debug:
        #     base.run()

        # add start to goal direct edges rgt-rgt
        for startnodeid in self.start_rgt_node_ids:
            for goalnodeid in self.goal_rgt_node_ids:
                # startnodeggid = start node global grip id
                startnodeggid = self.regg.nodes[startnodeid]['identity']
                goalnodeggid = self.regg.nodes[goalnodeid]['identity']
                print(startnodeggid, goalnodeggid)
                if startnodeggid == goalnodeggid:
                    self.regg.add_edge(startnodeid, goalnodeid, weight=1, edgetype='startgoalrgttransfer')

        # add start to goal direct edges lft-lft
        for startnodeid in self.start_lft_node_ids:
            for goalnodeid in self.goal_lft_node_ids:
                # startnodeggid = start node global grip id
                startnodeggid = self.regg.nodes[startnodeid]['identity']
                goalnodeggid = self.regg.nodes[goalnodeid]['identity']
                if startnodeggid == goalnodeggid:
                    self.regg.add_edge(startnodeid, goalnodeid, weight=1, edgetype='startgoallfttransfer')

    def _addend(self, objposmat, cond="startrgt", worldframevec3=None, tempobstacle=[]):
        """
        add a start or a goal for the regg, using different hand

        :param objposmat:
        :param cond: the specification of the rotmat4: "startrgt", "startlft", "goalrgt", "goallft"
        :param ctvec, ctangle: the conditions of filtering, the candidate hand z must have a smaller angle with vec
        :param toolvec: the direction to move the tool in the last step, it is described in the local coordinate system of the object
        :return:

        author: weiwei
        date: 20180925
        """

        if worldframevec3 is None:
            world_approach = np.array([0, 0, 1])
        else:
            world_approach = worldframevec3

        feasiblegrasps = 0
        ikfailedgrasps = 0
        ikapproachfailedgrasps = 0
        robotcollodedgrasps = 0
        handcollidedgrasps = 0
        # get the hand name
        if "rgt" in cond:
            hndname = "rgt_hnd"
            grasps = self.handover_data.grasp_list_rgt
        else:
            hndname = "lft_hnd"
            grasps = self.handover_data.grasp_list_lft

        print("Adding start or goal to the graph...")
        # the nodeids is also for quick access
        if cond == "startrgt":
            self.start_rgt_node_ids = []
            nodeids = self.start_rgt_node_ids
        elif cond == "startlft":
            self.start_lft_node_ids = []
            nodeids = self.start_lft_node_ids
        elif cond == "goalrgt":
            self.goal_rgt_node_ids = []
            nodeids = self.goal_rgt_node_ids
        elif cond == "goallft":
            self.goal_lft_node_ids = []
            nodeids = self.goal_lft_node_ids
        else:
            raise Exception("Wrong conditions!")
        # the node id of a globalgripid
        nodeidofglobalid = {}
        for graspid, graspinfo in enumerate(grasps):
            grasp_jawwidth = graspinfo[0]
            grasp_homomat = rm.homomat_from_posrot(graspinfo[1],
                                                   graspinfo[2])  # the mat of the grasp
            ttgs_homomat = np.dot(objposmat, grasp_homomat)  # the grasp at the obj posision
            # filtering
            approach_dir = -ttgs_homomat[:3, 2]
            # vector between object and robot ee
            # check if the hand collide with obstacles
            # set jawwidth to 50 to avoid collision with surrounding obstacles
            # set to gripping with is unnecessary
            is_hnd_collided = self.rbt.is_hnd_collided(hnd_name=hndname,
                                                       hnd_homomat=ttgs_homomat,
                                                       hnd_jawwidth=grasp_jawwidth,
                                                       obstacle_list=tempobstacle, )
            if not is_hnd_collided:
                ttgsfgrcenternp = ttgs_homomat[:3, 3]
                ttgsfgrcenternp_handa = ttgsfgrcenternp + approach_dir * self.ret_dis
                ttgsfgrcenternp_worlda = ttgsfgrcenternp + world_approach * self.ret_dis
                ttgsjawwidth = graspinfo[0]
                ttgsrotmat3 = ttgs_homomat[:3, :3]
                ikr = self.rbt.ik(component_name=hndname,
                                  tgt_pos=ttgs_homomat[:3, 3],
                                  tgt_rotmat=ttgs_homomat[:3, :3],
                                  all_sol=True, )
                ikr_ = []
                ikr_hnd_approach = []
                ikr_world_approach = []
                if ikr is not None and len(ikr) > 0:
                    for i in range(len(ikr)):
                        # collision detection:
                        self.rbt.fk(component_name=hndname, jnt_values=ikr[i])
                        if self.rbt.is_collided(obstacle_list=tempobstacle):
                            robotcollodedgrasps += 1
                            continue
                        ikr_hnd_approach_tmp = self.rbt.ik(component_name=hndname,
                                                           tgt_pos=ttgsfgrcenternp_handa,
                                                           tgt_rotmat=ttgs_homomat[:3, :3],
                                                           seed_jnt_values=ikr[i], )
                        if ikr_hnd_approach_tmp is not None:
                            # collision detection:
                            self.rbt.fk(component_name=hndname, jnt_values=ikr_hnd_approach_tmp)
                            if self.rbt.is_collided(obstacle_list=tempobstacle):
                                handcollidedgrasps += 1
                                continue
                            ikr_world_approach_tmp = self.rbt.ik(component_name=hndname,
                                                                 tgt_pos=ttgsfgrcenternp_worlda,
                                                                 tgt_rotmat=ttgsrotmat3,
                                                                 seed_jnt_values=ikr[i], )
                            if ikr_world_approach_tmp is not None:
                                # feasible grasp
                                self.rbt.fk(component_name=hndname, jnt_values=ikr_world_approach_tmp)
                                if self.rbt.is_collided(obstacle_list=tempobstacle):
                                    handcollidedgrasps += 1
                                    continue
                                ikr_.append(ikr[i])
                                ikr_hnd_approach.append(ikr_hnd_approach_tmp)
                                ikr_world_approach.append(ikr_world_approach_tmp)
                    if len(ikr_) > 0 and len(ikr_hnd_approach) > 0 and len(ikr_world_approach) > 0:
                        feasiblegrasps += 1
                        # note the tabletopposition here is not the contact for the intermediate states
                        # it is the zero pos
                        objposmat_copy = copy.deepcopy(objposmat)
                        tabletopposition = objposmat_copy[:3, :3]
                        startrotmat4worlda = copy.deepcopy(objposmat_copy)
                        startrotmat4worlda[:3, 3] = objposmat_copy[:3, 3] + world_approach * self.ret_dis
                        startrotmathanda = copy.deepcopy(objposmat_copy)
                        startrotmathanda[:3, 3] = objposmat_copy[:3, 3] + approach_dir * self.ret_dis
                        # manipulability
                        manipulability_world_approach = []
                        for _ik in ikr_world_approach:
                            self.rbt.fk(component_name=hndname,
                                        jnt_values=_ik, )
                            manipulability_world_approach.append(self.rbt.manipulability(component_name=hndname, ))
                        node_name = cond + str(graspid)
                        self.regg.add_node(node_name, fgrcenter=ttgsfgrcenternp,
                                           fgrcenterhanda=ttgsfgrcenternp_handa,
                                           fgrcenterworlda=ttgsfgrcenternp_worlda,
                                           jawwidth=ttgsjawwidth, hndrotmat3np=ttgsrotmat3,
                                           armjnts=ikr,
                                           armjntshanda=ikr_hnd_approach,
                                           armjntsworlda=ikr_world_approach,
                                           tabletopplacementrotmat=objposmat_copy,
                                           tabletopposition=tabletopposition,
                                           tabletopplacementrotmathanda=startrotmathanda,
                                           tabletopplacementrotmatworlda=startrotmat4worlda,
                                           identity=cond[-3:] + str(graspid),
                                           manipulability_approach=manipulability_world_approach,
                                           node_type='start' if 'start' in cond else 'goal',
                                           )
                        nodeidofglobalid[cond[-3:] + str(graspid)] = node_name
                        nodeids.append(cond + str(graspid))
                    else:
                        ikapproachfailedgrasps += 1
                else:
                    ikfailedgrasps += 1
                # ikcaz = self.robot.numikr(ttgsfgrcenternp_worldaworldz, ttgsrotmat3np, armname = 'rgt')
                # if (ikr is not None) and (ikr_hnd_approach is not None) and (ikr_world_approach is not None):
                #     feasiblegrasps += 1
                #     # note the tabletopposition here is not the contact for the intermediate states
                #     # it is the zero pos
                #     objposmat_copy = copy.deepcopy(objposmat)
                #     tabletopposition = objposmat_copy[:3, :3]
                #     startrotmat4worlda = copy.deepcopy(objposmat_copy)
                #     startrotmat4worlda[:3, 3] = objposmat_copy[:3, 3] + world_approach * self.retworlda
                #     # manipulability
                #     self.rbt.movearmfk(ikr_world_approach, armname=cond[-3:])
                #     manipulability = self.rbt.manipulability(armname=cond[-3:])
                #     self.regg.add_node(cond + str(graspid), fgrcenter=ttgsfgrcenternp,
                #                        fgrcenterhanda=ttgsfgrcenternp_handa,
                #                        fgrcenterworlda=ttgsfgrcenternp_worlda,
                #                        jawwidth=ttgsjawwidth, hndrotmat3np=ttgsrotmat3,
                #                        armjnts=ikr,
                #                        armjntshanda=ikr_hnd_approach,
                #                        armjntsworlda=ikr_world_approach,
                #                        tabletopplacementrotmat=objposmat_copy,
                #                        tabletopposition=tabletopposition,
                #                        tabletopplacementrotmathanda=objposmat_copy,
                #                        tabletopplacementrotmatworlda=startrotmat4worlda,
                #                        identity=cond[-3:] + str(graspid),
                #                        manipulability=manipulability
                #                        )
                #     nodeidofglobalid[cond[-3:] + str(graspid)] = cond + str(graspid)
                #     nodeids.append(cond + str(graspid))
                #     # tmprtq85.reparentTo(base.render)
                # else:
                #     if ikr is None:
                #         ikfailedgrasps += 1
                #     else:
                #         ikapproachfailedgrasps += 1
            else:
                handcollidedgrasps += 1

        # base.run()
        print("IK failed grasps:", ikfailedgrasps)
        print("IK handa failed grasps", ikapproachfailedgrasps)
        print("Hand collided grasps:", handcollidedgrasps)
        print("Robot collided grasps", robotcollodedgrasps)
        print("feasible grasps:", feasiblegrasps)

        if len(nodeids) == 0:
            print("No available " + cond[:-3] + " grip for " + cond[-3:] + " hand!")

        # connect nodes at the start or goal
        for edge in list(itertools.combinations(nodeids, 2)):
            self.regg.add_edge(*edge, weight=1, edgetype=cond + 'transit')

        # add transfer edge
        for reggnode, reggnodedata in self.regg.nodes(data=True):
            if reggnode.startswith(cond[-3:]) or reggnode.startswith('ho' + cond[-3:]):
                globalgripid = reggnodedata['identity']
                if globalgripid in nodeidofglobalid.keys():
                    nodeid = nodeidofglobalid[globalgripid]
                    self.regg.add_edge(nodeid, reggnode, weight=1, edgetype=cond + 'transfer')

    def _extract_valid_path(self, shortest_paths: iter) -> list:
        direct_paths = []
        for path in shortest_paths:
            start_idx, goal_idx = -1, -1
            # Find the last 'start' node before any other intermediate nodes
            for i, node in enumerate(path):
                if node.startswith('start'):
                    start_idx = i
                elif start_idx != -1:  # Stop at the first non-'start' node after 'start'
                    break
            # Find the first 'goal' node after the last 'start' node
            for i, node in enumerate(path[start_idx:], start=start_idx):
                if node.startswith('goal'):
                    goal_idx = i + 1  # Include the goal node in the slice
                    break
            # Append the valid path slice if both start and goal are found
            if start_idx != -1 and goal_idx != -1:
                direct_paths.append(path[start_idx:goal_idx])
                # return path[start_idx:goal_idx]
            else:
                continue
                # raise Exception('No valid start-goal path found')
        return direct_paths

    def update_shortest_path(self):
        """
        this function is assumed to be called after start and goal are set

        :return:
        """
        if len(self.start_lft_node_ids) > 0 and len(self.goal_lft_node_ids) > 0:
            startgrip = self.start_lft_node_ids[0]
            goalgrip = self.goal_lft_node_ids[0]
            start_hand = 'lft'
            goal_hand = 'lft'
        elif len(self.start_rgt_node_ids) > 0 and len(self.goal_rgt_node_ids) > 0:
            startgrip = self.start_rgt_node_ids[0]
            goalgrip = self.goal_rgt_node_ids[0]
            start_hand = 'rgt'
            goal_hand = 'rgt'
        elif len(self.start_lft_node_ids) > 0 and len(self.goal_rgt_node_ids) > 0:
            startgrip = self.start_lft_node_ids[0]
            goalgrip = self.goal_rgt_node_ids[0]
            start_hand = 'lft'
            goal_hand = 'rgt'
        elif len(self.start_rgt_node_ids) > 0 and len(self.goal_lft_node_ids) > 0:
            startgrip = self.start_rgt_node_ids[0]
            goalgrip = self.goal_lft_node_ids[0]
            start_hand = 'rgt'
            goal_hand = 'lft'
        else:
            raise Exception("No start or goal is set!")
        self.shortest_paths = list(nx.all_shortest_paths(self.regg, source=startgrip, target=goalgrip))
        print(self.shortest_paths)
        self.direct_shortest_paths = self._extract_valid_path(self.shortest_paths)
        if len(self.direct_shortest_paths) < 1:
            raise Exception(f"No start{start_hand} to goal{goal_hand}!")

    # startlft goalrgt
    # if len(self.start_lft_node_ids) > 0 and len(self.goal_rgt_node_ids) > 0:
    #     print("Number of start grasps: ", len(self.start_lft_node_ids), "; Number of goal grasps: ",
    #           len(self.goal_rgt_node_ids))
    #     startgrip = self.start_lft_node_ids[0]
    #     goalgrip = self.goal_rgt_node_ids[0]
    #     self.shortest_paths = nx.all_shortest_paths(self.regg, source=startgrip, target=goalgrip)
    #     self.directshortestpaths_startlftgoalrgt = []
    #     # first n obj in self.shortest_paths
    #     maxiter = 20
    #     counter = 0
    #     tmpshortestpaths = []
    #     for node in self.shortest_paths:
    #         tmpshortestpaths.append(node)
    #         counter += 1
    #         if counter >= maxiter:
    #             break
    #     self.shortest_paths = tmpshortestpaths
    #     self.shortest_paths.sort(
    #         key=lambda element: sum([self.regg.nodes[node]['manipulability'] for node in element]), reverse=True)
    #     try:
    #         for path in self.shortest_paths:
    #             for i, pathnode in enumerate(path):
    #                 if pathnode.startswith('start') and i < len(path) - 1:
    #                     continue
    #                 else:
    #                     self.directshortestpaths_startlftgoalrgt.append(path[i - 1:])
    #                     break
    #             for i, pathnode in enumerate(self.directshortestpaths_startlftgoalrgt[-1]):
    #                 if i > 0 and pathnode.startswith('goal'):
    #                     self.directshortestpaths_startlftgoalrgt[-1] = self.directshortestpaths_startlftgoalrgt[-1][
    #                                                                    :i + 1]
    #                     break
    #     except:
    #         raise Exception('No startlftgoalrgt')

    def get_motion_sequence(self, id, type="OO", previous_state: JJOState = None):
        """
            generate motion sequence using the shortest path
            right arm
            this function is for simple pick and place with regrasp

            # 20190319 comment by weiwei
            five letters are attached to nids,
            they are "x", "w", "o", "c", "i"
            where "x" indicates handa,
            "w" indicates worlda,
            "o" and "c" are at grasping psoe, they indicate the open and close states of a hand
            "i" indicates initial pose
            these letter will be use to determine planning methods in the planner.py file
            e.g. an o->c motion will be simple finger motion, no rrt planners will be called
            a x->w will be planning with hold, x->c will be interplation, x->i will be planning without hold, etc.
            see the planning.py file for details

            #20190319 comment by weiwei
            OO means start from a hand open pose and stop at a hand open pose
            OC means start from a hand open pose and stop at a hand close pose
            CC means start from a hand close pose and stop at a hand close pose
            To generate multiple motion sequences, OC->CC->CC->...->CC->CO is the preferred type order choice


            :param: regrip an object of the regriptppfp.RegripTppfp class
            :param id: which path to plot
            :param choice: startrgtgoalrgt/startrgtgoallft/startlftgoalrgt/startlftgoallft
            :param type: one of "OO', "OC", "CC", "CO"
            :param previous_state: set it to [] if the motion is not a continuing one, or else, set it to [lastobjmat4, lastikr, lastjawwidth]

            :return: [[waist, lftbody, rgtbody],...]

            author: weiwei
            date: 20170302
            """

        if (self.choice not in ["startrgtgoalrgt", "startrgtgoallft", "startlftgoalrgt", "startlftgoallft"]):
            raise Exception("The choice parameter of get_motion_sequence must be " +
                            "one of startrgtgoalrt, startrgtgoalft, startlftgoalrgt, startlftgoallft! " +
                            f"Right now it is {self.choice}.")

        assert type in ["OO", "OC", "CC",
                        "CO"], f"The choice parameter of type must be  one of OO, OC, CC, CO! Right now it is {self.choice}."

        direct_shortest_paths = self.direct_shortest_paths
        handover_pos = self.handover_data.floating_pose_list
        if len(direct_shortest_paths) == 0:
            raise Exception("No path Found")

        traj_data = RbtTrajData()

        path_nid_list = direct_shortest_paths[id]
        # initialize the first state
        if previous_state:
            traj_data.add_state(state=previous_state)
        else:
            self.rbt.init_conf()
            traj_data.add_state(get_jjostate_rbt_objhomomat(self.rbt, self.start_homomat))
        extendedpathnidlist = ['begin']
        print(path_nid_list)
        # generate the motion sequence
        for i in range(len(path_nid_list) - 1):
            if i == 0 and len(path_nid_list) == 2:
                # two node path
                # they must be both rgt or both lft
                # they cannot be handover
                ## starting node
                nid = path_nid_list[i]
                if 'lft' in nid:
                    component_name = "lft_arm"
                elif 'rgt' in nid:
                    component_name = "rgt_arm"
                else:
                    raise Exception("Unknown component")
                jnt_val = self.regg.nodes[nid]['armjnts']
                jnt_val_hand_approach = self.regg.nodes[nid]['armjntshanda']
                jnt_val_world_approach = self.regg.nodes[nid]['armjntsworlda']
                obj_homomat = self.regg.nodes[nid]['tabletopplacementrotmat']
                obj_homomat_handa = self.regg.nodes[nid]['tabletopplacementrotmathanda']
                obj_homomat_worlda = self.regg.nodes[nid]['tabletopplacementrotmatworlda']
                jawwidth = self.regg.nodes[nid]['jawwidth']
                # choice
                if ((type == "OC") or (type == "OO")):
                    state_approach = traj_data.copy_last_state()
                    state_grasp_o = traj_data.copy_last_state()
                    jawwidth_open = self.rbt.hnd_dict[component_name].jaw_range[1]
                    state_approach.set_state(component_name=component_name,
                                             jnt_val=jnt_val_hand_approach,
                                             jawwidth=jawwidth_open,
                                             obj_homomat=obj_homomat)
                    state_grasp_o.set_state(component_name=component_name,
                                            jnt_val=jnt_val,
                                            jawwidth=jawwidth_open,
                                            obj_homomat=obj_homomat)
                    extendedpathnidlist.append(nid + "x")
                    extendedpathnidlist.append(nid + "o")
                    traj_data.add_state(state=state_approach)
                    traj_data.add_state(state=state_grasp_o)
                state_grasp_c = traj_data.copy_last_state()
                state_detach = traj_data.copy_last_state()
                state_grasp_c.set_state(component_name=component_name,
                                        jnt_val=jnt_val,
                                        jawwidth=jawwidth,
                                        obj_homomat=obj_homomat)
                state_detach.set_state(component_name=component_name,
                                       jnt_val=jnt_val_world_approach,
                                       jawwidth=jawwidth,
                                       obj_homomat=obj_homomat_worlda)
                extendedpathnidlist.append(nid + "c")
                extendedpathnidlist.append(nid + "w")
                traj_data.add_state(state=state_grasp_c)
                traj_data.add_state(state=state_detach)
                ## goal node
                nid = path_nid_list[i + 1]
                jnt_val = self.regg.nodes[nid]['armjnts']
                jnt_val_hand_approach = self.regg.nodes[nid]['armjntshanda']
                jnt_val_world_approach = self.regg.nodes[nid]['armjntsworlda']
                obj_homomat = self.regg.nodes[nid]['tabletopplacementrotmat']
                obj_homomat_handa = self.regg.nodes[nid]['tabletopplacementrotmathanda']
                obj_homomat_worlda = self.regg.nodes[nid]['tabletopplacementrotmatworlda']
                jawwidth = self.regg.nodes[nid]['jawwidth']
                # initialize
                state_approach = traj_data.copy_last_state()
                state_grasp_c = traj_data.copy_last_state()
                state_approach.set_state(component_name=component_name,
                                         jnt_val=jnt_val_hand_approach,
                                         jawwidth=jawwidth,
                                         obj_homomat=obj_homomat_handa)
                state_grasp_c.set_state(component_name=component_name,
                                        jnt_val=jnt_val,
                                        jawwidth=jawwidth,
                                        obj_homomat=obj_homomat)
                extendedpathnidlist.append(nid + "w")
                extendedpathnidlist.append(nid + "c")
                traj_data.add_state(state=state_approach)
                traj_data.add_state(state=state_grasp_c)
                # choice
                if ((type == "CO") or (type == "OO")):
                    jawwidth_open = self.rbt.hnd_dict[component_name].jaw_range[1]
                    state_grasp_o = traj_data.copy_last_state()
                    state_detach = traj_data.copy_last_state()
                    state_grasp_o.set_state(component_name=component_name,
                                            jnt_val=jnt_val,
                                            jawwidth=jawwidth_open,
                                            obj_homomat=obj_homomat)
                    state_detach.set_state(component_name=component_name,
                                           jnt_val=jnt_val_world_approach,
                                           jawwidth=jawwidth_open,
                                           obj_homomat=obj_homomat)
                    extendedpathnidlist.append(nid + "o")
                    extendedpathnidlist.append(nid + "x")
                    traj_data.add_state(state=state_grasp_o)
                    traj_data.add_state(state=state_detach)
            elif i == 0:
                # not two nodepath, starting node, transfer
                ## starting node
                nid = path_nid_list[i]
                if 'lft' in nid:
                    component_name = "lft_arm"
                elif 'rgt' in nid:
                    component_name = "rgt_arm"
                else:
                    raise Exception("Unknown component")
                jnt_val = self.regg.nodes[nid]['armjnts']
                jnt_val_hand_approach = self.regg.nodes[nid]['armjntshanda']
                jnt_val_world_approach = self.regg.nodes[nid]['armjntsworlda']
                obj_homomat = self.regg.nodes[nid]['tabletopplacementrotmat']
                obj_homomat_handa = self.regg.nodes[nid]['tabletopplacementrotmathanda']
                obj_homomat_worlda = self.regg.nodes[nid]['tabletopplacementrotmatworlda']
                jawwidth = self.regg.nodes[nid]['jawwidth']
                # choice
                if nid.startswith('start'):
                    if ((type == "OC") or (type == "OO")):
                        state_approach = traj_data.copy_last_state()
                        state_grasp_o = traj_data.copy_last_state()
                        jawwidth_open = self.rbt.hnd_dict[component_name].jaw_range[1]
                        state_approach.set_state(component_name=component_name,
                                                 jnt_val=jnt_val_hand_approach,
                                                 jawwidth=jawwidth_open,
                                                 obj_homomat=obj_homomat)
                        state_grasp_o.set_state(component_name=component_name,
                                                jnt_val=jnt_val,
                                                jawwidth=jawwidth_open,
                                                obj_homomat=obj_homomat)
                        extendedpathnidlist.append(nid + "x")
                        extendedpathnidlist.append(nid + "o")
                        traj_data.add_state(state=state_approach)
                        traj_data.add_state(state=state_grasp_o)
                    state_grasp_c = traj_data.copy_last_state()
                    state_detach = traj_data.copy_last_state()
                    state_grasp_c.set_state(component_name=component_name,
                                            jnt_val=jnt_val,
                                            jawwidth=jawwidth,
                                            obj_homomat=obj_homomat)
                    state_detach.set_state(component_name=component_name,
                                           jnt_val=jnt_val_world_approach,
                                           jawwidth=jawwidth,
                                           obj_homomat=obj_homomat_worlda)
                    extendedpathnidlist.append(nid + "c")
                    extendedpathnidlist.append(nid + "w")
                    traj_data.add_state(state=state_grasp_c)
                    traj_data.add_state(state=state_detach)

            elif i + 1 != len(path_nid_list) - 1:
                # if handovertransit
                if self.regg.edges[path_nid_list[i], path_nid_list[i + 1]]['edgetype'] == "handovertransit":
                    nid0 = path_nid_list[i]
                    nid1 = path_nid_list[i + 1]
                    nid0_component_name = "lft_arm" if 'lft' in nid0 else "rgt_arm"
                    nid1_component_name = "lft_arm" if 'lft' in nid1 else "rgt_arm"
                    #### nid0 move to handover
                    nid0_state_hndovr_c = traj_data.copy_last_state()
                    nid0_state_hndovr_c.set_state(component_name=nid0_component_name,
                                                  jnt_val=self.regg.nodes[nid0]['armjnts'],
                                                  jawwidth=self.regg.nodes[nid0]['jawwidth'],
                                                  obj_homomat=handover_pos[self.regg.nodes[nid0]['handoverposid']])
                    extendedpathnidlist.append(nid0 + "c")
                    traj_data.add_state(state=nid0_state_hndovr_c)
                    #### nid1 move to handover
                    jawwidth_open = self.rbt.hnd_dict[nid1_component_name].jaw_range[1]
                    nid1_state_hndovr_x = traj_data.copy_last_state()
                    nid1_state_hndovr_x.set_state(component_name=nid1_component_name,
                                                  jnt_val=self.regg.nodes[nid1]['armjntshanda'],
                                                  jawwidth=jawwidth_open,
                                                  obj_homomat=handover_pos[self.regg.nodes[nid1]['handoverposid']])
                    nid1_state_hndovr_o = traj_data.copy_last_state()
                    nid1_state_hndovr_o.set_state(component_name=nid1_component_name,
                                                  jnt_val=self.regg.nodes[nid1]['armjnts'],
                                                  jawwidth=jawwidth_open,
                                                  obj_homomat=handover_pos[self.regg.nodes[nid1]['handoverposid']])
                    nid1_state_hndovr_c = traj_data.copy_last_state()
                    nid1_state_hndovr_c.set_state(component_name=nid1_component_name,
                                                  jnt_val=self.regg.nodes[nid1]['armjnts'],
                                                  jawwidth=self.regg.nodes[nid1]['jawwidth'],
                                                  obj_homomat=handover_pos[self.regg.nodes[nid1]['handoverposid']])
                    extendedpathnidlist.append(nid1 + "x")
                    extendedpathnidlist.append(nid1 + "o")
                    extendedpathnidlist.append(nid1 + "c")
                    traj_data.add_state(state=nid1_state_hndovr_x)
                    traj_data.add_state(state=nid1_state_hndovr_o)
                    traj_data.add_state(state=nid1_state_hndovr_c)
                    #### nid0 move back
                    jawwidth_open = self.rbt.hnd_dict[nid0_component_name].jaw_range[1]
                    nid0_state_hndovr_o = traj_data.copy_last_state()
                    nid0_state_hndovr_o.set_state(component_name=nid0_component_name,
                                                  jnt_val=self.regg.nodes[nid0]['armjnts'],
                                                  jawwidth=jawwidth_open,
                                                  obj_homomat=handover_pos[self.regg.nodes[nid0]['handoverposid']])
                    nid0_state_hndovr_x = traj_data.copy_last_state()
                    nid0_state_hndovr_x.set_state(component_name=nid0_component_name,
                                                  jnt_val=self.regg.nodes[nid0]['armjntshanda'],
                                                  jawwidth=jawwidth_open,
                                                  obj_homomat=handover_pos[self.regg.nodes[nid0]['handoverposid']])
                    extendedpathnidlist.append(nid0 + "o")
                    extendedpathnidlist.append(nid0 + "x")
                    traj_data.add_state(state=nid0_state_hndovr_o)
                    traj_data.add_state(state=nid0_state_hndovr_x)
                else:
                    print("--" * 20)
                    print("UNKNOWN EDGE TYPE")
                    print("Edge type is: ", self.regg.edges[path_nid_list[i], path_nid_list[i + 1]]['edgetype'])
                    print("--" * 20)
                    # not two node path, middle nodes, if transfer
                    ## middle first
                    nid = path_nid_list[i]
                    if nid.startswith('ho'):
                        pass
                    ## middle second
                    nid = path_nid_list[i + 1]
                    # could be ho
                    if nid.startswith('ho'):
                        pass
            else:
                ## last node
                nid = path_nid_list[i + 1]
                if 'lft' in nid:
                    component_name = "lft_arm"
                elif 'rgt' in nid:
                    component_name = "rgt_arm"
                else:
                    raise Exception("Unknown component")
                jnt_val = self.regg.nodes[nid]['armjnts']
                jnt_val_hand_approach = self.regg.nodes[nid]['armjntshanda']
                jnt_val_world_approach = self.regg.nodes[nid]['armjntsworlda']
                obj_homomat = self.regg.nodes[nid]['tabletopplacementrotmat']
                obj_homomat_handa = self.regg.nodes[nid]['tabletopplacementrotmathanda']
                obj_homomat_worlda = self.regg.nodes[nid]['tabletopplacementrotmatworlda']
                jawwidth = self.regg.nodes[nid]['jawwidth']
                # choice
                if nid.startswith('goal'):
                    state_approach = traj_data.copy_last_state()
                    state_grasp_c = traj_data.copy_last_state()
                    state_approach.set_state(component_name=component_name,
                                             jnt_val=jnt_val_hand_approach,
                                             jawwidth=jawwidth,
                                             obj_homomat=obj_homomat_handa)
                    state_grasp_c.set_state(component_name=component_name,
                                            jnt_val=jnt_val,
                                            jawwidth=jawwidth,
                                            obj_homomat=obj_homomat)
                    extendedpathnidlist.append(nid + "w")
                    extendedpathnidlist.append(nid + "c")
                    traj_data.add_state(state=state_approach)
                    traj_data.add_state(state=state_grasp_c)
                    # choice
                    if ((type == "CO") or (type == "OO")):
                        jawwidth_open = self.rbt.hnd_dict[component_name].jaw_range[1]
                        state_grasp_o = traj_data.copy_last_state()
                        state_detach = traj_data.copy_last_state()
                        state_grasp_o.set_state(component_name=component_name,
                                                jnt_val=jnt_val,
                                                jawwidth=jawwidth_open,
                                                obj_homomat=obj_homomat)
                        state_detach.set_state(component_name=component_name,
                                               jnt_val=jnt_val_world_approach,
                                               jawwidth=jawwidth_open,
                                               obj_homomat=obj_homomat)
                        extendedpathnidlist.append(nid + "o")
                        extendedpathnidlist.append(nid + "x")
                        traj_data.add_state(state=state_grasp_o)
                        traj_data.add_state(state=state_detach)
            if self.gobacktoinitafterplanning == True:
                """回到initialization的位置"""
                # # pre-place
                nid = pathnidlist[i + 1]
                # initilize
                grpjawwidth1 = self.regg.nodes[nid]['jawwidth']
                armjntsgrp1 = self.regg.nodes[nid]['armjnts']
                objmat4b = self.hndover_pos[self.regg.nodes[nid0]['handoverposid']]
                # move back to init pose
                if nid.startswith('goalrgt'):
                    numikrlist.append([self.rbt.initlftjntsr[0], armjntsgrp1, self.rbt.initlftjntsr[1:]])
                    jawwidth.append([grpjawwidth1, self.rbt.lfthnd.jawwidthopen])
                elif nid.startswith('goallft'):
                    numikrlist.append([self.rbt.initrgtjntsr[0], self.rbt.initrgtjntsr[1:], armjntsgrp1])
                    jawwidth.append([self.rbt.rgthnd.jawwidthopen, grpjawwidth1])
                objmat4list.append(objmat4b)
                fnid = 'goallft' + nid[nid.index('t') + 1:] if nid.startswith('goalrgt') else 'goalrgt' + nid[
                                                                                                          nid.index(
                                                                                                              't') + 1:]
                extendedpathnidlist.append(fnid + "i")
        extendedpathnidlist.append('end')

        return traj_data, extendedpathnidlist, path_nid_list

    def removeBadNodes(self, nodelist):
        """
        remove the invalidated nodes to prepare for a new plan

        :param nodelist: a list of invalidated nodes
        :return:

        author: weiwei
        date: 20170920
        """

        print("Removing nodes ", nodelist)
        self.regg.remove_nodes_from(nodelist)
        for node in nodelist:
            if node.startswith('startrgt'):
                try:
                    self.start_rgt_node_ids.remove(node)
                except KeyError:
                    pass
            if node.startswith('startlft'):
                try:
                    self.start_lft_node_ids.remove(node)
                except KeyError:
                    pass
            if node.startswith('goalrgt'):
                try:
                    self.goal_rgt_node_ids.remove(node)
                except KeyError:
                    pass
            if node.startswith('goallft'):
                try:
                    self.goal_lft_node_ids.remove(node)
                except KeyError:
                    pass

    def removeBadEdge(self, node0, node1):
        """
        remove an invalidated edge to prepare for a new plan

        :param node0, node1 two ends of an edge
        :return:

        author: weiwei
        date: 20190423
        """
        if node0 == node1:
            return
        print("Removing edge ", node0, node1)
        self.regg.remove_edge(node0, node1)

    def planRegrasp(self, objcm, obstaclecmlist=None, id=0, switch="OC", previous=[], end=False,
                    togglemp=True):
        """
        plan the regrasp sequences

        :param objpath:
        :param robot:
        :param hand:
        :param dbase:
        :param obstaclecmlist:
        :param id = 0
        :param switch in "OC" open-close "CC" close-close
        :param previous: set it to [] if the motion is not a continuing one, or else, set it to [lastikr, lastjawwidth,lastobjmat4]
        :param end: set it to True if it is the last one
        :param togglemp denotes whether the motion between the keyposes are planned or not, True by default
        :return:

        author: weiwei
        date: 20180924
        """

        robot = self.rbt
        cdchecker = self.cdchecker
        if obstaclecmlist == None:
            obstaclecmlist = self.obstacle_list
        else:
            obstaclecmlist = obstaclecmlist

        while True:
            print("new search")
            self.update_shortest_path()
            print("I Get Stuck updateshrtestpath")
            [objms, numikrms, jawwidth, pathnidlist, originalpathnidlist] = \
                self.get_motion_sequence(id=id, type=switch, previous=previous)
            print("I Get Stuck get_motion_sequence")
            if objms == None:
                return [None, None, None, None]
            bcdfree = True
            for i in range(len(numikrms)):
                rgtarmjnts = numikrms[i][1].tolist()
                lftarmjnts = numikrms[i][2].tolist()
                robot.movealljnts([numikrms[i][0], 0, 0] + rgtarmjnts + lftarmjnts)
                # skip the exact handover pose and only detect the cd between armhnd and body
                if pathnidlist[i].startswith('ho') and pathnidlist[i + 1].startswith('ho'):
                    abcd = cdchecker.isCollidedHO(robot, obstaclecmlist)
                    if abcd:
                        if self.inspector is not None:
                            # inspector:
                            self.inspector.add_error("handover collision",
                                                     Error_info(name=f"{pathnidlist[i]} -- {pathnidlist[i + 1]}",
                                                                objmat=objms[i], lftarmjnts=lftarmjnts,
                                                                rgtarmjnts=rgtarmjnts, lftjawwidth=jawwidth[i][0],
                                                                rgtjawwidth=jawwidth[i][1]), )
                        self.removeBadNodes([pathnidlist[i][:-1]])
                        print("Abcd collided at ho pose")
                        bcdfree = False
                        break
                else:
                    # NOTE: we ignore both arms here for conciseness
                    # This might be a potential bug
                    if cdchecker.isCollided(robot, obstaclecmlist, holdarmname="all"):
                        # inspector:
                        if self.inspector is not None:
                            self.inspector.add_error("non-ho pose collision",
                                                     Error_info(name=pathnidlist[i],
                                                                objmat=objms[i], lftarmjnts=lftarmjnts,
                                                                rgtarmjnts=rgtarmjnts, lftjawwidth=jawwidth[i][0],
                                                                rgtjawwidth=jawwidth[i][1]), )
                        self.removeBadNodes([pathnidlist[i][:-1]])
                        print("Robot collided at non-ho pose")
                        bcdfree = False
                        break
            robot.goinitpose()
            if bcdfree:
                objmsmp = []
                numikrmsmp = []
                jawwidthmp = []
                print(pathnidlist)
                if not togglemp:
                    for i, numikrm in enumerate(numikrms):
                        if i > 0:
                            startid = pathnidlist[i - 1]
                            endid = pathnidlist[i]
                            if (not end) and (endid == 'end'):
                                continue
                            if (len(previous) > 0) and (startid == 'begin'):
                                continue
                            numikrmsmp.append([numikrms[i - 1], numikrms[i]])
                            objmsmp.append([objms[i - 1], objms[i]])
                            jawwidthmp.append([jawwidth[i - 1], jawwidth[i]])
                    return objmsmp, numikrmsmp, jawwidthmp, originalpathnidlist

                # INNERLOOP motion planning
                smoother = sm.Smoother()
                ctcallback = ctcb.CtCallback(robot, cdchecker)
                breakflag = False
                for i, numikrm in enumerate(numikrms):
                    if i > 0:
                        # determine which arm to plan
                        # assume right
                        # assume redundant planning
                        robot.goinitpose()
                        startid = pathnidlist[i - 1]
                        endid = pathnidlist[i]
                        objmat = objms[i - 1]
                        objrot = objmat[:3, :3]
                        objpos = objmat[:3, 3]
                        if (not end) and (endid == 'end'):
                            continue
                        if (len(previous) > 0) and (startid == 'begin'):
                            continue
                        if (startid[-1] == "o" and endid[-1] == "c") or (startid[-1] == "c" and endid[-1] == "o"):
                            # open and close gripper
                            print("O/C hands, simply include ", pathnidlist[i - 1], " and ", pathnidlist[i])
                            numikrmsmp.append([numikrms[i - 1], numikrms[i]])
                            objmsmp.append([objms[i - 1], objms[i]])
                            jawwidthmp.append([jawwidth[i - 1], jawwidth[i]])
                            continue
                        if (startid[:-1] == endid[:-1]):  # move to handover pose or linear interpolation
                            if (startid[-1] != "i") and (endid[-1] != "i"):
                                # linear interpolation
                                tempnumikrmsmp = []
                                tempjawwidthmp = []
                                tempobjmsmp = []
                                temparmname = "rgt"
                                startjntags = numikrms[i - 1][1].tolist()
                                goaljntags = numikrms[i][1].tolist()
                                if "lft" in startid:
                                    temparmname = "lft"
                                    startjntags = numikrms[i - 1][2].tolist()
                                    goaljntags = numikrms[i][2].tolist()
                                # TODO there is about 0.1 mm error in the final position
                                [interplatedjnts, interplatedobjposes] = \
                                    ctcallback.isLMAvailableJNTwithObj(startjntags, goaljntags,
                                                                       [objpos, objrot], armname=temparmname,
                                                                       type=startid[-1])
                                if len(interplatedjnts) == 0:
                                    print("Failed to interpolate motion primitive! restarting...")
                                    # always a single hand
                                    if self.inspector is not None:
                                        # inspector:
                                        self.inspector.add_error("interplation error",
                                                                 Error_info(name=f"{startid}-{endid}",
                                                                            objmat=None,
                                                                            lftarmjnts=startjntags if temparmname == "lft" else
                                                                            numikrms[i - 1][2],
                                                                            rgtarmjnts=startjntags if temparmname == "rgt" else
                                                                            numikrms[i - 1][1],
                                                                            lftjawwidth=jawwidth[i - 1][0],
                                                                            rgtjawwidth=jawwidth[i - 1][1]), )
                                        self.inspector.add_error("interplation error",
                                                                 Error_info(name=f"{startid}-{endid}",
                                                                            objmat=None,
                                                                            lftarmjnts=goaljntags if temparmname == "lft" else
                                                                            numikrms[i - 1][2],
                                                                            rgtarmjnts=goaljntags if temparmname == "rgt" else
                                                                            numikrms[i - 1][1],
                                                                            lftjawwidth=jawwidth[i - 1][0],
                                                                            rgtjawwidth=jawwidth[i - 1][1]), )

                                    self.removeBadNodes([pathnidlist[i - 1][:-1]])
                                    breakflag = True
                                    break
                                print("Motion primitives, interplate ", pathnidlist[i - 1], " and ", pathnidlist[i])
                                for eachitem in interplatedjnts:
                                    if temparmname == "rgt":
                                        tempnumikrmsmp.append(
                                            [numikrms[i - 1][0], np.array(eachitem), numikrms[i - 1][2]])
                                    else:
                                        tempnumikrmsmp.append(
                                            [numikrms[i - 1][0], numikrms[i - 1][1], np.array(eachitem)])
                                    tempjawwidthmp.append(jawwidth[i - 1])
                                for eachitem in interplatedobjposes:
                                    tempobjmsmp.append(rm.homobuild(eachitem[0], eachitem[1]))
                                numikrmsmp.append(tempnumikrmsmp)
                                jawwidthmp.append(tempjawwidthmp)
                                objmsmp.append(tempobjmsmp)
                                # update the keypose to avoid non-continuous linear motion: numikrms and objms
                                if temparmname == "rgt":
                                    numikrms[i][1] = tempnumikrmsmp[-1][1]
                                elif temparmname == "lft":
                                    numikrms[i][2] = tempnumikrmsmp[-1][2]
                                objms[i] = tempobjmsmp[-1]
                                continue
                        # init robot pose
                        rgtarmjnts = numikrms[i - 1][1].tolist()
                        lftarmjnts = numikrms[i - 1][2].tolist()
                        robot.movealljnts([numikrms[i - 1][0], 0, 0] + rgtarmjnts + lftarmjnts)
                        # assume rgt
                        armname = 'rgt'
                        start = numikrms[i - 1][1].tolist()
                        goal = numikrms[i][1].tolist()
                        startjawwidth = jawwidth[i - 1][0]
                        if "lft" in endid:
                            armname = 'lft'
                            start = numikrms[i - 1][2].tolist()
                            goal = numikrms[i][2].tolist()
                            startjawwidth = jawwidth[i - 1][1]
                        starttreesamplerate = 25
                        endtreesamplerate = 30
                        print(armname)
                        print(startjawwidth)
                        ctcallback.setarmname(armname)
                        planner = ddrrtc.DDRRTConnect(start=start, goal=goal, ctcallback=ctcallback,
                                                      starttreesamplerate=starttreesamplerate,
                                                      endtreesamplerate=endtreesamplerate, expanddis=20,
                                                      maxiter=200, maxtime=7.0)
                        tempnumikrmsmp = []
                        tempjawwidthmp = []
                        tempobjmsmp = []
                        if (endid[-1] == "c") or (endid[-1] == "w"):
                            print("Planning hold motion between ", pathnidlist[i - 1], " and ", pathnidlist[i])
                            relpos, relrot = robot.getinhandpose(objpos, objrot, armname)

                            path, sampledpoints = planner.planninghold([objcm], [[relpos, relrot]], obstaclecmlist)
                            if path is False:
                                print("Motion planning with hold failed! restarting...")

                                # TODO remove bad edge?
                                # regrip.removeBadNodes([pathnidlist[i-1][:-1]])
                                self.removeBadNodes([pathnidlist[i][:-1]])
                                if self.inspector is not None:
                                    # inspector
                                    self.inspector.add_error("rrt-hold error",
                                                             Error_info(name=f"{startid}-{endid}",
                                                                        objmat=None,
                                                                        lftarmjnts=start if armname == "lft" else
                                                                        numikrms[i - 1][2],
                                                                        rgtarmjnts=start if armname == "rgt" else
                                                                        numikrms[i - 1][1],
                                                                        lftjawwidth=jawwidth[i - 1][0],
                                                                        rgtjawwidth=jawwidth[i - 1][1]), )

                                    self.inspector.add_error("rrt-hold error",
                                                             Error_info(name=f"{startid}-{endid}",
                                                                        objmat=None,
                                                                        lftarmjnts=goal if armname == "lft" else
                                                                        numikrms[i - 1][2],
                                                                        rgtarmjnts=goal if armname == "rgt" else
                                                                        numikrms[i - 1][1],
                                                                        lftjawwidth=jawwidth[i - 1][0],
                                                                        rgtjawwidth=jawwidth[i - 1][1]), )

                                breakflag = True
                                break
                            path = smoother.pathsmoothinghold(path, planner, 30)
                            npath = len(path)
                            for j in range(npath):
                                if armname == 'rgt':
                                    tempnumikrmsmp.append([0.0, np.array(path[j]), numikrms[i - 1][2]])
                                else:
                                    tempnumikrmsmp.append([0.0, numikrms[i - 1][1], np.array(path[j])])
                                robot.movearmfk(np.array(path[j]), armname=armname)
                                tempjawwidthmp.append(jawwidth[i - 1])
                                objpos, objrot = robot.getworldpose(relpos, relrot, armname)
                                tempobjmsmp.append(rm.homobuild(objpos, objrot))
                        else:
                            # if the arm is not holding an object, the object will be treated as an obstacle
                            print("Planning motion ", pathnidlist[i - 1], " and ", pathnidlist[i])
                            objcmcopy = copy.deepcopy(objcm)
                            objcmcopy.sethomomat(objms[i - 1])
                            obstaclecmlistnew = obstaclecmlist + [objcmcopy]

                            path, sampledpoints = planner.planning(obstaclecmlistnew)
                            if path is False:
                                print("Motion planning failed! restarting...")
                                if pathnidlist[i - 1] == "begin":
                                    self.removeBadNodes([pathnidlist[i][:-1]])
                                    breakflag = True
                                    break
                                if pathnidlist[i] == "end":
                                    self.removeBadNodes([pathnidlist[i - 1][:-1]])
                                    breakflag = True
                                    break
                                node0 = pathnidlist[i - 1][:-1]
                                node1 = pathnidlist[i][:-1]
                                self.removeBadEdge(node0, node1)
                                if self.inspector is not None:
                                    # inspector
                                    self.inspector.add_error("rrt error",
                                                             Error_info(name=f"{startid}-{endid}",
                                                                        objmat=objms[i - 1],
                                                                        lftarmjnts=start if armname == "lft" else
                                                                        numikrms[i - 1][2],
                                                                        rgtarmjnts=start if armname == "rgt" else
                                                                        numikrms[i - 1][1],
                                                                        lftjawwidth=jawwidth[i - 1][0],
                                                                        rgtjawwidth=jawwidth[i - 1][1]), )

                                    self.inspector.add_error("rrt error",
                                                             Error_info(name=f"{startid}-{endid}",
                                                                        objmat=objms[i - 1],
                                                                        lftarmjnts=goal if armname == "lft" else
                                                                        numikrms[i - 1][2],
                                                                        rgtarmjnts=goal if armname == "rgt" else
                                                                        numikrms[i - 1][1],
                                                                        lftjawwidth=jawwidth[i - 1][0],
                                                                        rgtjawwidth=jawwidth[i - 1][1]), )
                                breakflag = True
                                break
                            path = smoother.pathsmoothing(path, planner, 30)
                            npath = len(path)
                            for j in range(npath):
                                if armname == 'rgt':
                                    tempnumikrmsmp.append([0.0, np.array(path[j]), numikrms[i - 1][2]])
                                else:
                                    tempnumikrmsmp.append([0.0, numikrms[i - 1][1], np.array(path[j])])
                                tempjawwidthmp.append(jawwidth[i - 1])
                                tempobjmsmp.append(objms[i - 1])
                        numikrmsmp.append(tempnumikrmsmp)
                        jawwidthmp.append(tempjawwidthmp)
                        objmsmp.append(tempobjmsmp)
                print(i, len(numikrms) - 1)
                if breakflag is False:
                    # successfully finished!
                    return [objmsmp, numikrmsmp, jawwidthmp, originalpathnidlist]
                else:
                    # remov node and start new search
                    continue

    def plotgraph(self, pltfig):
        """
        plot the graph without start and goal

        :param pltfig: the matplotlib object
        :return:

        author: weiwei
        date: 20161217, sapporos
        """

        def add(num1, num2):
            try:
                total = float(num1) + float(num2)
            except ValueError:
                return None
            else:
                return total

        # biggest circle: grips; big circle: rotation; small circle: placements
        radiusplacement = 30
        radiusrot = 6
        radiusgrip = 1
        xyplacementspos = {}
        xydiscreterotspos = {}
        self.xyzglobalgrippos = {}
        self.fttpsids = []
        for i, ttpsid in enumerate(self.fttpsids):
            xydiscreterotspos[ttpsid] = {}
            self.xyzglobalgrippos[ttpsid] = {}
            xypos = [radiusplacement * math.cos(2 * math.pi / self.nfttps * i),
                     radiusplacement * math.sin(2 * math.pi / self.nfttps * i)]
            xyplacementspos[ttpsid] = xypos
            for j, anglevalue in enumerate(self.angles):
                self.xyzglobalgrippos[ttpsid][anglevalue] = {}
                xypos = [radiusrot * math.cos(math.radians(anglevalue)), radiusrot * math.sin(math.radians(anglevalue))]
                xydiscreterotspos[ttpsid][anglevalue] = \
                    [xyplacementspos[ttpsid][0] + xypos[0], xyplacementspos[ttpsid][1] + xypos[1]]
                for k, globalgripid in enumerate(self.globalgripids):
                    xypos = [radiusgrip * math.cos(2 * math.pi / len(self.globalgripids) * k),
                             radiusgrip * math.sin(2 * math.pi / len(self.globalgripids) * k)]
                    self.xyzglobalgrippos[ttpsid][anglevalue][globalgripid] = \
                        [xydiscreterotspos[ttpsid][anglevalue][0] + xypos[0],
                         xydiscreterotspos[ttpsid][anglevalue][1] + xypos[1], 0]

        # for start and goal grasps poses:

        self.xyzglobalgrippos_startgoal = {}
        for k, globalgripid in enumerate(self.grasp):
            xypos = [radiusgrip * math.cos(2 * math.pi / len(self.grasp) * k),
                     radiusgrip * math.sin(2 * math.pi / len(self.grasp) * k)]
            self.xyzglobalgrippos_startgoal[k] = [xypos[0], xypos[1], 0]

        # self.grasp, self.gridsfloatingposemat4np, self.fpgpairlist, self.fpglist, \
        # self.IKfeasibleHndover_rgt, self.IKfeasibleHndover_lft, self.jnts_rgt, self.jnts_lft

        # for handover
        nfp = len(self.fp_list)
        xdist = 10
        x = range(300, 501, xdist)
        y = range(-50, 50, int(100 * xdist / nfp))

        transitedges = []
        transferedges = []
        hotransitedges = []
        hotransferedges = []
        startrgttransferedges = []
        startlfttransferedges = []
        goalrgttransferedges = []
        goallfttransferedges = []
        startgoalrgttransferedges = []
        startgoallfttransferedges = []
        startrgttransitedges = []
        goalrgttransitedges = []
        startlfttransitedges = []
        goallfttransitedges = []
        counter = 0
        for nid0, nid1, reggedgedata in self.regg.edges(data=True):
            counter = counter + 1
            if counter > 100000:
                break
            xyzpos0 = [0, 0, 0]
            xyzpos1 = [0, 0, 0]
            if (reggedgedata['edgetype'] == 'transit') or (reggedgedata['edgetype'] == 'transfer'):
                if nid0.startswith('ho'):
                    fpind0 = int(nid0[nid0.index("s") + 1:])
                    fpgpind0 = self.regg.node[nid0]['floatingposegrippairind']
                    nfpgp = len(self.feasible_fp_grasp_pairs[fpind0])
                    xpos = x[fpind0 % len(x)]
                    ypos = y[int(fpind0 / len(x))]
                    xyzpos0 = [radiusgrip * math.cos(2 * math.pi / nfpgp * fpgpind0) + xpos,
                               radiusgrip * math.sin(2 * math.pi / nfpgp * fpgpind0) + ypos, 0]
                    if nid0.startswith('horgt'):
                        xyzpos0[1] = xyzpos0[1] - 100
                    if nid0.startswith('holft'):
                        xyzpos0[1] = xyzpos0[1] + 100
                else:
                    fttpid0 = self.regg.node[nid0]['freetabletopplacementid']
                    anglevalue0 = self.regg.node[nid0]['angle']
                    ggid0 = self.regg.node[nid0]['globalgripid']
                    tabletopposition0 = self.regg.node[nid0]['tabletopposition']
                    xyzpos0 = list(map(add, self.xyzglobalgrippos[fttpid0][anglevalue0][ggid0],
                                       [tabletopposition0[0], tabletopposition0[1], tabletopposition0[2]]))
                    if nid0.startswith('rgt'):
                        xyzpos0[1] = xyzpos0[1] - 800
                    if nid0.startswith('lft'):
                        xyzpos0[1] = xyzpos0[1] + 800
                if nid1.startswith('ho'):
                    fpind1 = int(nid1[nid1.index("s") + 1:])
                    fpgpind1 = self.regg.node[nid1]['floatingposegrippairind']
                    nfpgp = len(self.feasible_fp_grasp_pairs[fpind1])
                    xpos = x[fpind1 % len(x)]
                    ypos = y[int(fpind1 / len(x))]
                    xyzpos1 = [radiusgrip * math.cos(2 * math.pi / nfpgp * fpgpind1) + xpos,
                               radiusgrip * math.sin(2 * math.pi / nfpgp * fpgpind1) + ypos, 0]
                    if nid1.startswith('horgt'):
                        xyzpos1[1] = xyzpos1[1] - 100
                    if nid1.startswith('holft'):
                        xyzpos1[1] = xyzpos1[1] + 100
                else:
                    fttpid1 = self.regg.node[nid1]['freetabletopplacementid']
                    anglevalue1 = self.regg.node[nid1]['angle']
                    ggid1 = self.regg.node[nid1]['globalgripid']
                    tabletopposition1 = self.regg.node[nid1]['tabletopposition']
                    xyzpos1 = map(add, self.xyzglobalgrippos[fttpid1][anglevalue1][ggid1],
                                  [tabletopposition1[0], tabletopposition1[1], tabletopposition1[2]])
                    if nid1.startswith('rgt'):
                        xyzpos1[1] = xyzpos1[1] - 800
                    if nid1.startswith('lft'):
                        xyzpos1[1] = xyzpos1[1] + 800
                # 3d
                # if reggedgedata['edgetype'] == 'transit':
                #     transitedges.append([xyzpos0, xyzpos1])
                # if reggedgedata['edgetype'] == 'transfer':
                #     transferedges.append([xyzpos0, xyzpos1])
                # 2d
                # move the basic graph to x+600
                xyzpos0[0] = xyzpos0[0] + 600
                xyzpos1[0] = xyzpos1[0] + 600
                if reggedgedata['edgetype'] == 'transit':
                    transitedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'] == 'transfer':
                    if nid0.startswith('ho') or nid1.startswith('ho'):
                        hotransferedges.append([xyzpos0[:2], xyzpos1[:2]])
                    else:
                        transferedges.append([xyzpos0[:2], xyzpos1[:2]])
            elif (reggedgedata['edgetype'] == 'handovertransit'):
                fpind0 = int(nid0[nid0.index("s") + 1:])
                fpgpind0 = self.regg.node[nid0]['floatingposegrippairind']
                nfpgp = len(self.fp_list[fpind0])
                xpos = x[int(fpind0 % len(x))]
                ypos = y[int(fpind0 / len(x))]
                xyzpos0 = [radiusgrip * math.cos(2 * math.pi / nfpgp * fpgpind0) + xpos,
                           radiusgrip * math.sin(2 * math.pi / nfpgp * fpgpind0) + ypos, 0]
                if nid0.startswith('horgt'):
                    xyzpos0[1] = xyzpos0[1] - 100
                if nid0.startswith('holft'):
                    xyzpos0[1] = xyzpos0[1] + 100
                fpind1 = int(nid1[nid1.index("s") + 1:])
                fpgpind1 = self.regg.node[nid1]['floatingposegrippairind']
                nfpgp = len(self.fp_list[fpind1])
                xpos = x[int(fpind1 % len(x))]
                ypos = y[int(fpind1 / len(x))]
                xyzpos1 = [radiusgrip * math.cos(2 * math.pi / nfpgp * fpgpind1) + xpos,
                           radiusgrip * math.sin(2 * math.pi / nfpgp * fpgpind1) + ypos, 0]
                if nid1.startswith('horgt'):
                    xyzpos1[1] = xyzpos1[1] - 100
                if nid1.startswith('holft'):
                    xyzpos1[1] = xyzpos1[1] + 100
                # move the basic graph to x+600
                xyzpos0[0] = xyzpos0[0] + 600
                xyzpos1[0] = xyzpos1[0] + 600
                hotransitedges.append([xyzpos0[:2], xyzpos1[:2]])
            elif reggedgedata['edgetype'].endswith('transit'):
                gid0 = int(nid0[-nid0[::-1].index("t"):])
                gid1 = int(nid1[-nid1[::-1].index("t"):])
                tabletopposition0 = self.regg.node[nid0]['tabletopposition']
                tabletopposition1 = self.regg.node[nid1]['tabletopposition']
                xyzpos0 = list(map(add, self.xyzglobalgrippos_startgoal[gid0],
                                   [tabletopposition0[0], tabletopposition0[1], tabletopposition0[2]]))
                xyzpos1 = list(map(add, self.xyzglobalgrippos_startgoal[gid1],
                                   [tabletopposition1[0], tabletopposition1[1], tabletopposition1[2]]))
                if reggedgedata['edgetype'] == 'startrgttransit':
                    startrgttransitedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'] == 'goalrgttransit':
                    goalrgttransitedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'] == 'startlfttransit':
                    startlfttransitedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'] == 'goallfttransit':
                    goallfttransitedges.append([xyzpos0[:2], xyzpos1[:2]])
            elif reggedgedata['edgetype'].endswith('transfer'):
                if nid0.startswith('ho'):
                    fpind0 = int(nid0[nid0.index("s") + 1:])
                    fpgpind0 = self.regg.node[nid0]['floatingposegrippairind']
                    nfpgp = len(self.fp_list[fpind0])
                    xpos = x[int(fpind0 % len(x))]
                    ypos = y[int(fpind0 / len(x))]
                    xyzpos0 = [radiusgrip * math.cos(2 * math.pi / nfpgp * fpgpind0) + xpos,
                               radiusgrip * math.sin(2 * math.pi / nfpgp * fpgpind0) + ypos, 0]
                    if nid0.startswith('horgt'):
                        xyzpos0[1] = xyzpos0[1] - 100
                    if nid0.startswith('holft'):
                        xyzpos0[1] = xyzpos0[1] + 100
                    xyzpos0[0] = xyzpos0[0] + 600
                elif nid0.startswith('rgt') or nid0.startswith('lft'):
                    fttpid0 = self.regg.node[nid0]['freetabletopplacementid']
                    anglevalue0 = self.regg.node[nid0]['angle']
                    ggid0 = self.regg.node[nid0]['globalgripid']
                    tabletopposition0 = self.regg.node[nid0]['tabletopposition']
                    xyzpos0 = map(add, self.xyzglobalgrippos[fttpid0][anglevalue0][ggid0],
                                  [tabletopposition0[0], tabletopposition0[1], tabletopposition0[2]])
                    if nid0.startswith('rgt'):
                        xyzpos0[1] = xyzpos0[1] - 800
                    if nid0.startswith('lft'):
                        xyzpos0[1] = xyzpos0[1] + 800
                    xyzpos0[0] = xyzpos0[0] + 600
                else:
                    gid0 = self.regg.node[nid0]['globalgripid']
                    tabletopposition0 = self.regg.node[nid0]['tabletopposition']
                    xyzpos0 = list(map(add, self.xyzglobalgrippos_startgoal[gid0],
                                       [tabletopposition0[0], tabletopposition0[1], tabletopposition0[2]]))
                if nid1.startswith('ho'):
                    fpind1 = int(nid1[nid1.index("s") + 1:])
                    fpgpind1 = self.regg.node[nid1]['floatingposegrippairind']
                    nfpgp = len(self.fp_list[fpind1])
                    xpos = x[int(fpind1 % len(x))]
                    ypos = y[int(fpind1 / len(x))]
                    xyzpos1 = [radiusgrip * math.cos(2 * math.pi / nfpgp * fpgpind1) + xpos,
                               radiusgrip * math.sin(2 * math.pi / nfpgp * fpgpind1) + ypos, 0]
                    if nid1.startswith('horgt'):
                        xyzpos1[1] = xyzpos1[1] - 100
                    if nid1.startswith('holft'):
                        xyzpos1[1] = xyzpos1[1] + 100
                    xyzpos1[0] = xyzpos1[0] + 600
                elif nid1.startswith('lft') or nid1.startswith('rgt'):
                    fttpid1 = self.regg.node[nid1]['freetabletopplacementid']
                    anglevalue1 = self.regg.node[nid1]['angle']
                    ggid1 = self.regg.node[nid1]['globalgripid']
                    tabletopposition1 = self.regg.node[nid1]['tabletopposition']
                    xyzpos1 = list(map(add, self.xyzglobalgrippos[fttpid1][anglevalue1][ggid1],
                                       [tabletopposition1[0], tabletopposition1[1], tabletopposition1[2]]))
                    if nid1.startswith('rgt'):
                        xyzpos1[1] = xyzpos1[1] - 800
                    if nid1.startswith('lft'):
                        xyzpos1[1] = xyzpos1[1] + 800
                    xyzpos1[0] = xyzpos1[0] + 600
                else:
                    ggid1 = int(nid1[-nid1[::-1].index("t"):])
                    tabletopposition1 = self.regg.node[nid1]['tabletopposition']
                    xyzpos1 = list(map(add, self.xyzglobalgrippos_startgoal[ggid1],
                                       [tabletopposition1[0], tabletopposition1[1], tabletopposition1[2]]))
                if reggedgedata['edgetype'].startswith('startgoalrgt'):
                    startgoalrgttransferedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'].startswith('startgoallft'):
                    startgoallfttransferedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'].startswith('startrgt'):
                    startrgttransferedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'].startswith('startlft'):
                    startlfttransferedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'].startswith('goalrgt'):
                    goalrgttransferedges.append([xyzpos0[:2], xyzpos1[:2]])
                if reggedgedata['edgetype'].startswith('goallft'):
                    goallfttransferedges.append([xyzpos0[:2], xyzpos1[:2]])

            # self.gnodesplotpos[nid0] = xyzpos0[:2]
            # self.gnodesplotpos[nid1] = xyzpos1[:2]
        # 3d
        # transitec = mc3d.Line3DCollection(transitedges, colors=[0,1,1,1], linewidths=1)
        # transferec = mc3d.Line3DCollection(transferedges, colors=[0,0,0,.1], linewidths=1)
        # 2d
        transitec = mc.LineCollection(transitedges, colors=[0, 1, 1, 1], linewidths=1)
        transferec = mc.LineCollection(transferedges, colors=[0, 0, 0, .1], linewidths=1)
        hotransitec = mc.LineCollection(hotransitedges, colors=[1, 0, 1, .1], linewidths=1)
        hotransferec = mc.LineCollection(hotransferedges, colors=[.5, .5, 0, .03], linewidths=1)
        # transfer
        startrgttransferec = mc.LineCollection(startrgttransferedges, colors=[.7, 0, 0, .3], linewidths=1)
        startlfttransferec = mc.LineCollection(startlfttransferedges, colors=[.3, 0, 0, .3], linewidths=1)
        goalrgttransferec = mc.LineCollection(goalrgttransferedges, colors=[0, 0, .7, .3], linewidths=1)
        goallfttransferec = mc.LineCollection(goallfttransferedges, colors=[0, 0, .3, .3], linewidths=1)
        startgoalrgttransferec = mc.LineCollection(startgoalrgttransferedges, colors=[0, 0, .7, .3], linewidths=1)
        startgoallfttransferec = mc.LineCollection(startgoallfttransferedges, colors=[0, 0, .3, .3], linewidths=1)
        # transit
        startrgttransitec = mc.LineCollection(startrgttransitedges, colors=[0, .5, 1, .3], linewidths=1)
        startlfttransitec = mc.LineCollection(startlfttransitedges, colors=[0, .2, .4, .3], linewidths=1)
        goalrgttransitec = mc.LineCollection(goalrgttransitedges, colors=[0, .5, 1, .3], linewidths=1)
        goallfttransitec = mc.LineCollection(goallfttransitedges, colors=[0, .2, .4, .3], linewidths=1)

        ax = pltfig.add_subplot(111)
        ax.add_collection(transferec)
        ax.add_collection(transitec)
        ax.add_collection(hotransferec)
        ax.add_collection(hotransitec)
        ax.add_collection(startrgttransferec)
        ax.add_collection(startlfttransferec)
        ax.add_collection(goalrgttransferec)
        ax.add_collection(goallfttransferec)
        ax.add_collection(startgoalrgttransferec)
        ax.add_collection(startgoallfttransferec)

    def plot_regrasp_graph(self):
        """
            Function to plot the input networkx graph with custom layout and edge labels.

            Parameters:
            G (networkx.Graph): A networkx graph where nodes and edges are pre-defined with groups and labels.
        """

        G = self.regg
        # Get subgraphs for each cluster
        handover_rgt_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'handover_rgt']
        handover_lft_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'handover_lft']
        start_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'start']
        goal_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'goal']

        # Define the circular layout for each cluster
        pos_handover_rgt = nx.circular_layout(G.subgraph(handover_rgt_nodes))
        pos_handover_lft = nx.circular_layout(G.subgraph(handover_lft_nodes))
        pos_start = nx.circular_layout(G.subgraph(start_nodes))
        pos_goal = nx.circular_layout(G.subgraph(goal_nodes))

        # Scale and offset the clusters
        scale_factor_handover = 2.0
        scale_factor_start = 1.0
        scale_factor_goal = 1.0

        for node in pos_handover_rgt:
            pos_handover_rgt[node] *= scale_factor_handover
        for node in pos_handover_lft:
            pos_handover_lft[node] *= scale_factor_handover
        for node in pos_start:
            pos_start[node] *= scale_factor_start
        for node in pos_goal:
            pos_goal[node] *= scale_factor_goal

        # Offset the positions to separate the clusters
        offset_handover_rgt = np.array([-5, 5])  # Move handover cluster to the left
        offset_handover_lft = np.array([5, 5])  # Move handover cluster to the left
        offset_start = np.array([0, 0])  # Move start cluster to the right
        offset_goal = np.array([0, 10])  # Move goal cluster down

        for node in pos_handover_rgt:
            pos_handover_rgt[node] += offset_handover_rgt
        for node in pos_handover_lft:
            pos_handover_lft[node] += offset_handover_lft
        for node in pos_start:
            pos_start[node] += offset_start
        for node in pos_goal:
            pos_goal[node] += offset_goal
        # Combine all positions
        pos = {**pos_handover_rgt, **pos_handover_lft, **pos_start, **pos_goal}

        # Draw nodes with color based on groups
        node_color_map = []
        for node in G.nodes(data=True):
            if node[1].get('node_type') == 'handover_rgt':
                node_color_map.append('black')
            if node[1].get('node_type') == 'handover_lft':
                node_color_map.append('black')
            elif node[1].get('node_type') == 'start':
                node_color_map.append('lightcoral')
            elif node[1].get('node_type') == 'goal':
                node_color_map.append('lightgreen')

        # Draw the graph
        plt.figure(figsize=(20, 20))
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=.1)
        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                               node_color=node_color_map,
                               node_size=1, edgecolors='gray',
                               )
        # Draw node labels
        # nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.text(offset_handover_rgt[0], offset_handover_rgt[1], 'Handover Right', fontsize=20, ha='center',
                 color='red')  # Position near handover cluster
        plt.text(offset_handover_lft[0], offset_handover_lft[1], 'Handover Left', fontsize=20, ha='center',
                 color='red')  # Position near handover cluster
        plt.text(offset_start[0], offset_start[1], 'Start', fontsize=20, ha='center',
                 color='red')  # Position near handover cluster
        plt.text(offset_goal[0], offset_goal[1], 'Goal', fontsize=20, ha='center',
                 color='red')  # Position near handover cluster

        # show shortest paths
        print(self.direct_shortest_paths)
        for path in self.direct_shortest_paths:
            sub_G = G.subgraph(path)
            tmp_pos = {}
            for n_name in path:
                if n_name.startswith('start'):
                    pos_group = pos_start
                elif n_name.startswith('goal'):
                    pos_group = pos_goal
                elif n_name.startswith('holft'):
                    pos_group = pos_handover_lft
                elif n_name.startswith('horgt'):
                    pos_group = pos_handover_rgt
                else:
                    raise Exception("No support node cluster")
                tmp_pos[n_name] = pos_group[n_name]
            # Draw edges
            nx.draw_networkx_edges(sub_G, tmp_pos, width=3, edge_color='g')
            break
        # Display the graph
        plt.gca().set_aspect('equal')
        plt.title('Network Graph with Separated Handover, Start, Goal Clusters')
        plt.show()


if __name__ == "__main__":
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    from regrasp.utils import *

    base = wd.World(cam_pos=[3, 0, 4], lookat_pos=[0, 0, 0])
    DATA_PATH = DATA_DIR.joinpath("data_handover_hndovrinfo.pickle")
    hndovr_data = load_handover_data(DATA_PATH)
    sm = Sequencer(get_rbt_by_name("ur3e"),
                   handover_data=hndovr_data, )
