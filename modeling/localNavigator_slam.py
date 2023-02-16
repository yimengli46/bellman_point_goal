from .utils.fmm_planner import FMMPlanner
import skimage
import matplotlib.pyplot as plt
from core import cfg
from .utils.baseline_utils import pose_to_coords, pxl_coords_to_pose, gen_arrow_head_marker
from math import pi


class localNav_slam(object):

    def __init__(self,
                 pose_range,
                 coords_range,
                 WH,
                 dt=30,
                 mark_locs=False,
                 reset_if_drift=False,
                 count=-1,
                 close_small_openings=False,
                 recover_on_collision=False,
                 fix_thrashing=False,
                 goal_f=1.1,
                 point_cnt=2):

        self.pose_range = pose_range
        self.coords_range = coords_range
        self.WH = WH

        self.dt = dt
        self.count = count
        self.mark_locs = mark_locs
        self.reset_if_drift = reset_if_drift
        self.close_small_openings = close_small_openings
        self.num_erosions = 2
        self.recover_on_collision = recover_on_collision
        self.fix_thrashing = fix_thrashing
        self.goal_f = goal_f
        self.point_cnt = point_cnt

        self.act_trans_dict = {0: 1, 1: 2, 2: 3, 3: 0}

    def reset(self, occ_map, soft=False):
        self.RESET = False

        self.selem = skimage.morphology.disk(
            int(cfg.SENSOR.AGENT_RADIUS / cfg.SEM_MAP.CELL_SIZE))
        self.selem_small = skimage.morphology.disk(1)
        # 0 agent moves forward. Agent moves in the direction of +x
        # 1 rotates left
        # 2 rotates right
        # 3 agent stop
        self.visited = np.zeros(occ_map.shape, dtype=np.int16)
        self.collision_map = np.zeros(occ_map.shape, dtype=np.int16)

        self.current_loc = None

        self.goal_loc = None
        self.last_act = 3
        self.locs = []
        self.acts = []
        self.recovery_actions = []
        self.flag_collision = False
        self.num_resets = 0
        self.col_width = 1

    def plan_to_reach_subgoal(self, agent_map_pose, subgoal_coords, occ_map):
        '''
                # check collision
                if self.current_loc is not None and self.last_act == 0: # last action is moving forward
                        if self.check_drift(agent_map_pose) and len(self.recovery_actions) == 0:
                                if self.num_resets == 6:
                                        print(f'!! collision detected, entering recovery actions ...')
                                        num_rots = int(np.round(180 / self.dt))
                                        self.recovery_actions = [1]*num_rots + [0]*6
                                        self.flag_collision = True
                                        self.num_resets = 0
                                else:
                                        print(f'!! collision detected, do nothing ...')
                                        self.num_resets += 1
                '''

        # ============================ collision check ===========================
        if self.current_loc is not None and self.last_act == 0:
            x1, y1, t1 = self.current_loc
            x2, y2, _ = agent_map_pose
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            if self.check_drift(agent_map_pose):  # collison
                # if True:
                print(f'!! collision detected, do nothing ...')
                self.flag_collision = True
                width = self.col_width
                for i in range(length):
                    for j in range(width):

                        wx = x1 + 0.05 * (i + buf) * np.sin(t1) + 0.05 * (
                            j - width // 2) * np.cos(t1)
                        wy = y1 + 0.05 * (i + buf) * np.cos(t1) - 0.05 * (
                            j - width // 2) * np.sin(t1)
                        cell_coords = pose_to_coords([wx, wy], self.pose_range,
                                                     self.coords_range,
                                                     self.WH)
                        self.collision_map[cell_coords[1], cell_coords[0]] = 1

        # ===========================================================================
        self.current_loc = agent_map_pose

        subgoal_pose = pxl_coords_to_pose(subgoal_coords, self.pose_range,
                                          self.coords_range, self.WH)

        agent_coords = pose_to_coords(agent_map_pose, self.pose_range,
                                      self.coords_range, self.WH)
        self.mark_on_map(agent_coords)
        theta = agent_map_pose[2]
        theta = (theta - .5 * pi)
        state = np.array([agent_coords[0], agent_coords[1], theta])

        # =========================== update traversible map =========================

        # dilate the obstacle in the configuration space
        obstacle = (occ_map == cfg.FE.COLLISION_VAL)
        if cfg.NAVI.GT_OCC_MAP_TYPE == 'PCD_HEIGHT':
            traversible = skimage.morphology.binary_dilation(
                obstacle, self.selem) != True
        else:
            traversible = (obstacle != True)
        traversible_original = traversible.copy()

        # add the collision map
        traversible[self.collision_map == 1] = 0

        # add visited map
        traversible[self.visited == 1] = 1

        # add current loc
        traversible[agent_coords[1] - 1:agent_coords[1] + 2,
                    agent_coords[0] - 1:agent_coords[0] + 2] = 1

        # add goal loc
        traversible[subgoal_coords[1] - 1:subgoal_coords[1] + 2,
                    subgoal_coords[0] - 1:subgoal_coords[0] + 2] = 1

        # '''
        if self.flag_collision:
            if False:
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 10))
                ax[0].imshow(traversible_original, cmap='gray')
                marker, scale = gen_arrow_head_marker(agent_map_pose[2])
                ax[0].scatter(agent_coords[0],
                              agent_coords[1],
                              marker=marker,
                              s=(30 * scale)**2,
                              c='red',
                              zorder=5)
                ax[1].imshow(traversible, cmap='gray')
                ax[2].imshow(self.collision_map)
                ax[2].scatter(agent_coords[0],
                              agent_coords[1],
                              marker=marker,
                              s=(10 * scale)**2,
                              c='red',
                              zorder=5)
                fig.tight_layout()
                plt.title('observed area')
                plt.show()
            self.flag_collision = False
        # '''

        # ================================== planning =================================
        planner = FMMPlanner(traversible, 360 // self.dt)
        goal_loc_int = np.array(subgoal_coords).astype(np.int32)
        reachable = planner.set_goal(goal_loc_int)

        self.fmm_dist = planner.fmm_dist * 1
        a, state, act_seq = planner.get_action(state)
        for i in range(len(act_seq)):
            if act_seq[i] == 3:
                act_seq[i] = 0
            elif act_seq[i] == 0:
                act_seq[i] = 3
        if a == 3:
            a = 0
        elif a == 0:
            a = 3

        self.act_seq = act_seq
        self.last_act = a

        act = self.act_trans_dict[a]
        return act, act_seq

    def plan_to_reach_frontier_for_fig(self, agent_map_pose, chosen_frontier,
                                       occ_map):

        # ===========================================================================
        self.current_loc = agent_map_pose
        fron_centroid_coords = (int(chosen_frontier.centroid[1]),
                                int(chosen_frontier.centroid[0]))

        subgoal_coords = fron_centroid_coords
        subgoal_pose = pxl_coords_to_pose(subgoal_coords, self.pose_range,
                                          self.coords_range, self.WH)

        agent_coords = pose_to_coords(agent_map_pose, self.pose_range,
                                      self.coords_range, self.WH)
        # self.mark_on_map(agent_coords)
        theta = agent_map_pose[2]
        theta = (theta - .5 * pi)
        state = np.array([agent_coords[0], agent_coords[1], theta])

        # =========================== update traversible map =========================

        # dilate the obstacle in the configuration space
        obstacle = (occ_map == cfg.FE.COLLISION_VAL)
        if cfg.NAVI.GT_OCC_MAP_TYPE == 'PCD_HEIGHT':
            traversible = skimage.morphology.binary_dilation(
                obstacle, self.selem) != True
        else:
            traversible = (obstacle != True)
        traversible_original = traversible.copy()

        # add the collision map
        traversible[self.collision_map == 1] = 0

        # add visited map
        traversible[self.visited == 1] = 1

        # add current loc
        traversible[agent_coords[1] - 1:agent_coords[1] + 2,
                    agent_coords[0] - 1:agent_coords[0] + 2] = 1

        # add goal loc
        traversible[subgoal_coords[1] - 1:subgoal_coords[1] + 2,
                    subgoal_coords[0] - 1:subgoal_coords[0] + 2] = 1

        # '''
        if self.flag_collision:
            if False:
                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 10))
                ax[0].imshow(traversible_original, cmap='gray')
                marker, scale = gen_arrow_head_marker(agent_map_pose[2])
                ax[0].scatter(agent_coords[0],
                              agent_coords[1],
                              marker=marker,
                              s=(30 * scale)**2,
                              c='red',
                              zorder=5)
                ax[1].imshow(traversible, cmap='gray')
                ax[2].imshow(self.collision_map)
                ax[2].scatter(agent_coords[0],
                              agent_coords[1],
                              marker=marker,
                              s=(10 * scale)**2,
                              c='red',
                              zorder=5)
                fig.tight_layout()
                plt.title('observed area')
                plt.show()
            self.flag_collision = False
        # '''

        # ================================== planning =================================
        planner = FMMPlanner(traversible, 360 // self.dt)
        goal_loc_int = np.array(subgoal_coords).astype(np.int32)
        reachable = planner.set_goal(goal_loc_int)

        self.fmm_dist = planner.fmm_dist * 1
        a, state, act_seq = planner.get_action(state)
        for i in range(len(act_seq)):
            if act_seq[i] == 3:
                act_seq[i] = 0
            elif act_seq[i] == 0:
                act_seq[i] = 3
        if a == 3:
            a = 0
        elif a == 0:
            a = 3

        self.act_seq = act_seq
        self.last_act = a

        act = self.act_trans_dict[a]
        return act, act_seq, subgoal_coords, subgoal_pose

    def mark_on_map(self, loc):
        # input is the coordinates on the map
        self.visited[loc[1], loc[0]] = 1

    def soft_reset(self, pointgoal):
        # This reset is called if there is drift in the position of the goal
        # location, indicating that there had been collisions.
        if self.out_dir is not None:
            self.save_vis()
        self._reset(pointgoal[0] * 100., soft=True)
        self.trials = self.trials + 1
        self.num_resets = self.num_resets + 1
        xy = self.compute_xy_from_pointnav(pointgoal)
        # self.current_loc has been set inside reset
        self.goal_loc = xy * 1
        self.goal_loc[0] = self.goal_loc[0] + self.current_loc[0]
        self.goal_loc[1] = self.goal_loc[1] + self.current_loc[1]
        self.mark_on_map(self.goal_loc)
        self.mark_on_map(self.current_loc)
        if self.num_resets == 6:
            # We don't want to keep resetting. First few resets fix themselves,
            # so do it for later resets.
            num_rots = int(np.round(180 / self.dt))
            self.recovery_actions = [1] * num_rots + [0] * 6
        else:
            self.recovery_actions = []

    def check_drift(self, agent_map_pose):
        previous_pose = np.array((self.current_loc[0], self.current_loc[1]))
        current_pose = np.array((agent_map_pose[0], agent_map_pose[1]))
        moving_dist = np.linalg.norm(previous_pose - current_pose)
        return moving_dist < 0.20
