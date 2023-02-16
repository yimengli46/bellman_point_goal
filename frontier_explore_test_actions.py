import numpy as np
import matplotlib.pyplot as plt
import random
from modeling.utils.navigation_utils import get_obs_and_pose, get_obs_and_pose_by_action
from modeling.utils.baseline_utils import pose_to_coords, gen_arrow_head_marker, read_occ_map_npy, plus_theta_fn
from modeling.utils.map_utils_pcd_height import SemanticMap
from modeling.localNavigator_Astar import localNav_Astar
import habitat
from core import cfg
import modeling.utils.frontier_utils as fr_utils
from timeit import default_timer as timer
from modeling.localNavigator_slam import localNav_slam
from skimage.morphology import skeletonize
from modeling.utils.UNet import UNet
import torch
from collections import OrderedDict

split = 'test'  # 'test' #'train'
env_scene = 'yqstnuAEVhm'  # '17DRP5sb8fy' #'yqstnuAEVhm'
floor_id = 0
scene_name = 'yqstnuAEVhm_0'  # '17DRP5sb8fy_0' #'yqstnuAEVhm_0'

cfg.merge_from_file(
    'configs/large_exp_90degree_Optimistic_PCDHEIGHT_MAP_1STEP_500STEPS.yaml')
cfg.freeze()

if cfg.EVAL.SIZE == 'small':
    scene_floor_dict = np.load(
        f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy',
        allow_pickle=True).item()
elif cfg.EVAL.SIZE == 'large':
    scene_floor_dict = np.load(
        f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/large_scale_{split}_scene_floor_dict.npy',
        allow_pickle=True).item()

act_dict = {-1: 'Done', 0: 'stop', 1: 'forward', 2: 'left', 3: 'right'}

# ================================ load habitat env============================================
config = habitat.get_config(config_paths=cfg.GENERAL.DATALOADER_CONFIG_PATH)
config.defrost()
if split == 'train':
    config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_TRAIN_EPISODE_DATA_PATH
elif split == 'test':
    config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_TEST_EPISODE_DATA_PATH
config.SIMULATOR.SCENE = f'{cfg.GENERAL.HABITAT_SCENE_DATA_PATH}/mp3d/{env_scene}/{env_scene}.glb'
config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
config.freeze()

env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
env.reset()

scene_height = scene_floor_dict[env_scene][floor_id]['y']
start_pose, goal_pose, _ = scene_floor_dict[env_scene][floor_id][
    'start_goal_pair'][1]
saved_folder = f'output/TESTING_RESULTS_Frontier'

# ============================ get scene ins to cat dict
scene = env.semantic_annotations()
ins2cat_dict = {
    int(obj.id.split("_")[-1]): obj.category.index()
    for obj in scene.objects
}

# =================================== start original navigation code ========================
np.random.seed(cfg.GENERAL.RANDOM_SEED)
random.seed(cfg.GENERAL.RANDOM_SEED)

if True:  # cfg.NAVI.GT_OCC_MAP_TYPE == 'NAV_MESH':
    occ_map_npy = np.load(
        f'output/semantic_map/{split}/{scene_name}/BEV_occupancy_map.npy',
        allow_pickle=True).item()
gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)
H, W = gt_occ_map.shape[:2]
# for computing gt skeleton
if cfg.NAVI.D_type == 'Skeleton':
    skeleton = skeletonize(gt_occ_map)
    if cfg.NAVI.PRUNE_SKELETON:
        skeleton = fr_utils.prune_skeleton(gt_occ_map, skeleton)

# ===================================== load modules ==========================================
device = torch.device('cuda:0')
if cfg.NAVI.PERCEPTION == 'UNet_Potential':
    unet_model = UNet(
        n_channel_in=cfg.PRED.PARTIAL_MAP.INPUT_CHANNEL,
        n_class_out=cfg.PRED.PARTIAL_MAP.OUTPUT_CHANNEL).to(device)
    if cfg.PRED.PARTIAL_MAP.INPUT == 'occ_and_sem':
        checkpoint = torch.load(
            f'{cfg.PRED.PARTIAL_MAP.SAVED_FOLDER}/{cfg.PRED.PARTIAL_MAP.INPUT}/experiment_1/best_checkpoint.pth.tar',
            map_location=device)
    elif cfg.PRED.PARTIAL_MAP.INPUT == 'occ_only':
        checkpoint = torch.load(
            f'run/MP3D/unet/experiment_5/checkpoint.pth.tar',
            map_location=device)

    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]  # remove 'module'
        new_state_dict[name] = v
    unet_model.load_state_dict(new_state_dict)

LN = localNav_Astar(pose_range, coords_range, WH, scene_name)

LS = localNav_slam(pose_range,
                   coords_range,
                   WH,
                   mark_locs=True,
                   close_small_openings=False,
                   recover_on_collision=True,
                   fix_thrashing=False,
                   point_cnt=2)
LS.reset(gt_occ_map)

semMap_module = SemanticMap(split, scene_name, pose_range, coords_range, WH,
                            ins2cat_dict)  # build the observed sem map
traverse_lst = []

# ===================================== setup the start location ===============================#
goal_coord = pose_to_coords((goal_pose[0], -goal_pose[1]), pose_range,
                            coords_range, WH)
agent_pos = np.array([start_pose[0], scene_height,
                      start_pose[1]])  # (6.6, -6.9), (3.6, -4.5)
# check if the start point is navigable
if not env.is_navigable(agent_pos):
    print(f'start pose is not navigable ...')
    assert 1 == 2

if cfg.NAVI.HFOV == 90:
    obs_list, pose_list = [], []
    heading_angle = start_pose[2]
    obs, pose = get_obs_and_pose(env, agent_pos, heading_angle)
    obs_list.append(obs)
    pose_list.append(pose)
elif cfg.NAVI.HFOV == 360:
    obs_list, pose_list = [], []
    for rot in [90, 180, 270, 0]:
        heading_angle = rot / 180 * np.pi
        heading_angle = plus_theta_fn(heading_angle, start_pose[2])
        obs, pose = get_obs_and_pose(env, agent_pos, heading_angle)
        obs_list.append(obs)
        pose_list.append(pose)

step = 0
subgoal_coords = None
subgoal_pose = None
MODE_FIND_SUBGOAL = True
explore_steps = 0
MODE_FIND_GOAL = False
visited_frontier = set()
chosen_frontier = None
old_frontiers = None
frontiers = None

while step < cfg.NAVI.NUM_STEPS:
    print(f'step = {step}')

    # =============================== get agent global pose on habitat env ========================#
    pose = pose_list[-1]
    agent_map_pose = (pose[0], -pose[1], -pose[2])
    traverse_lst.append(agent_map_pose)

    # add the observed area
    t0 = timer()
    semMap_module.build_semantic_map(obs_list,
                                     pose_list,
                                     step=step,
                                     saved_folder=saved_folder)
    t1 = timer()
    print(f'build map time = {t1 - t0}')

    if MODE_FIND_SUBGOAL:
        t1 = timer()
        observed_occupancy_map, gt_occupancy_map, observed_area_flag, built_semantic_map = semMap_module.get_observed_occupancy_map(
            agent_map_pose)
        t2 = timer()
        print(f'get occupan map time = {t2 - t1}')

        # ======================= check if goal point is reachable =============================
        if LN.evaluate_point_goal_reachable(goal_coord, agent_map_pose,
                                            observed_occupancy_map):
            subgoal_coords = goal_coord
            MODE_FIND_GOAL = True
            chosen_frontier = None
        # ============================== find the nearest frontier ==========================
        else:
            if frontiers is not None:
                old_frontiers = frontiers

            frontiers = fr_utils.get_frontiers(observed_occupancy_map)
            frontiers = frontiers - visited_frontier
            print(f'before filtering, num(frontiers) = {len(frontiers)}')
            t3 = timer()
            print(f'get frontier time = {t3 - t2}')
            frontiers, dist_occupancy_map = LN.filter_unreachable_frontiers(
                frontiers, agent_map_pose, observed_occupancy_map)
            print(f'after filtering, num(frontiers) = {len(frontiers)}')
            t4 = timer()
            print(f'filter unreachable frontiers time = {t4 - t3}')
            if cfg.NAVI.PERCEPTION == 'UNet_Potential':
                frontiers = fr_utils.compute_frontier_potential(
                    frontiers, goal_coord, dist_occupancy_map,
                    observed_occupancy_map, gt_occupancy_map,
                    observed_area_flag, built_semantic_map, None, unet_model,
                    device, LN, agent_map_pose)
            elif cfg.NAVI.PERCEPTION == 'Potential':
                if cfg.NAVI.D_type == 'Skeleton':
                    frontiers = fr_utils.compute_frontier_potential(
                        frontiers, goal_coord, dist_occupancy_map,
                        observed_occupancy_map, gt_occupancy_map,
                        observed_area_flag, built_semantic_map, skeleton)

            t5 = timer()
            print(f'compute frontier potential time = {t5 - t4}')

            if old_frontiers is not None:
                frontiers = fr_utils.update_frontier_set(
                    old_frontiers,
                    frontiers,
                    max_dist=5,
                    chosen_frontier=chosen_frontier)
            t6 = timer()
            print(f'update frontiers time = {t6 - t5}')

            if cfg.NAVI.STRATEGY == 'Optimistic':
                chosen_frontier = fr_utils.get_frontier_nearest_to_goal(
                    agent_map_pose, frontiers, goal_coord, LN,
                    observed_occupancy_map)
            elif cfg.NAVI.STRATEGY == 'DP':
                top_frontiers = fr_utils.select_top_frontiers(frontiers,
                                                              top_n=8)
                chosen_frontier = fr_utils.get_frontier_with_DP_accel(
                    top_frontiers, agent_map_pose, dist_occupancy_map,
                    goal_coord, LN)

            t7 = timer()
            print(f'select frontiers time = {t7 - t6}')

            subgoal_coords = (int(chosen_frontier.centroid[1]),
                              int(chosen_frontier.centroid[0]))

        MODE_FIND_SUBGOAL = False

    # ============================================= visualize semantic map ===========================================#
    if True:

        # =================================== visualize the agent pose as red nodes =======================
        x_coord_lst, z_coord_lst, theta_lst = [], [], []
        for cur_pose in traverse_lst:
            x_coord, z_coord = pose_to_coords((cur_pose[0], cur_pose[1]),
                                              pose_range, coords_range, WH)
            x_coord_lst.append(x_coord)
            z_coord_lst.append(z_coord)
            theta_lst.append(cur_pose[2])

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
        ax[0].imshow(observed_occupancy_map, cmap='gray')
        marker, scale = gen_arrow_head_marker(theta_lst[-1])
        ax[0].scatter(x_coord_lst[-1],
                      z_coord_lst[-1],
                      marker=marker,
                      s=(30 * scale)**2,
                      c='red',
                      zorder=5)
        ax[0].scatter(goal_coord[0],
                      goal_coord[1],
                      marker='*',
                      s=50,
                      c='yellow',
                      zorder=5)
        ax[0].plot(x_coord_lst, z_coord_lst, lw=5, c='blue', zorder=3)
        if not MODE_FIND_GOAL:
            for f in top_frontiers:
                ax[0].scatter(f.points[1], f.points[0], c='green', zorder=2)
                ax[0].scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
            if chosen_frontier is not None:
                ax[0].scatter(chosen_frontier.points[1],
                              chosen_frontier.points[0],
                              c='yellow',
                              zorder=4)
                ax[0].scatter(chosen_frontier.centroid[1],
                              chosen_frontier.centroid[0],
                              c='red',
                              zorder=4)
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)

        ax[1].imshow(observed_occupancy_map)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)

        fig.tight_layout()
        plt.title('observed area')
        plt.show()

    # ===================================== check if exploration is done ========================
    if (chosen_frontier is None) and (not MODE_FIND_GOAL):
        print('There are no more frontiers to explore. Stop navigation.')
        break

    # ================================ take next action ====================================

    act, act_seq = LS.plan_to_reach_subgoal(agent_map_pose, subgoal_coords,
                                            observed_occupancy_map)

    print(f'action = {act_dict[act]}')

    if act == -1 or act == 0:  # finished navigating to the subgoal
        if MODE_FIND_GOAL:
            print('Reached the point goal! Stop the episode.')
            break
        else:
            print(f'reached the subgoal')
            MODE_FIND_SUBGOAL = True
            visited_frontier.add(chosen_frontier)
    else:
        step += 1
        explore_steps += 1
        # output rot is negative of the input angle
        if cfg.NAVI.HFOV == 90:
            obs_list, pose_list = [], []
            obs, pose = get_obs_and_pose_by_action(env, act)
            obs_list.append(obs)
            pose_list.append(pose)
        elif cfg.NAVI.HFOV == 360:
            obs_list, pose_list = [], []
            obs, pose = get_obs_and_pose_by_action(env, act)
            next_pose = pose
            agent_pos = np.array([next_pose[0], scene_height, next_pose[1]])
            for rot in [90, 180, 270, 0]:
                heading_angle = rot / 180 * np.pi
                heading_angle = plus_theta_fn(heading_angle, -next_pose[2])
                obs, pose = get_obs_and_pose(env, agent_pos, heading_angle)
                obs_list.append(obs)
                pose_list.append(pose)

    if explore_steps == cfg.NAVI.NUM_STEPS_EXPLORE:
        explore_steps = 0
        MODE_FIND_SUBGOAL = True

if cfg.NAVI.PERCEPTION == 'UNet_Potential':
    del unet_model
    del checkpoint

env.close()
