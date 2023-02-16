import numpy as np

import habitat

from modeling.utils.navigation_utils import SimpleRLEnv,  get_obs_and_pose
from core import cfg
from modeling.localNavigator_Astar import localNav_Astar
from modeling.utils.baseline_utils import read_occ_map_npy
import matplotlib.pyplot as plt

from models.predictors import get_predictor_from_options
from utils.test_utils import get_latest_model, load_model
from models.semantic_grid import SemanticGrid
import torch
import torch.nn as nn
import utils.utils as utils
import utils.test_utils as tutils
import utils.m_utils as mutils
import utils.viz_utils as vutils

device = 'cuda'

# ========================= load the occupancy map =============================
predictor = get_predictor_from_options(cfg).to('cuda')
# Needed only for models trained with multi-gpu setting
predictor = nn.DataParallel(predictor)

checkpoint_dir = "trained_weights/resnet_unet_occ_ensemble0_dataPer0-7_4"
latest_checkpoint = get_latest_model(save_dir=checkpoint_dir)
print(f"loading checkpoint: {latest_checkpoint}")
load_model(model=predictor, checkpoint_file=latest_checkpoint)
predictor.eval()


class Param():

    def __init__(self):
        self.hfov = float(90.) * np.pi / 180.
        self.xs, self.ys = torch.tensor(np.meshgrid(
            np.linspace(-1, 1, cfg.MAP.IMG_SIZE),
            np.linspace(1, -1, cfg.MAP.IMG_SIZE)),
            device='cuda')
        self.xs = self.xs.reshape(1, cfg.MAP.IMG_SIZE, cfg.MAP.IMG_SIZE)
        self.ys = self.ys.reshape(1, cfg.MAP.IMG_SIZE, cfg.MAP.IMG_SIZE)
        K = np.array([[1 / np.tan(self.hfov / 2.), 0., 0., 0.],
                      [0., 1 / np.tan(self.hfov / 2.), 0., 0.], [0., 0., 1, 0],
                      [0., 0., 0, 1]])
        self.inv_K = torch.tensor(np.linalg.inv(K), device='cuda')

        self.grid_dim = (cfg.MAP.GRID_DIM, cfg.MAP.GRID_DIM)
        self.img_size = (cfg.MAP.IMG_SIZE, cfg.MAP.IMG_SIZE)
        self.crop_size = (cfg.MAP.CROP_SIZE, cfg.MAP.CROP_SIZE)


def run_map_predictor(step_ego_grid_crops):
    input_batch = step_ego_grid_crops.to(device).unsqueeze(0)

    # Estimate average predictions from the ensemble
    print(f'input_batch.shape = {input_batch.shape}')
    mean_ensemble_spatial = predictor(input_batch)
    return mean_ensemble_spatial


split = 'test'
env_scene = 'yqstnuAEVhm'
floor_id = 0
scene_name = 'yqstnuAEVhm_0'

scene_floor_dict = np.load(
    f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy',
    allow_pickle=True).item()

# ================================ load habitat env============================================
config = habitat.get_config(
    config_paths=cfg.GENERAL.LEARNED_MAP_GG_CONFIG_PATH)
config.defrost()
config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_EPISODE_DATA_PATH
config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
if cfg.NAVI.HFOV == 360:
    config.SIMULATOR.RGB_SENSOR.WIDTH = 480
    config.SIMULATOR.RGB_SENSOR.HFOV = 120
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = 480
    config.SIMULATOR.DEPTH_SENSOR.HFOV = 120
    config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = 480
    config.SIMULATOR.SEMANTIC_SENSOR.HFOV = 120
config.freeze()
env = SimpleRLEnv(config=config)

scene_height = scene_floor_dict[env_scene][floor_id]['y']
start_pose = (0.03828, -8.55946, 0.2964)
saved_folder = cfg.SAVE.TESTING_RESULTS_FOLDER

with torch.no_grad():
    par = Param()
    # For each episode we need a new instance of a fresh global grid
    sg = SemanticGrid(1,
                      par.grid_dim,
                      cfg.MAP.CROP_SIZE,
                      cfg.MAP.CELL_SIZE,
                      spatial_labels=cfg.MAP.N_SPATIAL_CLASSES)

    if cfg.NAVI.FLAG_GT_OCC_MAP:
        occ_map_npy = np.load(
            f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/{scene_name}/BEV_occupancy_map.npy',
            allow_pickle=True).item()
    gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)

    LN = localNav_Astar(pose_range, coords_range, WH, scene_name)

    abs_poses = []
    rel_poses_list = []
    abs_poses_noisy = []
    pose_coords_list = []
    pose_coords_noisy_list = []
    stg_pos_list = []
    agent_height = []

    agent_pos = np.array([start_pose[0], scene_height, start_pose[1]])
    heading_angle = start_pose[2]

    t = 0
    obs, pose = get_obs_and_pose(env, agent_pos, heading_angle)

    agent_map_pose = (pose[0], -pose[1], -pose[2])

    chosen_frontier = (0.03828, -6.55946, 0.2964)
    #flag_plan, subgoal_coords, subgoal_pose = LN.plan_to_reach_frontier(chosen_frontier, agent_map_pose, observed_occupancy_map, step, saved_folder)

    while t < 1000:

        rgb_img = obs['rgb']
        depth_img = 10. * obs['depth'][:, :, 0]
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        ax[0].imshow(rgb_img)
        ax[0].get_xaxis().set_visible(False)
        ax[0].get_yaxis().set_visible(False)
        ax[0].set_title("rgb")
        ax[1].imshow(depth_img)
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[1].set_title("depth")
        fig.tight_layout()
        plt.show()
        # fig.savefig(f'{saved_folder}/step_{step}_obs.jpg')
        # plt.close()

        depth_abs = obs['depth'].reshape(cfg.MAP.IMG_SIZE, cfg.MAP.IMG_SIZE, 1)
        depth_abs = torch.tensor(depth_abs).to(device)
        local3D_step = utils.depth_to_3D(depth_abs, par.img_size, par.xs,
                                         par.ys, par.inv_K)

        agent_pose, y_height = utils.get_sim_location(
            agent_state=env.habitat_env.sim.get_agent_state())

        abs_poses.append(agent_pose)
        agent_height.append(y_height)

        # get the relative pose with respect to the first pose in the sequence
        rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0])
        _rel_pose = torch.Tensor(rel).unsqueeze(0).float()
        _rel_pose = _rel_pose.to(device)
        rel_poses_list.append(_rel_pose.clone())

        pose_coords = tutils.get_coord_pose(sg, _rel_pose, abs_poses[0],
                                            par.grid_dim[0], cfg.MAP.CELL_SIZE,
                                            device)  # B x T x 3
        pose_coords_list.append(pose_coords.clone().cpu().numpy())
        '''
		if t==0:
			# get gt map from initial agent pose for visualization at end of episode
			x, y, label_seq = map_utils.slice_scene(x=self.test_ds.pcloud[0].copy(),
													y=self.test_ds.pcloud[1].copy(),
													z=self.test_ds.pcloud[2].copy(),
													label_seq=self.test_ds.label_seq_spatial.copy(),
													height=agent_height[0])
			gt_map_initial = map_utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[0],
														grid_dim=self.test_ds.grid_dim, cell_size=self.test_ds.cell_size)
		'''

        # do ground-projection, update the map
        ego_grid_sseg_3 = mutils.est_occ_from_depth(
            [local3D_step],
            grid_dim=par.grid_dim,
            cell_size=cfg.MAP.CELL_SIZE,
            device=device,
            occupancy_height_thresh=cfg.MAP.OCCUPANCY_HEIGHT_THRESH)
        #print(f'ego_grid_sseg_3.shape = {ego_grid_sseg_3.shape}')

        # Transform the ground projected egocentric grids to geocentric using relative pose
        geo_grid_sseg = sg.spatialTransformer(
            grid=ego_grid_sseg_3,
            pose=rel_poses_list[t],
            abs_pose=torch.tensor(abs_poses).to(device))
        # step_geo_grid contains the map snapshot every time a new observation is added
        step_geo_grid_sseg = sg.update_proj_grid_bayes(
            geo_grid=geo_grid_sseg.unsqueeze(0))
        # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
        step_ego_grid_sseg = sg.rotate_map(
            grid=step_geo_grid_sseg.squeeze(0),
            rel_pose=rel_poses_list[t],
            abs_pose=torch.tensor(abs_poses).to(device))
        # Crop the grid around the agent at each timestep
        #print(f'step_ego_grid_sseg.shape = {step_ego_grid_sseg.shape}')
        step_ego_grid_crops = mutils.crop_grid(grid=step_ego_grid_sseg,
                                               crop_size=par.crop_size)
        #print(f'step_ego_grid_crops.shape = {step_ego_grid_crops.shape}')

        mean_ensemble_spatial = run_map_predictor(step_ego_grid_crops)

        # add occupancy prediction to semantic map
        sg.register_occ_pred(prediction_crop=mean_ensemble_spatial,
                             pose=_rel_pose,
                             abs_pose=torch.tensor(abs_poses, device=device))

        # ============ take action, move
        t += 1
        '''
		action, next_pose = LN.next_action(env, scene_height)
		agent_pos = np.array([next_pose[0], scene_height, next_pose[1]])
		# output rot is negative of the input angle
		agent_rot = habitat_sim.utils.common.quat_from_angle_axis(-next_pose[2], habitat_sim.geo.GRAVITY)
		obs = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)
		'''
        #obs = env.step('TURN_RIGHT')[0]

        color_spatial_pred = vutils.colorize_grid(mean_ensemble_spatial,
                                                  color_mapping=3)
        im_spatial_pred = color_spatial_pred[0, 0, :, :, :].permute(
            1, 2, 0).cpu().numpy()
        plt.imshow(im_spatial_pred)
        plt.show()

        print(f'sg.occ_grid.shape = {sg.occ_grid.shape}')
        color_occ_grid = vutils.colorize_grid(sg.occ_grid.unsqueeze(1),
                                              color_mapping=3)
        im = color_occ_grid[0, 0, :, :, :].permute(1, 2, 0).cpu().numpy()
        plt.imshow(im)
        plt.show()
