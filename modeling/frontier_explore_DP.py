import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor, degrees
import random
from .utils.navigation_utils import change_brightness, SimpleRLEnv, get_obs_and_pose, get_obs_and_pose_by_action, get_metrics
from .utils.baseline_utils import apply_color_to_map, pose_to_coords, gen_arrow_head_marker, read_map_npy, read_occ_map_npy, plus_theta_fn, Euclidean_Distance
from .utils.map_utils_pcd_height import SemanticMap
from .localNavigator_Astar import localNav_Astar
import habitat
import habitat_sim
import random
from core import cfg
from .utils import frontier_utils as fr_utils
from modeling.localNavigator_slam import localNav_slam
from skimage.morphology import skeletonize
from modeling.utils.UNet import UNet
import torch
from collections import OrderedDict

def nav_DP(split, env, episode_id, scene_name, scene_height, start_pose, goal_pose, start_goal_geodesic_distance, saved_folder, device):
	"""Major function for navigation.
	
	Takes in initialized habitat environment and start location.
	Start exploring the environment.
	Explore detected frontiers.
	Use local navigator to reach the frontiers.
	When reach the limited number of steps, compute the explored area and return the numbers.
	"""

	act_dict = {-1: 'Done', 0: 'stop', 1: 'forward', 2: 'left', 3:'right'}

	#============================ get scene ins to cat dict
	scene = env.semantic_annotations()
	ins2cat_dict = {
		int(obj.id.split("_")[-1]): obj.category.index()
		for obj in scene.objects
	}

	#=================================== start original navigation code ========================
	np.random.seed(cfg.GENERAL.RANDOM_SEED)
	random.seed(cfg.GENERAL.RANDOM_SEED)

	if True: #cfg.NAVI.GT_OCC_MAP_TYPE == 'NAV_MESH':
		if cfg.EVAL.SIZE == 'small':
			occ_map_npy = np.load(
				f'output/semantic_map/{split}/{scene_name}/BEV_occupancy_map.npy',
				allow_pickle=True).item()
		elif cfg.EVAL.SIZE == 'large':
			occ_map_npy = np.load(
				f'output/large_scale_semantic_map/{scene_name}/BEV_occupancy_map.npy',
				allow_pickle=True).item()
	gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)
	H, W = gt_occ_map.shape[:2]
	# for computing gt skeleton
	if cfg.NAVI.D_type == 'Skeleton':
		skeleton = skeletonize(gt_occ_map)
		if cfg.NAVI.PRUNE_SKELETON:
			skeleton = fr_utils.prune_skeleton(gt_occ_map, skeleton)

	#===================================== load modules ==========================================
	if cfg.NAVI.PERCEPTION == 'UNet_Potential':
		unet_model = UNet(n_channel_in=cfg.PRED.PARTIAL_MAP.INPUT_CHANNEL, n_class_out=cfg.PRED.PARTIAL_MAP.OUTPUT_CHANNEL).to(device)

		if cfg.NAVI.STRATEGY == 'DP':
			if cfg.PRED.PARTIAL_MAP.INPUT == 'occ_and_sem':
				checkpoint = torch.load(f'{cfg.PRED.PARTIAL_MAP.SAVED_FOLDER}/{cfg.PRED.PARTIAL_MAP.INPUT}/experiment_25/best_checkpoint.pth.tar', map_location=device)
			elif cfg.PRED.PARTIAL_MAP.INPUT == 'occ_only':
				checkpoint = torch.load(f'run/MP3D/unet/experiment_5/checkpoint.pth.tar', map_location=device)
		
			new_state_dict = OrderedDict()
			for k, v in checkpoint['state_dict'].items():
				name = k[7:] #remove 'module'
				new_state_dict[name] = v
			unet_model.load_state_dict(new_state_dict)
		
		unet_model.eval()

	LN = localNav_Astar(pose_range, coords_range, WH, scene_name)

	LS = localNav_slam(pose_range, coords_range, WH, mark_locs=True, close_small_openings=False, recover_on_collision=False, 
	fix_thrashing=False, point_cnt=2)
	LS.reset(gt_occ_map)

	semMap_module = SemanticMap(split, scene_name, pose_range, coords_range, WH,
								ins2cat_dict)  # build the observed sem map
	traverse_lst = []
	action_lst = []
	agent_episode_distance = 0.0
	
	#===================================== setup the start location ===============================#
	goal_coord = pose_to_coords((goal_pose[0], -goal_pose[1]), pose_range, coords_range, WH)
	agent_pos = np.array([start_pose[0], scene_height,
						  start_pose[1]])  # (6.6, -6.9), (3.6, -4.5)
	# check if the start point is navigable
	if not env.is_navigable(agent_pos):
		print(f'start pose is not navigable ...')
		assert 1 == 2

	# ================= get the area connected with start pose ==============
	gt_reached_area = LN.get_start_pose_connected_component((agent_pos[0], -agent_pos[2], 0), gt_occ_map)

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
	previous_pose = pose_list[-1]
	# for model state transition
	subgoal_coords = None
	subgoal_pose = None
	MODE_FIND_SUBGOAL = True
	explore_steps = 0
	MODE_FIND_GOAL = False
	# for frontiers
	visited_frontier = set()
	chosen_frontier = None
	old_frontiers = None
	frontiers = None

	try:
		while step < cfg.NAVI.NUM_STEPS:
			print(f'step = {step}')

			#=============================== get agent global pose on habitat env ========================#
			pose = pose_list[-1]
			print(f'agent position = {pose[:2]}, angle = {pose[2]}')
			agent_map_pose = (pose[0], -pose[1], -pose[2])
			traverse_lst.append(agent_map_pose)

			# add the observed area
			semMap_module.build_semantic_map(obs_list, pose_list, step=step, saved_folder=saved_folder)

			if MODE_FIND_SUBGOAL:
				observed_occupancy_map, gt_occupancy_map, observed_area_flag, built_semantic_map = \
					semMap_module.get_observed_occupancy_map(agent_map_pose)

				#======================= check if goal point is reachable =============================
				if LN.evaluate_point_goal_reachable(goal_coord, agent_map_pose, observed_occupancy_map):
					subgoal_coords = goal_coord
					MODE_FIND_GOAL = True
					chosen_frontier = None
				#============================== find the nearest frontier ==========================
				else:
					if frontiers is not None:
						old_frontiers = frontiers

					frontiers = fr_utils.get_frontiers(observed_occupancy_map)
					frontiers = frontiers - visited_frontier

					frontiers, dist_occupancy_map = LN.filter_unreachable_frontiers(
						frontiers, agent_map_pose, observed_occupancy_map)

					if cfg.NAVI.PERCEPTION == 'UNet_Potential':
						frontiers = fr_utils.compute_frontier_potential(frontiers, observed_occupancy_map, gt_occupancy_map, 
							observed_area_flag, built_semantic_map, None, unet_model, device, LN, agent_map_pose)
					elif cfg.NAVI.PERCEPTION == 'Potential':
						if cfg.NAVI.D_type == 'Skeleton':
							frontiers = fr_utils.compute_frontier_potential(frontiers, observed_occupancy_map, gt_occupancy_map, 
								observed_area_flag, built_semantic_map, skeleton)
						else:
							frontiers = fr_utils.compute_frontier_potential(frontiers, observed_occupancy_map, gt_occupancy_map, 
								observed_area_flag, built_semantic_map, None)

					if old_frontiers is not None:
						frontiers = fr_utils.update_frontier_set(old_frontiers, frontiers, max_dist=5, chosen_frontier=chosen_frontier)

					if cfg.NAVI.STRATEGY == 'Optimistic':
						chosen_frontier = fr_utils.get_frontier_nearest_to_goal(agent_map_pose, frontiers, goal_coord, LN, observed_occupancy_map)
					elif cfg.NAVI.STRATEGY == 'DP':
						top_frontiers = fr_utils.select_top_frontiers(frontiers,
																	  top_n=6)
						chosen_frontier = fr_utils.get_frontier_with_DP(top_frontiers, agent_map_pose, dist_occupancy_map, \
						 cfg.NAVI.NUM_STEPS-step, LN)

					subgoal_coords = (int(chosen_frontier.centroid[1]), int(chosen_frontier.centroid[0]))

				MODE_FIND_SUBGOAL = False

			#============================================= visualize semantic map ===========================================#
			if cfg.NAVI.FLAG_VISUALIZE_MIDDLE_TRAJ:
				#==================================== visualize the path on the map ==============================
				#built_semantic_map, observed_area_flag, _ = semMap_module.get_semantic_map()

				#color_built_semantic_map = apply_color_to_map(built_semantic_map)
				#color_built_semantic_map = change_brightness(color_built_semantic_map, observed_area_flag, value=60)

				#=================================== visualize the agent pose as red nodes =======================
				x_coord_lst, z_coord_lst, theta_lst = [], [], []
				for cur_pose in traverse_lst:
					x_coord, z_coord = pose_to_coords(
						(cur_pose[0], cur_pose[1]), pose_range, coords_range,
						WH)
					x_coord_lst.append(x_coord)
					z_coord_lst.append(z_coord)
					theta_lst.append(cur_pose[2])

				#'''
				fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
				ax.imshow(observed_occupancy_map, cmap='gray')
				marker, scale = gen_arrow_head_marker(theta_lst[-1])
				ax.scatter(x_coord_lst[-1],
						   z_coord_lst[-1],
						   marker=marker,
						   s=(30 * scale)**2,
						   c='red',
						   zorder=5)
				ax.scatter(goal_coord[0], goal_coord[1], marker='*', s=50, c='cyan', zorder=5)
				ax.scatter(x_coord_lst, 
						   z_coord_lst, 
						   c=range(len(x_coord_lst)), 
						   cmap='viridis', 
						   s=np.linspace(5, 2, num=len(x_coord_lst))**2, 
						   zorder=3)
				if not MODE_FIND_GOAL:
					for f in frontiers:
						ax.scatter(f.points[1], f.points[0], c='yellow', zorder=2)
						ax.scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
					if chosen_frontier is not None:
						ax.scatter(chosen_frontier.points[1],
								   chosen_frontier.points[0],
								   c='green',
								   zorder=4)
						ax.scatter(chosen_frontier.centroid[1],
								   chosen_frontier.centroid[0],
								   c='red',
								   zorder=4)
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)

				fig.tight_layout()
				plt.title('observed area')
				#plt.show()
				fig.savefig(f'{saved_folder}/step_{step}_semmap.jpg')
				plt.close()
				#assert 1==2
				#'''

			#===================================== check if exploration is done ========================
			if (chosen_frontier is None) and (not MODE_FIND_GOAL):
				print('There are no more frontiers to explore. Stop navigation.')
				break

			#====================================== take next action ================================
			act, act_seq = LS.plan_to_reach_subgoal(agent_map_pose, subgoal_coords, observed_occupancy_map)
			#print(f'subgoal_coords = {subgoal_coords}')
			#print(f'action = {act_dict[act]}')
			action_lst.append(act)
			
			if act == -1 or act == 0: # finished navigating to the subgoal
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

			# compute agent traveled distance
			current_pose = pose_list[-1]
			agent_episode_distance += Euclidean_Distance(current_pose, previous_pose)
			previous_pose = current_pose

			if explore_steps == cfg.NAVI.NUM_STEPS_EXPLORE:
				explore_steps = 0
				MODE_FIND_SUBGOAL = True
	except:
		print(f'running into a Bug running navigation. scene_name={scene_name}, episode_id={episode_id}')

	#============================================ Finish exploration =============================================
	if cfg.NAVI.FLAG_VISUALIZE_FINAL_TRAJ:
		#==================================== visualize the path on the map ==============================
		built_semantic_map, observed_area_flag, _ = semMap_module.get_semantic_map(
		)

		color_built_semantic_map = apply_color_to_map(built_semantic_map)
		color_built_semantic_map = change_brightness(color_built_semantic_map,
													 observed_area_flag,
													 value=60)

		#=================================== visualize the agent pose as red nodes =======================
		x_coord_lst, z_coord_lst, theta_lst = [], [], []
		for cur_pose in traverse_lst:
			x_coord, z_coord = pose_to_coords((cur_pose[0], cur_pose[1]),
											  pose_range, coords_range, WH)
			x_coord_lst.append(x_coord)
			z_coord_lst.append(z_coord)
			theta_lst.append(cur_pose[2])

		#'''
		fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
		ax[0].imshow(observed_occupancy_map, cmap='gray')
		marker, scale = gen_arrow_head_marker(theta_lst[-1])
		ax[0].scatter(x_coord_lst[-1],
					  z_coord_lst[-1],
					  marker=marker,
					  s=(30 * scale)**2,
					  c='red',
					  zorder=5)
		ax[0].scatter(x_coord_lst[0],
					  z_coord_lst[0],
					  marker='s',
					  s=50,
					  c='red',
					  zorder=5)
		ax[0].scatter(goal_coord[0], goal_coord[1], marker='*', s=50, c='cyan', zorder=5)
		ax[0].scatter(x_coord_lst, 
				   z_coord_lst, 
				   c=range(len(x_coord_lst)), 
				   cmap='viridis', 
				   s=np.linspace(5, 2, num=len(x_coord_lst))**2, 
				   zorder=3)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title('improved observed_occ_map + frontiers')

		ax[1].imshow(color_built_semantic_map)
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title('built semantic map')

		fig.tight_layout()
		plt.title('observed area')
		#plt.show()
		fig.savefig(f'{saved_folder}/final_semmap.jpg')
		plt.close()
		#assert 1==2
		#'''

	#====================================== compute statistics =================================
	if cfg.NAVI.PERCEPTION == 'UNet_Potential':
		del unet_model
		del checkpoint

	nav_metrics = get_metrics(env=env,
		episode_goal_positions=[goal_pose[0], scene_height, goal_pose[1]],
		success_distance=cfg.EVAL.TASK_SUCCESS_DISTANCE,
		start_end_episode_distance=start_goal_geodesic_distance,
		agent_episode_distance=agent_episode_distance,
		stop_signal=True
		)

	return step, traverse_lst, action_lst, nav_metrics
