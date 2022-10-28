import os
import numpy as np
import gzip
import json
import matplotlib.pyplot as plt
import math
import glob
from modeling.utils.baseline_utils import read_occ_map_npy, pose_to_coords
from modeling.localNavigator_Astar import localNav_Astar
import scipy.ndimage

split = 'val'
scene_floor_folder = 'output/scene_height_distribution'
saved_folder = 'output/point_goal_episodes'

scene_floor_dict = np.load(f'{scene_floor_folder}/{split}_scene_floor_dict.npy', allow_pickle=True).item()

gap_thresh = 0.01

filename = f'data/datasets/pointnav/mp3d/v1/{split}/{split}.json.gz'
with gzip.open(filename , 'rb') as f:
	data = json.loads(f.read())

episodes = data['episodes']

dict_episodes = {}

#============================= summarize the start point y of each scene =========================
for episode in episodes:
	scene_id = episode['scene_id']

	pos_slash = scene_id.rfind('/')
	pos_dot = scene_id.rfind('.')
	episode_scene = scene_id[pos_slash+1:pos_dot]

	start_pose_y = episode['start_position'][1]
	scene_y = scene_floor_dict[episode_scene][0]['y']
	env_scene = episode_scene
	scene_name = f'{env_scene}_0'

	#============================================================================
	occ_map_npy = np.load(f'output/semantic_map/{split}/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
	gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)
	H, W = gt_occ_map.shape[:2]
	LN = localNav_Astar(pose_range, coords_range, WH, scene_name)

	binary_occupancy_map = gt_occ_map.copy()
	labels, nb = scipy.ndimage.label(binary_occupancy_map, structure=np.ones((3,3)))

	if abs(start_pose_y - scene_y) <= gap_thresh:
		start_position = episode['start_position']
		goal_position = episode['goals'][0]['position']
		#===================================== setup the start location ===============================#
		start_pose = np.array([start_position[0], start_pose_y, start_position[2]])  
		goal_pose = np.array([goal_position[0], start_pose_y, goal_position[2]])
		start_coord = pose_to_coords((start_pose[0], -start_pose[1]), pose_range, coords_range, WH)
		goal_coord = pose_to_coords((goal_pose[0], -goal_pose[1]), pose_range, coords_range, WH)

		if start_coord[0] >= 0 and start_coord[0] < W and start_coord[1] >= 0 and start_coord[1] < H and goal_coord[0] >= 0 and goal_coord[0] < W and goal_coord[1] >= 0 and goal_coord[1] < H:

			if binary_occupancy_map[start_coord[1], start_coord[0]] > 0:
				start_label = labels[start_coord[1], start_coord[0]]
				goal_label = labels[goal_coord[1], goal_coord[0]]

				flag_goal_reachable = (start_label == goal_label)

				if flag_goal_reachable:
					if episode_scene in dict_episodes:
						dict_episodes[episode_scene].append(episode)
					else:
						dict_episodes[episode_scene] = [episode]

#print(f'sum(episodes) = {len(episodes)}, count_epi = {count_epi}')
# save {split}_scene_floor_start_goal_points.numpy
for scene_name in list(dict_episodes.keys()):
	print(f'scene_name = {scene_name}, num_episodes = {len(dict_episodes[scene_name])}')
	np.save(f'{saved_folder}/{split}/{scene_name}.npy', dict_episodes[scene_name])




			