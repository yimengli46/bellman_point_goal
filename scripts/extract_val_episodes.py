import os
import numpy as np
import gzip
import json
import matplotlib.pyplot as plt
import math
import glob

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
	if abs(start_pose_y - scene_y) <= gap_thresh:
		if episode_scene in dict_episodes:
			dict_episodes[episode_scene].append(episode)
		else:
			dict_episodes[episode_scene] = [episode]

#print(f'sum(episodes) = {len(episodes)}, count_epi = {count_epi}')
# save {split}_scene_floor_start_goal_points.numpy
for scene_name in list(dict_episodes.keys()):
	np.save(f'{saved_folder}/{split}/{scene_name}.npy', dict_episodes[scene_name])




			