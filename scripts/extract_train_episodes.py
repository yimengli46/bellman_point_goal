import os
import numpy as np
import gzip
import json
import matplotlib.pyplot as plt
import math
import glob

split = 'train'
scene_floor_folder = 'output/scene_height_distribution'
saved_folder = 'output/point_goal_episodes'

scene_floor_dict = np.load(f'{scene_floor_folder}/{split}_scene_floor_dict.npy', allow_pickle=True).item()

gap_thresh = 0.01


foldername = f'data/datasets/pointnav/mp3d/v1/{split}/content/*.json.gz'
filenames = [os.path.basename(x) for x in glob.glob(foldername)]
print(f'There are {len(filenames)} scenes in the training episodes data.')

for filename in filenames:
	print(f'filename is {filename}.')
	with gzip.open(f'data/datasets/pointnav/mp3d/v1/{split}/content/{filename}' , 'rb') as f:
		data = json.loads(f.read())
		episodes = data['episodes']

		scene_y = scene_floor_dict[filename[:-8]][0]['y']
		count_epi = 0
		list_episodes = []

		#============================= summarize the start point y of each scene =========================
		for episode in episodes:
			scene_id = episode['scene_id']	

			pos_slash = scene_id.rfind('/')
			pos_dot = scene_id.rfind('.')
			episode_scene = scene_id[pos_slash+1:pos_dot]

			start_pose_y = episode['start_position'][1]

			if abs(start_pose_y - scene_y) <= gap_thresh:
				count_epi += 1
				list_episodes.append(episode)

		print(f'sum(episodes) = {len(episodes)}, count_epi = {count_epi}')

		# save {split}_scene_floor_start_goal_points.numpy
		np.save(f'{saved_folder}/{split}/{filename[:-8]}.npy', list_episodes)
