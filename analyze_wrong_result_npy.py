import numpy as np 
from core import cfg
import pandas as pd
from modeling.utils.baseline_utils import read_map_npy
import matplotlib.pyplot as plt
import glob
import os
from modeling.utils.baseline_utils import Euclidean_Distance

cfg.merge_from_file(f'configs/large_exp_90degree_Optimistic_PCDHEIGHT_MAP_1STEP_500STEPS.yaml')
cfg.freeze()

output_folder = 'output' #cfg.SAVE.TESTING_RESULTS_FOLDER
result_folder = 'LARGE_TESTING_RESULTS_90degree_Optimistic_NAVMESH_MAP_1STEP_500STEPS'

if cfg.EVAL.SIZE == 'small':
	scene_floor_dict = np.load(
		f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/test_scene_floor_dict.npy',
		allow_pickle=True).item()
elif cfg.EVAL.SIZE == 'large':
	scene_floor_dict = np.load(
		f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/large_scale_test_scene_floor_dict.npy',
		allow_pickle=True).item()

df = pd.DataFrame(columns=['Scene', 'Run', 'Success', 'SPL', 'SoftSPL'])
df['Success'] = df['Success'].astype(int)
df['SPL'] = df['SPL'].astype(float)
df['SoftSPL'] = df['SoftSPL'].astype(float)

for env_scene in cfg.MAIN.TEST_SCENE_NO_FLOOR_LIST:
	scene_dict = scene_floor_dict[env_scene]
	for floor_id in list(scene_dict.keys()):
		scene_name = f'{env_scene}_{floor_id}'
		try:
			results_npy = np.load(f'{output_folder}/{result_folder}/results_{scene_name}.npy', allow_pickle=True).item()
			num_test = len(results_npy.keys())

			if num_test > 0:
				testing_data = scene_dict[floor_id]['start_goal_pair']

			for i in range(num_test):
				result = results_npy[i]
				start_pose, goal_pose, start_end_episode_distance = testing_data[i]
				
				eps_id = result['eps_id']
				if 'success' in result['nav_metrics']:
					metrics_success = result['nav_metrics']['success']
					if metrics_success > 0.:
						metrics_spl = result['nav_metrics']['spl']
					else:
						#================= recompute the metrics ======================
						traverse_lst = result['trajectory']
						agent_episode_distance = 0.
						for pose_idx in range(len(traverse_lst)-1):
							previous_pose = traverse_lst[pose_idx]
							current_pose  = traverse_lst[pose_idx+1]
							agent_episode_distance += Euclidean_Distance(current_pose, previous_pose)

						agent_map_pose = traverse_lst[-1]
						distance_to_goal = Euclidean_Distance([agent_map_pose[0], -agent_map_pose[1]], goal_pose) 

						if distance_to_goal <= 0.2 and True:
							success = 1.0
						else:
							success = 0.0

						spl = success * (start_end_episode_distance / max(start_end_episode_distance, agent_episode_distance) )
						metrics_success = success
						metrics_spl = spl

					metrics_softspl = result['nav_metrics']['softspl']

					df = df.append({'Scene': scene_name, 'Run': eps_id, 'Success': metrics_success, 'SPL': metrics_spl, 'SoftSPL': metrics_softspl}, 
						ignore_index=True)

					print(f'scene_name = {scene_name}, eps = {eps_id}, Success = {metrics_success}, SPL = {metrics_spl}')
				else:
					df = df.append({'Scene': scene_name, 'Run': eps_id, 'Success': 0., 'SPL': 0, 'SoftSPL': 0}, 
						ignore_index=True)
					print(f'scene_name = {scene_name}, eps = {eps_id} failed')
		except:
			print('doesnt have this file')


print('=========================================================================================')

#=================================== write df to html ======================================
html = df.to_html()
  
# write html to file
html_f = open(f'{output_folder}/pandas_results/{result_folder}.html', "w")
html_f.write(f'<h5>All data</h5>')
html_f.write(html)

#==================================== clean up df ===========================
df2 = df.dropna()


#============================= compute data by scene ===============================
scene_grp = df2.groupby(['Scene'])
scene_success = scene_grp['Success'].mean()
scene_spl = scene_grp['SPL'].mean()
scene_softspl = scene_grp['SoftSPL'].mean()

scene_info = pd.concat([scene_success, scene_spl, scene_softspl], axis='columns', sort=False)

#================================ write df to html ==========================================
html = scene_info.to_html()
html_f.write(f'<h5>Description by each scene</h5>')
html_f.write(html)


df3 = df2[['Success', 'SPL', 'SoftSPL']].mean()
html = df3.to_frame('mean').to_html()
html_f.write(f'<h5>Mean over all episodes</h5>')
html_f.write(html)

html_f.close()

