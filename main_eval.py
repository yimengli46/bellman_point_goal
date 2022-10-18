import numpy as np
from modeling.frontier_explore_DP import nav_DP
from modeling.frontier_explore_ANS import nav_ANS
from modeling.utils.baseline_utils import create_folder
import habitat
import habitat_sim
from modeling.utils.navigation_utils import SimpleRLEnv, get_scene_name
from core import cfg
import argparse
import torch

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config',
						type=str,
						required=False,
						default='exp_90degree_Optimistic_NAVMESH_MAP_1STEP_500STEPS.yaml')
	args = parser.parse_args()

	cfg.merge_from_file(f'configs/{args.config}')
	cfg.freeze()

	#=============================== basic setup =======================================
	split = 'test'
	if cfg.EVAL.SIZE == 'small':
		scene_floor_dict = np.load(
			f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy',
			allow_pickle=True).item()
	elif cfg.EVAL.SIZE == 'large':
		scene_floor_dict = np.load(
			f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/large_scale_{split}_scene_floor_dict.npy',
			allow_pickle=True).item()

	#for env_scene in cfg.MAIN.TEST_SCENE_NO_FLOOR_LIST:
	for env_scene in ['yqstnuAEVhm']:
	#for env_scene in ['5ZKStnWn8Zo']:

		#================================ load habitat env============================================
		config = habitat.get_config(config_paths=cfg.GENERAL.DATALOADER_CONFIG_PATH)
		config.defrost()
		config.SIMULATOR.SCENE = f'{cfg.GENERAL.HABITAT_SCENE_DATA_PATH}/mp3d/{env_scene}/{env_scene}.glb'
		config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
		config.freeze()
		env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)

		env.reset()
		scene_dict = scene_floor_dict[env_scene]

		device = torch.device('cuda:0')

		#=============================== traverse each floor ===========================
		for floor_id in list(scene_dict.keys()):
			height = scene_dict[floor_id]['y']
			scene_name = f'{env_scene}_{floor_id}'

			if scene_name in cfg.MAIN.TEST_SCENE_LIST:
				print(f'**********scene_name = {scene_name}***********')

				if cfg.EVAL.SIZE == 'small':
					output_folder = cfg.SAVE.TESTING_RESULTS_FOLDER
				elif cfg.EVAL.SIZE == 'large':
					output_folder = cfg.SAVE.LARGE_TESTING_RESULTS_FOLDER
				create_folder(output_folder)
				scene_output_folder = f'{output_folder}/{scene_name}'
				create_folder(scene_output_folder)

				testing_data = scene_dict[floor_id]['start_goal_pair']
				if not cfg.EVAL.USE_ALL_START_POINTS:
					if len(testing_data) > 3:
						testing_data = testing_data[:3]

				#'''
				results = {}
				for idx, data in enumerate(testing_data):
				#for idx in range(1, 2):
					data = testing_data[idx]
					print(
						f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA EPS {idx} BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB'
					)
					start_pose, goal_pose, start_goal_geodesic_distance = data
					print(f'start_pose = {start_pose}')
					saved_folder = f'{scene_output_folder}/eps_{idx}'
					create_folder(saved_folder, clean_up=False)

					steps = 0
					trajectory = []
					action_lst = []
					#try:
					if cfg.NAVI.STRATEGY == 'ANS':
						covered_area_percent, steps, trajectory, action_lst, step_cov_pairs = nav_ANS(split, env, idx, scene_name, height, start_pose, saved_folder, device)
					else:
						steps, trajectory, action_lst, nav_metrics = nav_DP(split, env, idx, scene_name, height, start_pose, goal_pose, start_goal_geodesic_distance, saved_folder, device)
					#except:
					#	print(f'CCCCCCCCCCCCCC failed EPS {idx} DDDDDDDDDDDDDDD')

					result = {}
					result['eps_id'] = idx
					result['steps'] = steps
					result['trajectory'] = trajectory
					result['actions'] = action_lst
					result['nav_metrics'] = nav_metrics
					results[idx] = result

				np.save(f'{output_folder}/results_{scene_name}.npy', results)
				#'''

		env.close()

if __name__ == "__main__":

	main()
