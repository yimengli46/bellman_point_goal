import numpy as np
from modeling.frontier_explore_DP import nav_DP
from modeling.frontier_explore_ANS import nav_ANS
from modeling.utils.baseline_utils import create_folder
import habitat
import habitat_sim
from modeling.utils.navigation_utils import SimpleRLEnv, get_scene_name
from core import cfg
import argparse
import multiprocessing
import os
import torch

def build_env(env_scene, device_id=0):
	#================================ load habitat env============================================
	config = habitat.get_config(config_paths=cfg.GENERAL.DATALOADER_CONFIG_PATH)
	config.defrost()
	#config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_TEST_EPISODE_DATA_PATH
	config.SIMULATOR.SCENE = f'{cfg.GENERAL.HABITAT_SCENE_DATA_PATH}/mp3d/{env_scene}/{env_scene}.glb'
	config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
	config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = device_id
	config.freeze()
	env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
	return env

def nav_test(env_scene, output_folder, scene_floor_dict):
	#============================ get a gpu
	device_id = gpu_Q.get()

	#================ initialize habitat env =================
	env = build_env(env_scene, device_id=device_id)
	env.reset()

	scene_dict = scene_floor_dict[env_scene]
	for floor_id in list(scene_dict.keys()):
		split = 'test'
		height = scene_dict[floor_id]['y']
		scene_name = f'{env_scene}_{floor_id}'

		scene_output_folder = f'{output_folder}/{scene_name}'
		create_folder(scene_output_folder)

		print(f'**********scene_name = {scene_name}, device_id ={device_id}***********')
		scene_output_folder = f'{output_folder}/{scene_name}'
		create_folder(scene_output_folder)

		device = torch.device(f'cuda:{device_id}')

		#================ load testing data ==================
		try:
			testing_data = scene_dict[floor_id]['start_goal_pair']
		except:
			testing_data = []
		if not cfg.EVAL.USE_ALL_START_POINTS:
			if len(testing_data) > 3:
				testing_data = testing_data[:3]

		results = {}
		for idx, data in enumerate(testing_data):
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
			
			#'''
			try:
				if cfg.NAVI.STRATEGY == 'ANS':
					covered_area_percent, steps, trajectory, action_lst, step_cov_pairs = nav_ANS(split, env, idx, scene_name, height, start_pose, saved_folder, device)
				else:
					steps, trajectory, action_lst, nav_metrics = nav_DP(split, env, idx, scene_name, height, start_pose, goal_pose, start_goal_geodesic_distance, saved_folder, device)
			except:
				print(f'CCCCCCCCCCCCCC failed {scene_name} EPS {idx} DDDDDDDDDDDDDDD')

			result = {}
			result['eps_id'] = idx
			result['steps'] = steps
			result['trajectory'] = trajectory
			result['actions'] = action_lst
			result['nav_metrics'] = nav_metrics
			results[idx] = result
			#'''

		np.save(f'{output_folder}/results_{scene_name}.npy', results)

	env.close()

	#================================ release the gpu============================
	gpu_Q.put(device_id)

def multi_run_wrapper(args):
	""" wrapper for multiprocessor """
	nav_test(args[0], args[1], args[2])

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config',
						type=str,
						required=False,
						default='exp_90degree_Optimistic_NAVMESH_MAP_1STEP_500STEPS.yaml')
	parser.add_argument('--j',
						type=int,
						required=False,
						default=1)
	args = parser.parse_args()

	cfg.merge_from_file(f'configs/{args.config}')
	cfg.freeze()

	#====================== get the available GPU devices ============================
	visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
	devices = [int(dev) for dev in visible_devices]

	for device_id in devices:
		for _ in range(args.j):
			gpu_Q.put(device_id)

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

	if cfg.EVAL.SIZE == 'small':
		output_folder = cfg.SAVE.TESTING_RESULTS_FOLDER
	elif cfg.EVAL.SIZE == 'large':
		output_folder = cfg.SAVE.LARGE_TESTING_RESULTS_FOLDER
	create_folder(output_folder)

	args0 = cfg.MAIN.TEST_SCENE_NO_FLOOR_LIST
	with multiprocessing.Pool(processes=len(args0), maxtasksperchild=1) as pool:
		args1 = [output_folder for _ in range(len(args0))]
		args2 = [scene_floor_dict for _ in range(len(args0))]
		pool.map(multi_run_wrapper, list(zip(args0, args1, args2)))
		pool.close()
		pool.join()

if __name__ == "__main__":
	gpu_Q = multiprocessing.Queue()
	main()
