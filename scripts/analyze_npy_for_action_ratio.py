import numpy as np
from core import cfg

scene_list = cfg.MAIN.TEST_SCENE_LIST
output_folder = 'output'  # cfg.SAVE.TESTING_RESULTS_FOLDER
result_folder = 'TESTING_RESULTS_360degree_DP_NAVMESH_MAP_GT_Potential_D_Skeleton_Dall_1STEP_500STEPS_whole_Skeleton_graph'
#scene_list = ['yqstnuAEVhm_0']


thresh_percent = .01

total_action_list = []

for scene_name in scene_list:
    # for scene_name in scene_list:
    try:

        results_npy = np.load(
            f'{output_folder}/{result_folder}/results_{scene_name}.npy', allow_pickle=True).item()
        num_test = len(results_npy.keys())

        for i in range(num_test):
            result = results_npy[i]

            total_action_list += result['actions']

    except:
        print(f'failed to process scene {scene_name}.')

values, counts = np.unique(total_action_list, return_counts=True)
