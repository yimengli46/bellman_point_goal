import numpy as np 
from core import cfg
import pandas as pd
from modeling.utils.baseline_utils import read_map_npy
import matplotlib.pyplot as plt
import glob
import os

#scene_list = cfg.MAIN.TEST_SCENE_LIST
output_folder = 'output' #cfg.SAVE.TESTING_RESULTS_FOLDER
result_folder = 'LARGE_TESTING_RESULTS_90degree_Optimistic_NAVMESH_MAP_1STEP_500STEPS'
npy_list = [os.path.splitext(os.path.basename(x))[0] for x in sorted(glob.glob(f'{output_folder}/{result_folder}/*.npy'))]

df = pd.DataFrame(columns=['Scene', 'Run', 'Success', 'SPL', 'SoftSPL'])
df['Success'] = df['Success'].astype(int)
df['SPL'] = df['SPL'].astype(float)
df['SoftSPL'] = df['SoftSPL'].astype(float)

for npy_name in npy_list:
	scene_name = npy_name[8:]
	results_npy = np.load(f'{output_folder}/{result_folder}/{npy_name}.npy', allow_pickle=True).item()
	num_test = len(results_npy.keys())

	percent_list = []
	step_list = []

	for i in range(num_test):
		result = results_npy[i]

		eps_id = result['eps_id']
		if 'success' in result['nav_metrics']:
			metrics_success = result['nav_metrics']['success']
			metrics_spl = result['nav_metrics']['spl']
			metrics_softspl = result['nav_metrics']['softspl']

			df = df.append({'Scene': scene_name, 'Run': eps_id, 'Success': metrics_success, 'SPL': metrics_spl, 'SoftSPL': metrics_softspl}, 
				ignore_index=True)

			print(f'scene_name = {scene_name}, eps = {eps_id}, Success = {metrics_success}, SPL = {metrics_spl}')
		else:
			df = df.append({'Scene': scene_name, 'Run': eps_id, 'Success': 0., 'SPL': 0, 'SoftSPL': 0}, 
				ignore_index=True)
			print(f'scene_name = {scene_name}, eps = {eps_id} failed')


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

