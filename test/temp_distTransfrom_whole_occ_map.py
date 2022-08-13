import numpy as np 
import matplotlib.pyplot as plt
from modeling.utils.baseline_utils import read_occ_map_npy
from core import cfg

import skimage.measure

import modeling.utils.frontier_utils as fr_utils

from modeling.localNavigator_Astar import localNav_Astar
import networkx as nx

import scipy.ndimage
from math import sqrt

from skimage.morphology import skeletonize
import sknw
from modeling.utils.navigation_utils import change_brightness
import cv2

def my_tsp(G, source_node=0, weight="weight", cycle=True):
	#method = nx.algorithms.approximation.christofides
	method = nx.algorithms.approximation.greedy_tsp
	nodes = list(G.nodes)

	dist = {}
	path = {}
	for n, (d, p) in nx.all_pairs_dijkstra(G, weight=weight):
		dist[n] = d
		path[n] = p

	GG = nx.Graph()
	for u in nodes:
		for v in nodes:
			if u == v:
				continue
			GG.add_edge(u, v, weight=dist[u][v])
	best_GG = method(GG, weight, source=source_node)

	best_path = []
	for u, v in nx.utils.pairwise(best_GG):
		best_path.extend(path[u][v][:-1])
	best_path.append(v)
	return best_path


def skeletonize_frontier(component_occ_grid, skeleton):
	skeleton_component = np.where(component_occ_grid, skeleton, False)

	#'''
	cp_component_occ_grid = component_occ_grid.copy().astype('int16')
	cp_component_occ_grid[skeleton_component] = 3	
	plt.imshow(cp_component_occ_grid)
	
	plt.show()
	#'''

	cost_din = np.sum(skeleton_component)
	cost_dout = np.sum(skeleton_component)
	cost_dall = (cost_din + cost_dout)/2

	return cost_dall, cost_din, cost_dout, skeleton_component

def get_frontiers(occupancy_grid, gt_occupancy_grid, observed_area_flag, skeleton):
	filtered_grid = scipy.ndimage.maximum_filter(occupancy_grid == cfg.FE.UNOBSERVED_VAL, size=3)
	frontier_point_mask = np.logical_and(filtered_grid, occupancy_grid == cfg.FE.FREE_VAL)

	if cfg.FE.GROUP_INFLATION_RADIUS < 1:
		inflated_frontier_mask = frontier_point_mask
	else:
		inflated_frontier_mask = gridmap.utils.inflate_grid(frontier_point_mask,
			inflation_radius=cfg.FE.GROUP_INFLATION_RADIUS, obstacle_threshold=0.5,
			collision_val=1.0) > 0.5

	# Group the frontier points into connected components
	labels, nb = scipy.ndimage.label(inflated_frontier_mask)

	# Extract the frontiers
	frontiers = set()
	for ii in range(nb):
		raw_frontier_indices = np.where(np.logical_and(labels == (ii + 1), frontier_point_mask))
		frontiers.add(
			fr_utils.Frontier(
				np.concatenate((raw_frontier_indices[0][None, :],
								raw_frontier_indices[1][None, :]),
							   axis=0)))

	# Compute potential
	if cfg.NAVI.PERCEPTION == 'Potential':
		free_but_unobserved_flag = np.logical_and(gt_occupancy_grid == cfg.FE.FREE_VAL, observed_area_flag == False)
		free_but_unobserved_flag = scipy.ndimage.maximum_filter(free_but_unobserved_flag, size=3)

		labels, nb = scipy.ndimage.label(free_but_unobserved_flag)

		for ii in range(nb):
			component = (labels == (ii+1))
			for f in frontiers:
				if component[int(f.centroid[0]), int(f.centroid[1])]:
					f.R = np.sum(component)
					#f.D = round(sqrt(f.R), 2)
					#try:
					#cost_dall, cost_din, cost_dout, skeleton, skeleton_graph = skeletonize_map(component)#, display=True)
					cost_dall, cost_din, cost_dout, skeleton_component = skeletonize_frontier(component, skeleton)
					'''
					except:
						cost_dall = round(sqrt(f.R), 2)
						cost_din = cost_dall
						cost_dout = cost_dall
					'''
					print(f'cost_dall = {cost_dall}, cost_din = {cost_din}, cost_dout = {cost_dout}')

					if True:
						fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
						ax[0].imshow(occupancy_grid, cmap='gray')
						ax[0].scatter(f.points[1], f.points[0], c='yellow', zorder=2)
						ax[0].scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
						ax[0].get_xaxis().set_visible(False)
						ax[0].get_yaxis().set_visible(False)
						ax[0].set_title('explored occupancy map')

						ax[1].imshow(component, cmap='gray')
						ax[1].get_xaxis().set_visible(False)
						ax[1].get_yaxis().set_visible(False)
						ax[1].set_title(f'area potential, component {ii}')

						cp_component = component.copy().astype('int16')
						cp_component[skeleton_component] = 3	
						ax[2].imshow(cp_component)
						ax[2].get_xaxis().set_visible(False)
						ax[2].get_yaxis().set_visible(False)
						ax[2].set_title('skeleton')

						fig.tight_layout()
						#plt.title(f'component {ii}')
						plt.show()

	return frontiers

scene_name = 'yqstnuAEVhm_0'
occ_map_npy = np.load(f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/test/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)

cp_gt_occ_map = gt_occ_map.copy()
cp_gt_occ_map = cp_gt_occ_map.astype(np.uint8)
#=========================== get the skeleton image of the whole map ==========================
distT = cv2.distanceTransform(cp_gt_occ_map, distanceType=cv2.DIST_L2, maskSize=5)
plt.imshow(distT)
plt.show()

assert 1==2

LN = localNav_Astar(pose_range, coords_range, WH, scene_name)

gt_occ_map = np.where(gt_occ_map==1, cfg.FE.FREE_VAL, gt_occ_map) # free cell
gt_occ_map = np.where(gt_occ_map==0, cfg.FE.COLLISION_VAL, gt_occ_map) # occupied cell

occupancy_map = gt_occ_map.copy()

H, W = occupancy_map.shape
BLOCK = 30
observed_area_flag = np.zeros((H, W), dtype=bool)
observed_area_flag[BLOCK:H-BLOCK, BLOCK:W-BLOCK] = True

occupancy_map[~observed_area_flag] = cfg.FE.UNOBSERVED_VAL

#================================= visualize skeleton on the unobserved area ======================
color_occ_map = np.zeros((H, W, 3), dtype='uint8')
color_occ_map[cp_gt_occ_map == 1] = [255, 255, 255]
color_occ_map[cp_gt_occ_map == 0] = [120, 120, 130]
color_occ_map = change_brightness(color_occ_map, occupancy_map != cfg.FE.UNOBSERVED_VAL, value=60) 
plt.imshow(color_occ_map)
nodes = skeleton_G.nodes()
ps = np.array(nodes)
plt.plot(ps[:,1], ps[:,0], 'r.')
plt.show()
#assert 1==2


agent_map_pose = (31, 111)
frontiers = get_frontiers(occupancy_map, gt_occ_map, observed_area_flag, skeleton)

frontiers = LN.filter_unreachable_frontiers_temp(frontiers, agent_map_pose, occupancy_map)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 15))
ax.imshow(occupancy_map)
for f in frontiers:
	ax.scatter(f.points[1], f.points[0], c='white', zorder=2)
	ax.scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
ax.scatter(agent_map_pose[0], agent_map_pose[1], marker='s', s=50, c='red', zorder=5)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('improved observed_occ_map + frontiers')

fig.tight_layout()
plt.title('observed area')
plt.show()

