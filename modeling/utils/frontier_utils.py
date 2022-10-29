import numpy as np
import matplotlib.pyplot as plt
from core import cfg
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from core import cfg
import scipy.ndimage
from .baseline_utils import pose_to_coords, apply_color_to_map, crop_map, spatial_transform_map, pxl_coords_to_pose
from math import sqrt
from operator import itemgetter
import torch
import cv2
from skimage.morphology import skeletonize
import sknw
import networkx as nx
from skimage.graph import MCP_Geometric as MCPG
from skimage.graph import route_through_array
import torch.nn.functional as F
import math
import lsp_accel
import itertools

def skeletonize_map(occupancy_grid):
	skeleton = skeletonize(occupancy_grid)
	
	graph = sknw.build_sknw(skeleton)

	tsp = nx.algorithms.approximation.traveling_salesman_problem
	path = tsp(graph)

	nodes = list(graph.nodes)
	for i in range(len(path)):
		if not nodes:
			index = i
			break
		if path[i] in nodes:
			nodes.remove(path[i])
	
	d_in = path[:index]
	d_out = path[index-1:]

	cost_din = 0
	for i in range(len(d_in)-1):
		cost_din += graph[d_in[i]][d_in[i+1]]['weight']

	cost_dout = 0
	for i in range(len(d_out)-1):
		cost_dout += graph[d_out[i]][d_out[i+1]]['weight']
	
	return cost_din+cost_dout, cost_din, cost_dout, skeleton, graph

def skeletonize_frontier_graph(component_occ_grid, skeleton):
	component_skeleton = np.where(component_occ_grid, skeleton, False)

	if np.sum(component_skeleton) > 0:
		component_G = sknw.build_sknw(component_skeleton)

		#================= computed connected components =============================
		list_ccs = [component_G.subgraph(c).copy() for c in nx.connected_components(component_G)] 
		#print(f'len(list_ccs) = {len(list_ccs)}')

		'''	
		plt.imshow(component_occ_grid, cmap='gray')
		for sub_G in set_ccs:
			nodes = sub_G.nodes()
			ps = np.array(nodes)
			plt.plot(ps[:,1], ps[:,0], c=np.random.rand(3,))
		plt.show()
		'''

		#====================== compute the cost of each component and then add them up
		arr_cost_dall = np.zeros(len(list_ccs))
		arr_cost_din = np.zeros(len(list_ccs))
		arr_cost_dout = np.zeros(len(list_ccs))
		for idx, sub_G in enumerate(list_ccs):
			#print(f'sub_G has {len(sub_G.nodes)} nodes.')
			if len(sub_G.nodes) > 1: # sub_G has more than one nodes
				path = my_tsp(sub_G)
				#=================== split path into d_in and d_out
				nodes = list(sub_G.nodes)
				for i in range(len(path)):
					if not nodes:
						index = i
						break
					if path[i] in nodes:
						nodes.remove(path[i])
				#================== compute cost_din and cost_dout
				d_in = path[:index]
				d_out = path[index-1:]
				cost_din = 0
				for i in range(len(d_in)-1):
					cost_din += sub_G[d_in[i]][d_in[i+1]]['weight']
				cost_dout = 0
				for i in range(len(d_out)-1):
					cost_dout += sub_G[d_out[i]][d_out[i+1]]['weight']
				cost_dall = cost_din + cost_dout
			else:
				cost_din = 1
				cost_dout = 1
				cost_dall = cost_din + cost_dout
			
			arr_cost_dall[idx] = cost_dall
			arr_cost_din[idx] = cost_din
			arr_cost_dout[idx] = cost_dout

		cost_dall = np.sum(arr_cost_dall)
		cost_din = np.sum(arr_cost_din)
		cost_dout = np.sum(arr_cost_dout)
	else:
		cost_din = 1
		cost_dout = 1
		cost_dall = cost_din + cost_dout
		component_G = nx.Graph()

	return cost_dall, cost_din, cost_dout, component_G

def skeletonize_frontier(component_occ_grid, skeleton):
	skeleton_component = np.where(component_occ_grid, skeleton, False)

	'''
	cp_component_occ_grid = component_occ_grid.copy().astype('int16')
	cp_component_occ_grid[skeleton_component] = 3	
	plt.imshow(cp_component_occ_grid)
	
	plt.show()
	'''

	cost_din = max(np.sum(skeleton_component), 1)
	cost_dout = max(np.sum(skeleton_component), 1)
	cost_dall = (cost_din + cost_dout)

	return cost_dall, cost_din, cost_dout, skeleton_component

def create_dense_graph(skeleton, flag_eight_neighs=True):
	H, W = skeleton.shape
	G = nx.grid_2d_graph(H, W)

	if flag_eight_neighs:
		for edge in G.edges:
			G.edges[edge]['weight'] = 1
		G.add_edges_from([((x, y), (x + 1, y + 1)) for x in range(0, H - 1)
						  for y in range(0, W - 1)] + [((x + 1, y), (x, y + 1))
													   for x in range(0, H - 1)
													   for y in range(0, W - 1)], weight=1.4)
	# remove those nodes where map is occupied
	mask_occupied_node = (skeleton.ravel() == False)
	nodes_npy = np.array(sorted(G.nodes))
	nodes_occupied = nodes_npy[mask_occupied_node]
	lst_nodes_occupied = list(map(tuple, nodes_occupied))
	G.remove_nodes_from(lst_nodes_occupied)

	return G

def my_tsp(G, weight="weight"):
	method = nx.algorithms.approximation.christofides
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
	best_GG = method(GG, weight)

	best_path = []
	for u, v in nx.utils.pairwise(best_GG):
		best_path.extend(path[u][v][:-1])
	best_path.append(v)
	return best_path

def prune_skeleton_graph(skeleton_G):
	dict_node_numEdges = {}
	for edge in skeleton_G.edges():
		u, v = edge
		for node in [u, v]:
			if node in dict_node_numEdges:
				dict_node_numEdges[node] += 1
			else:
				dict_node_numEdges[node] = 1
	to_prune_nodes = []
	for k, v in dict_node_numEdges.items():
		if v < 2:
			to_prune_nodes.append(k)
	skeleton_G_pruned = skeleton_G.copy()
	skeleton_G_pruned.remove_nodes_from(to_prune_nodes)
	return skeleton_G_pruned

def skeleton_G_to_skeleton(occ_grid, skeleton_G):
	skeleton = np.zeros(occ_grid.shape, dtype=bool)
	for edge in skeleton_G.edges():
		pts = np.array(skeleton_G.edges[edge]['pts'])
		skeleton[pts[:, 0], pts[:, 1]] = True 
	return skeleton

def prune_skeleton(occ_grid, skeleton):
	skeleton_G = sknw.build_sknw(skeleton)
	pruned_skeleton_G = prune_skeleton_graph(skeleton_G)
	skeleton = skeleton_G_to_skeleton(occ_grid, pruned_skeleton_G)
	return skeleton

class Frontier(object):

	def __init__(self, points):
		"""Initialized with a 2xN numpy array of points (the grid cell
		coordinates of all points on frontier boundary)."""
		inds = np.lexsort((points[0, :], points[1, :]))
		sorted_points = points[:, inds]
		
		self.is_from_last_chosen = False

		# Any duplicate points should be eliminated (would interfere with
		# equality checking).
		dupes = []
		for ii in range(1, sorted_points.shape[1]):
			if (sorted_points[:, ii - 1] == sorted_points[:, ii]).all():
				dupes += [ii]
		self.points = np.delete(sorted_points, dupes, axis=1)

		# Compute and cache the hash
		self.hash = hash(self.points.tobytes())

		self.P_S = 1.
		self.R_S = 0.
		self.R_E = 0.

	@property
	def centroid(self):
		#return self.get_centroid()
		return self.get_frontier_point()

	#'''
	def get_centroid(self):
		"""Returns the point that is the centroid of the frontier"""
		centroid = np.mean(self.points, axis=1)
		return centroid

	#'''
	'''
	def get_centroid(self):
		#print(f'points.shape = {self.points.shape}')
		points = self.points.transpose()
		distMatrix = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1)
		centroid_idx = np.argmin(distMatrix.sum(axis=0))
		centroid = self.points[:, centroid_idx]
		return centroid
	'''

	def get_frontier_point(self):
		"""Returns the point that is on the frontier that is closest to the
		actual centroid"""
		center_point = np.mean(self.points, axis=1)
		norm = np.linalg.norm(self.points - center_point[:, None], axis=0)
		ind = np.argmin(norm)
		return self.points[:, ind]

	def get_distance_to_point(self, point):
		norm = np.linalg.norm(self.points - point[:, None], axis=0)
		return norm.min()

	def __hash__(self):
		return self.hash

	def __eq__(self, other):
		return hash(self) == hash(other)


def mask_grid_with_frontiers(occupancy_grid, frontiers, do_not_mask=None):
	"""Mask grid cells in the provided occupancy_grid with the frontier points
	contained with the set of 'frontiers'. If 'do_not_mask' is provided, and
	set to either a single frontier or a set of frontiers, those frontiers are
	not masked."""

	if do_not_mask is not None:
		# Ensure that 'do_not_mask' is a set
		if isinstance(do_not_mask, Frontier):
			do_not_mask = set([do_not_mask])
		elif not isinstance(do_not_mask, set):
			raise TypeError("do_not_mask must be either a set or a Frontier")
		masking_frontiers = frontiers - do_not_mask
	else:
		masking_frontiers = frontiers

	masked_grid = occupancy_grid.copy()
	for frontier in masking_frontiers:
		masked_grid[frontier.points[0, :], frontier.points[1, :]] = 2

	return masked_grid


def get_frontiers(occupancy_grid):
	""" detect frontiers from occupancy_grid. 
	"""

	filtered_grid = scipy.ndimage.maximum_filter(
		occupancy_grid == cfg.FE.UNOBSERVED_VAL, size=3)
	frontier_point_mask = np.logical_and(filtered_grid,
										 occupancy_grid == cfg.FE.FREE_VAL)

	if cfg.FE.GROUP_INFLATION_RADIUS < 1:
		inflated_frontier_mask = frontier_point_mask
	else:
		inflated_frontier_mask = gridmap.utils.inflate_grid(
			frontier_point_mask,
			inflation_radius=cfg.FE.GROUP_INFLATION_RADIUS,
			obstacle_threshold=0.5,
			collision_val=1.0) > 0.5

	# Group the frontier points into connected components
	labels, nb = scipy.ndimage.label(inflated_frontier_mask, structure=np.ones((3,3)))

	# Extract the frontiers
	frontiers = set()
	for ii in range(nb):
		raw_frontier_indices = np.where(
			np.logical_and(labels == (ii + 1), frontier_point_mask))
		frontiers.add(
			Frontier(
				np.concatenate((raw_frontier_indices[0][None, :],
								raw_frontier_indices[1][None, :]),
							   axis=0)))

	return frontiers

def _eucl_dist(p1, p2):
    """Helper to compute Euclidean distance."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def _get_nearest_feasible_frontier(frontier, reference_frontier_set):
    """Returns the nearest 'feasible' frontier from a reference set."""
    f_gen = [(of, _eucl_dist(of.get_centroid(), frontier.get_centroid()))
             for of in reference_frontier_set]
    if len(f_gen) == 0:
        return None, 1e10
    else:
        return min(f_gen, key=lambda fd: fd[1])

def update_frontier_set(old_set, new_set, max_dist=6, chosen_frontier=None):
	for frontier in old_set:
		frontier.is_from_last_chosen = False 

	# shallow copy of the set
	old_set = old_set.copy()

	# These are the frontiers that will not appear in the new set
	outgoing_frontier_set = old_set - new_set
	# These will appear in the new set
	added_frontier_set = new_set - old_set

	if max_dist is not None:
		# loop through the newly added frontier set and set properties based upon the outgoing frontier set
		for af in added_frontier_set:
			nearest_frontier, nearest_frontier_dist = _get_nearest_feasible_frontier(af, outgoing_frontier_set)
			#print(f'nearest_frontier_dist = {nearest_frontier_dist}')
			if nearest_frontier_dist < max_dist:
				'''
				# this frontier R and D is not computed correctly
				if af.R < 1.1 and af.D < 1.1:
					af.R = nearest_frontier.R
					af.D = nearest_frontier.D
					af.Din = nearest_frontier.Din 
					af.Dout = nearest_frontier.Dout
				'''

				if nearest_frontier == chosen_frontier:
					af.is_from_last_chosen = True 

	if len(added_frontier_set) == 0:
		print(f'*** corner case, no new frontier.')
		chosen_frontier.is_from_last_chosen = True

	# Remove frontier_set that don't appear in the new set
	old_set.difference_update(outgoing_frontier_set)

	# Add the new frontier_set
	old_set.update(added_frontier_set)

	return old_set

def inter_local_map_global_map(local_map, global_map, robot_center):
	H_local, W_local = local_map.shape
	H_global, W_global = global_map.shape

	left_corner_local = np.array((0, 0))
	right_corner_local = np.array((W_local-1, H_local-1))
	left_corner_global = np.array((0, 0))
	right_corner_global = np.array((W_global-1, H_global-1))

	# move local map whose center is now at robot center
	robot_center = np.array(robot_center)
	local_map_center = np.array((W_local//2, H_local//2))
	trans = robot_center - local_map_center
	left_corner_local += trans
	right_corner_local += trans

	#print(f'local: {left_corner_local}, {right_corner_local}')

	# find intersection
	x0_global = max(left_corner_local[0], left_corner_global[0])
	x1_global = min(right_corner_local[0], right_corner_global[0])
	y0_global = max(left_corner_local[1], left_corner_global[1])
	y1_global = min(right_corner_local[1], right_corner_global[1])

	# move bbox back to local map coords
	x0_local, y0_local = np.array((x0_global, y0_global)) - trans
	x1_local, y1_local = np.array((x1_global, y1_global)) - trans

	return np.array((x0_local, y0_local, x1_local, y1_local)), np.array((x0_global, y0_global, x1_global, y1_global))


def compute_frontier_potential(frontiers, point_goal_coord, dist_occupancy_map, occupancy_grid, gt_occupancy_grid, observed_area_flag, sem_map, skeleton=None, unet_model=None, device=None, LN=None, agent_map_pose=None):
	# When the perception info is 'Potential', we use gt_occupancy_grid to compute the area of the component.

	# Compute potential
	if cfg.NAVI.PERCEPTION == 'Potential':
		free_but_unobserved_flag = np.logical_and(
			gt_occupancy_grid == cfg.FE.FREE_VAL, observed_area_flag == False)
		free_but_unobserved_flag = scipy.ndimage.maximum_filter(
			free_but_unobserved_flag, size=3)

		labels, nb = scipy.ndimage.label(free_but_unobserved_flag)

		for ii in range(nb):
			component = (labels == (ii + 1))
			for fron in frontiers:
				if component[int(fron.centroid[0]), int(fron.centroid[1])]:
					# decide if the point goal is reachable
					if component[point_goal_coord[1], point_goal_coord[0]]:
						# point goal is reachable from this frontier
						fron.P_S = 1.
						_, L = route_through_array(dist_occupancy_map, (point_goal_coord[1], point_goal_coord[0]), 
							(int(fron.centroid[0]), int(fron.centroid[1])))
						fron.R_S = L
						fron.R_E = 0.
					else:
						fron.P_S = 0.
						cost_dall, _, _, component_G = skeletonize_frontier_graph(component, skeleton)
						fron.R_E = cost_dall
						fron.R_S = 0.
					#print(f'P_S = {fron.P_S}, R_S = {fron.R_S}, R_E = {fron.R_E}')

					if cfg.NAVI.FLAG_VISUALIZE_FRONTIER_POTENTIAL:
						fig, ax = plt.subplots(nrows=1,
											   ncols=2,
											   figsize=(12, 5))
						ax[0].imshow(occupancy_grid, cmap='gray')
						ax[0].scatter(fron.points[1],
									  fron.points[0],
									  c='yellow',
									  zorder=2)
						ax[0].scatter(fron.centroid[1],
									  fron.centroid[0],
									  c='red',
									  zorder=2)
						ax[0].get_xaxis().set_visible(False)
						ax[0].get_yaxis().set_visible(False)
						ax[0].set_title('explored occupancy map')

						ax[1].imshow(component)
						ax[1].scatter(point_goal_coord[0], point_goal_coord[1], marker='*', s=50, c='cyan', zorder=5)
						ax[1].get_xaxis().set_visible(False)
						ax[1].get_yaxis().set_visible(False)
						ax[1].set_title('area potential')

						fig.tight_layout()
						plt.title(f'component {ii}')
						plt.show()

	elif cfg.NAVI.PERCEPTION == 'UNet_Potential':
		agent_coord = LN.get_agent_coords(agent_map_pose)
		#============================================ prepare input data ====================================
		sem_map = np.where(sem_map >= cfg.SEM_MAP.GRID_CLASS_SIZE, 0, sem_map)

		M_p = np.stack((occupancy_grid, sem_map), axis=0)
		tensor_M_p = torch.tensor(M_p).float().unsqueeze(0)
		#print(f'tensor_M_p.shape = {tensor_M_p.shape}')

		#================== crop out the map centered at the agent ==========================
		_, H, W = M_p.shape
		Wby2, Hby2 = W // 2, H // 2
		tform_trans = torch.Tensor([[agent_coord[0] - Wby2, agent_coord[1] - Hby2, 0]])
		crop_center = torch.Tensor([[W / 2.0, H / 2.0]]) + tform_trans[:, :2]
		# Crop out the appropriate size of the map
		map_size = int(cfg.PRED.PARTIAL_MAP.OUTPUT_MAP_SIZE / cfg.SEM_MAP.CELL_SIZE)
		tensor_M_p = crop_map(tensor_M_p, crop_center, map_size, 'nearest')

		tensor_Mp = tensor_M_p.long().squeeze(0)

		#==================== convert into one hot vector ==================================
		tensor_Mp_occ = tensor_Mp[0] # H x W
		tensor_Mp_occ = F.one_hot(tensor_Mp_occ, num_classes=3).permute(2, 0, 1) # 3 x H x W
		tensor_Mp_sem = tensor_Mp[1]
		tensor_Mp_sem = F.one_hot(tensor_Mp_sem, num_classes=cfg.SEM_MAP.GRID_CLASS_SIZE).permute(2, 0, 1) # num_classes x H x W
		tensor_Mp = torch.cat((tensor_Mp_occ, tensor_Mp_sem), 0).float()

		if cfg.PRED.PARTIAL_MAP.INPUT == 'occ_only':
			tensor_Mp = tensor_Mp[0:3]

		tensor_Mp = tensor_Mp.unsqueeze(0).to(device) # for batch
		
		with torch.no_grad():
			outputs = unet_model(tensor_Mp)
			output = outputs.cpu().numpy()[0].transpose((1, 2, 0))

		bbox_local, bbox_global = inter_local_map_global_map(output[:, :, 0], M_p[0], agent_coord)
		results = np.zeros((H, W, 4))
		results[bbox_global[1]:bbox_global[3]+1, bbox_global[0]:bbox_global[2]+1] = output[bbox_local[1]:bbox_local[3]+1, bbox_local[0]:bbox_local[2]+1]	
		output = results

		#=========================== reshape output and mask out non zero points =============================== 
		H, W = occupancy_grid.shape
		#output = cv2.resize(output, (W, H), interpolation=cv2.INTER_NEAREST)

		for f in frontiers:
			points = f.points.transpose()
			points_vals = output[points[:, 0], points[:, 1]] # N, 4
			#print(f'points_vals.shape = {points_vals.shape}')
			mask_points = (points_vals[:, 0] > 0) # N
			#print(f'mask_points.shape = {mask_points.shape}')
			if mask_points.shape[0] > 0:
				U_a = max(np.mean(points_vals[mask_points, 0]) * cfg.PRED.PARTIAL_MAP.DIVIDE_AREA, 1.)
				U_dall = max(np.mean(points_vals[mask_points, 1]) * cfg.PRED.PARTIAL_MAP.DIVIDE_D, 1.)
				U_din = max(np.mean(points_vals[mask_points, 2]) * cfg.PRED.PARTIAL_MAP.DIVIDE_D, 1.)
				U_dout = max(np.mean(points_vals[mask_points, 3]) * cfg.PRED.PARTIAL_MAP.DIVIDE_D, 1.)
			else:
				U_a, U_dall, U_din, U_dout = 1.0, 1.0, 1.0, 1.0

			if cfg.NAVI.D_type == 'Sqrt_R':
				f.R = U_a
				f.D = round(sqrt(f.R), 2)
				f.Din = f.D
				f.Dout = f.D
			elif cfg.NAVI.D_type == 'Skeleton':
				f.R = U_a
				f.D = U_dall
				f.Din = U_din
				f.Dout = U_dout

		if cfg.NAVI.FLAG_VISUALIZE_FRONTIER_POTENTIAL:
			fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
			ax[0][0].imshow(occupancy_grid, cmap='gray')
			ax[0][0].get_xaxis().set_visible(False)
			ax[0][0].get_yaxis().set_visible(False)
			ax[0][0].set_title('input: occupancy_map_Mp')
			color_sem_map = apply_color_to_map(sem_map)
			ax[0][1].imshow(color_sem_map)
			ax[0][1].get_xaxis().set_visible(False)
			ax[0][1].get_yaxis().set_visible(False)
			ax[0][1].set_title('input: semantic_map_Mp')
			ax[1][0].imshow(output[:, :, 0])
			ax[1][0].get_xaxis().set_visible(False)
			ax[1][0].get_yaxis().set_visible(False)
			ax[1][0].set_title('output: U_a')
			ax[1][1].imshow(output[:, :, 1])
			ax[1][1].get_xaxis().set_visible(False)
			ax[1][1].get_yaxis().set_visible(False)
			ax[1][1].set_title('output: U_dall')
			fig.tight_layout()
			plt.show()

	return frontiers


def remove_isolated_points(occupancy_grid, threshold=2):
	""" remove isolated points to clean up the occupancy_grid"""
	H, W = occupancy_grid.shape
	new_grid = occupancy_grid.copy()
	for i in range(1, H - 1):
		for j in range(1, W - 1):
			if occupancy_grid[i][j] == cfg.FE.UNOBSERVED_VAL:
				new_grid[i][j] = nearest_value_og(occupancy_grid,
												  i,
												  j,
												  threshold=threshold)
	return new_grid


def nearest_value_og(occupancy_grid, i, j, threshold=4):
	d = {cfg.FE.COLLISION_VAL: 0, cfg.FE.FREE_VAL: 0, cfg.FE.UNOBSERVED_VAL: 0}
	d[occupancy_grid[i - 1][j]] += 1
	d[occupancy_grid[i + 1][j]] += 1
	d[occupancy_grid[i][j - 1]] += 1
	d[occupancy_grid[i][j + 1]] += 1

	for occupancy_value, count in d.items():
		if count >= threshold:
			return occupancy_value
	return occupancy_grid[i][j]


def get_frontier_nearest_to_goal(agent_pose, frontiers, goal_coord, LN, occupancy_map):
	""" select frontier nearest to the point goal
	used for the 'Optmistic' strategy.
	"""
	agent_coord = LN.get_agent_coords(agent_pose)

	binary_occupancy_map = occupancy_map.copy()
	#plt.imshow(binary_occupancy_map)
	#plt.show()
	binary_occupancy_map[binary_occupancy_map == cfg.FE.UNOBSERVED_VAL] = cfg.FE.FREE_VAL
	binary_occupancy_map[binary_occupancy_map == cfg.FE.COLLISION_VAL] = 0
	binary_occupancy_map[binary_occupancy_map != 0] = 1
	binary_occupancy_map[binary_occupancy_map == 0] = 1000

	#plt.imshow(binary_occupancy_map)
	#plt.show()

	min_dist = 1e10
	min_fron = None
	for fron in frontiers:
		# compute dist from the agent to the frontier
		_, L_agent2fron = route_through_array(binary_occupancy_map, (agent_coord[1], agent_coord[0]), 
			(int(fron.centroid[0]), int(fron.centroid[1])))
		# compute dist from the frontier to the point goal
		_, L_fron2goal = route_through_array(binary_occupancy_map, (int(fron.centroid[0]), int(fron.centroid[1])), 
			(goal_coord[1], goal_coord[0]))

		dist = L_agent2fron + L_fron2goal 

		if dist < min_dist:
			min_dist = dist 
			min_fron = fron 
		elif dist == min_dist and hash(fron) > hash(min_fron):
			min_dist = dist 
			min_fron = fron
	return min_fron

def get_the_nearest_frontier(frontiers, agent_pose, dist_occupancy_map, LN):
	""" select nearest frontier to the robot.
	used for the 'FME' strategy.
	"""
	agent_coord = LN.get_agent_coords(agent_pose)
	min_L = 10000000
	min_frontier = None

	for fron in frontiers:
		_, L = route_through_array(dist_occupancy_map, (agent_coord[1], agent_coord[0]), 
			(int(fron.centroid[0]), int(fron.centroid[1])))

		if L < min_L:
			min_L = L
			min_frontier = fron 
		elif L == min_L and hash(fron) > hash(min_frontier):
			min_L = L
			min_frontier = fron
	return min_frontier

def get_the_nearest_frontier_to_the_long_term_goal(frontiers, long_term_goal_coords):
	""" select nearest frontier to the robot.
	used for the 'FME' strategy.
	"""
	min_L = 10000000
	min_frontier = None

	for fron in frontiers:
		fron_centroid_coords = (int(fron.centroid[1]),
								int(fron.centroid[0]))
		L = _eucl_dist(fron_centroid_coords, long_term_goal_coords)

		if L < min_L:
			min_L = L
			min_frontier = fron 
		elif L == min_L and hash(fron) > hash(min_frontier):
			min_L = L
			min_frontier = fron
	return min_frontier


def count_free_space_at_frontiers(frontiers, gt_occupancy_grid, area=10):
	""" compute the free space in the neighborhoadd of the frontier center.
	"""
	H, W = gt_occupancy_grid.shape
	for fron in frontiers:
		centroid = (int(fron.centroid[1]), int(fron.centroid[0]))
		x1 = max(0, centroid[0] - area)
		x2 = min(W, centroid[0] + area)
		y1 = max(0, centroid[1] - area)
		y2 = min(H, centroid[1] + area)
		fron_neigh = gt_occupancy_grid[y1:y2, x1:x2]
		#print(f'centroid[0] = {centroid[0]}, y1 = {y1}, y2= {y2}, x1 = {x1}, x2 = {x2}')
		#plt.imshow(fron_neigh)
		#plt.show()
		fron.area_neigh = np.sum(fron_neigh == cfg.FE.FREE_VAL)
		#print(f'fron.area_neigh = {fron.area_neigh}')


def get_frontier_with_DP(frontiers, agent_pose, dist_occupancy_map, LN):
	""" select the frontier from frontiers with the Bellman Equation.

	from agent_pose and the observed_occupancy_map, compute D and L.
	"""
	min_Q = 1e10
	min_frontier = None
	agent_coord = LN.get_agent_coords(agent_pose)

	for fron in frontiers:
		#print('-------------------------------------------------------------')
		visited_frontiers = set()
		Q = compute_Q(agent_coord, fron, frontiers, visited_frontiers, dist_occupancy_map)
		print(f'Q = {Q}, R_S = {fron.R_S}')
		if Q < min_Q:
			min_Q = Q
			min_frontier = fron

	return min_frontier


def compute_Q(agent_coord, target_frontier, frontiers, visited_frontiers, dist_occupancy_map):
	""" compute the Q values of the frontier 'target_frontier'"""
	#print(f'agent_coord = {agent_coord}, target_frontier = {target_frontier.centroid}, steps = {steps}')
	Q = 0
	#L = LN.compute_L(G, agent_coord, target_frontier)
	_, D = route_through_array(dist_occupancy_map, (agent_coord[1], agent_coord[0]), 
		(int(target_frontier.centroid[0]), int(target_frontier.centroid[1])))
	# move forward 5 cells. every move forward is combined with 2 turnings.

	Q += (D / 5. * cfg.NAVI.STEP_RATIO) + (target_frontier.P_S * (target_frontier.R_S / 5. * cfg.NAVI.STEP_RATIO))

	visited_frontiers.add(target_frontier)
	rest_frontiers = frontiers - visited_frontiers

	min_next_Q = 1e10
	for fron in rest_frontiers:
		fron_centroid_coords = (int(target_frontier.centroid[1]),
								int(target_frontier.centroid[0]))
		next_Q = compute_Q(fron_centroid_coords, fron, frontiers,
						   visited_frontiers.copy(), dist_occupancy_map)
		if next_Q < min_next_Q:
			min_next_Q = next_Q
	#print(f'min_next_Q = {min_next_Q}')
	Q += (1 - target_frontier.P_S) * (target_frontier.R_E / 5. * cfg.NAVI.STEP_RATIO + min_next_Q)

	#print(f'Q = {Q}')
	return Q


def select_top_frontiers(frontiers, top_n=5):
	""" select a few frontiers with the largest value.

	The objective is to reduce the number of frontiers when using the 'DP' strategy.
	top_n decides the number of frontiers to keep.
	"""
	if len(frontiers) <= top_n:
		return frontiers

	lst_frontiers = []
	for fron in frontiers:
		lst_frontiers.append((fron, fron.R_E))

	res = sorted(lst_frontiers, key=itemgetter(1), reverse=False)[:top_n]

	new_frontiers = set()
	for fron, _ in res:
		new_frontiers.add(fron)

	return new_frontiers

def get_frontier_with_DP_accel(frontiers, agent_pose, dist_occupancy_map, goal_coord, LN):
	""" select the frontier from frontiers with the Bellman Equation.

	from agent_pose and the observed_occupancy_map, compute D and L.
	"""
	agent_coord = LN.get_agent_coords(agent_pose)

	frontiers = list(frontiers)

	#============ create distances
	goal_distances = {}
	for fron in frontiers:
		# compute dist from the frontier to the point goal
		_, L_fron2goal = route_through_array(dist_occupancy_map, (int(fron.centroid[0]), int(fron.centroid[1])), 
			(goal_coord[1], goal_coord[0]))
		goal_distances[fron] = L_fron2goal# / 5. * cfg.NAVI.STEP_RATIO

	robot_distances = {}
	for fron in frontiers:
		# compute dist from the robot to the frontier
		_, L_robot2fron = route_through_array(dist_occupancy_map, (agent_coord[1], agent_coord[0]), 
			(int(fron.centroid[0]), int(fron.centroid[1])))
		robot_distances[fron] = L_robot2fron# / 5. * cfg.NAVI.STEP_RATIO

	frontier_distances = {
		frozenset(pair): route_through_array(dist_occupancy_map, (int(pair[0].centroid[0]), int(pair[0].centroid[1])), 
		 (int(pair[1].centroid[0]), int(pair[1].centroid[1])))[1]# / 5. * cfg.NAVI.STEP_RATIO
		for pair in itertools.combinations(frontiers, 2)
	}

	distances = {
		'goal': goal_distances,
		'robot': robot_distances,
		'frontier': frontier_distances
	}

	cpp_cost, cpp_ordering = get_lowest_cost_ordering(frontiers, distances)
	#print(f'len(cpp_ordering) = {len(cpp_ordering)}')
	return cpp_ordering[0]


def get_lowest_cost_ordering(subgoals, distances):
	if len(subgoals) == 0:
		return None, None

	h = {
		s: distances['goal'][s] + distances['robot'][s] +
		s.P_S * s.R_S +
		(1 - s.P_S) * s.R_E
		for s in subgoals
	}
	subgoals.sort(reverse=False, key=lambda s: h[s])
	s_dict = {hash(s): s for s in subgoals}
	rd_cpp = {hash(s): distances['robot'][s] for s in subgoals}
	gd_cpp = {hash(s): distances['goal'][s] for s in subgoals}
	fd_cpp = {(hash(sp[0]), hash(sp[1])): distances['frontier'][frozenset(sp)]
			  for sp in itertools.permutations(subgoals, 2)}
	s_cpp = [
		lsp_accel.FrontierData(s.P_S, s.R_S,
							   s.R_E, hash(s),
							   False) #s.is_from_last_chosen) 
		for s in subgoals
	]

	cost, ordering = lsp_accel.get_lowest_cost_ordering(
		s_cpp, rd_cpp, gd_cpp, fd_cpp)
	#print(f'ordering = {ordering}')
	ordering = [s_dict[sid] for sid in ordering]

	return cost, ordering