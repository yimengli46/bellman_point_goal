from math import sqrt
from operator import itemgetter
import lsp_accel

class fron(object):
	def __init__(self, P_S, R_S, R_E, center):
		self.P_S = P_S
		self.R_S = R_S
		self.R_E = R_E 
		self.center = center

	def __str__(self):
		return f'center = {self.center}'

	def __repr__(self):
		return f'center = {self.center}'

def compute_Q(agent_coord, target_frontier, frontiers, visited_frontiers):
	print(f'agent_coord = {agent_coord}, target_frontier = {target_frontier.center}')
	Q = 0
	D = abs(agent_coord[0] - target_frontier.center[0]) + abs(agent_coord[1] - target_frontier.center[1])

	Q += D + target_frontier.P_S * target_frontier.R_S

	visited_frontiers.add(target_frontier)
	rest_frontiers = frontiers - visited_frontiers

	min_next_Q = 0

	for fron in rest_frontiers:
		print(f'visited_frontiers = {visited_frontiers}')
		next_Q = compute_Q(target_frontier.center, fron, frontiers, visited_frontiers)
		if next_Q < min_next_Q:
			min_next_Q = next_Q

	Q += (1 - target_frontier.P_S) * (target_frontier.R_E + min_next_Q)

	return Q

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
							   s.is_from_last_chosen) for s in subgoals
	]

	cost, ordering = lsp_accel.get_lowest_cost_ordering(
		s_cpp, rd_cpp, gd_cpp, fd_cpp)
	ordering = [s_dict[sid] for sid in ordering]

	return cost, ordering



frontiers = set()
f1 = fron(0.8, 15, 30, (10, 10))
f2 = fron(0.6, 12, 30, (15, 15))
f3 = fron(0.1, 13, 30, (20, 20))


frontiers.add(f1)
frontiers.add(f2)
frontiers.add(f3)


agent_start = (0, 0)

#for steps in list(range(10, 300, 10)):
min_Q = 1e10
min_frontier = None
for fron in frontiers:
	print('-------------------------------------------------------------')
	visited_frontiers = set()
	Q = compute_Q(agent_start, fron, frontiers, visited_frontiers)
	print(f'fron = {fron.center}, Q = {Q}')
	if Q < min_Q:
		min_Q = Q
		min_frontier = fron

print(f'min_frontier = {min_frontier.center}')

