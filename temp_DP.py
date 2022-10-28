from math import sqrt
from operator import itemgetter

class fron(object):
	def __init__(self, R, D, center):
		self.R = R
		self.D = D 
		self.center = center

	def __str__(self):
		return f'center = {self.center}'

	def __repr__(self):
		return f'center = {self.center}'

def compute_Q(agent_coord, target_frontier, frontiers, visited_frontiers, steps):
	print(f'agent_coord = {agent_coord}, target_frontier = {target_frontier.center}, steps = {steps}')
	Q = 0
	L = abs(agent_coord[0] - target_frontier.center[0]) + abs(agent_coord[1] - target_frontier.center[1])

	# cond 1: agent has enough steps to reach target_frontier
	if steps > L:
		steps -= L

		# cond 2: agent does not have enough steps to traverse target_frontier:
		if steps <= target_frontier.D:
			Q += 1. * steps / target_frontier.D * target_frontier.R
			steps = 0
		else:
			steps -= target_frontier.D
			Q += target_frontier.R
			# cond 3: agent does have enough steps to go back to the entry of target_frontier
			if steps >= target_frontier.D:
				steps -= target_frontier.D
				visited_frontiers.add(target_frontier)
				rest_frontiers = frontiers - visited_frontiers

				max_next_Q = 0
				max_rest_steps = 0
				for fron in rest_frontiers:
					print(f'visited_frontiers = {visited_frontiers}')
					next_Q, rest_steps = compute_Q(target_frontier.center, fron, frontiers, visited_frontiers, steps)
					if next_Q > max_next_Q:
						max_next_Q = next_Q
						max_rest_steps = rest_steps
					if next_Q == max_next_Q and rest_steps > max_rest_steps:
						max_next_Q = next_Q
						max_rest_steps = rest_steps
				Q += max_next_Q
				steps = max_rest_steps
	#print(f'Q = {Q}')
	return Q, steps


frontiers = set()
f1 = fron(50, 50, (10, 10))
f2 = fron(20, 20, (15, 15))
f3 = fron(10, 10, (20, 20))
#f4 = fron(50, 50, (10, 10))
#f5 = fron(20, 20, (15, 15))
#f6 = fron(10, 10, (20, 20))


frontiers.add(f1)
frontiers.add(f2)
frontiers.add(f3)
#frontiers.add(f4)
#frontiers.add(f5)
#frontiers.add(f6)

agent_start = (0, 0)
steps = 200



#for steps in list(range(10, 300, 10)):
max_Q = 0
max_steps = 0
max_frontier = None
for fron in frontiers:
	print('-------------------------------------------------------------')
	visited_frontiers = set()
	Q, rest_steps = compute_Q(agent_start, fron, frontiers, visited_frontiers, steps)
	if Q >= max_Q:
		max_Q = Q
		max_steps = rest_steps
		max_frontier = fron
	elif Q == max_Q and rest_steps > max_steps: #hash(fron) > hash(max_frontier):
		max_Q = Q
		max_steps = rest_steps
		max_frontier = fron

print(f'steps = {steps}, max_frontier = {max_frontier.center}')

