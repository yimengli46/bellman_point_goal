import numpy as np
from numpy import ma
import skfmm

step_size = 5
num_rots = 12


class FMMPlanner():

    def __init__(self, traversible, num_rots):
        self.traversible = traversible
        self.angle_value = [
            0, -2.0 * np.pi / num_rots, 2.0 * np.pi / num_rots, 0
        ]
        self.du = step_size
        self.num_rots = num_rots
        self.action_list = self.search_actions()

    def set_goal(self, goal):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0]), int(goal[1])
        traversible_ma[goal_y, goal_x] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        '''
        plt.imshow(self.fmm_dist)
        plt.title(f'fmm_dist')
        plt.show()
        '''
        return dd_mask

    def search_actions(self):
        action_list = [[3], [0]]
        append_list_pos = []
        append_list_neg = []
        for i in range(self.num_rots):
            append_list_pos.append(1)
            append_list_neg.append(2)
            action_list.append(append_list_pos[:] + [3])
            action_list.append(append_list_neg[:] + [3])
        return action_list

    def _virtual_steps(self, u_list, state, check_collision=True):
        traversible = self.traversible
        goal_dist = self.fmm_dist
        angle_value = self.angle_value
        boundary_limits = np.array(traversible.shape)[::-1]
        x, y, t = state
        out_states = []
        cost_start = goal_dist[int(y), int(x)]  # Actual distance in cm.

        collision_reward = 0
        for i in range(len(u_list)):
            action = u_list[i]
            x_new, y_new, t_new = x * 1., y * 1., t * 1.
            if action == 3:
                angl = t
                x_new = x + np.cos(angl) * self.du
                y_new = y + np.sin(angl) * self.du
                t_new = angl
            elif action > 0:
                t_new = t + angle_value[action]

            collision_reward = -1
            inside = np.all(
                np.array([int(x_new), int(y_new)]) < np.array(boundary_limits))
            inside = inside and np.all(
                np.array([int(x_new), int(y_new)]) >= np.array([0, 0]))
            _new_state = [x, y, t]

            if inside:
                not_collided = True
                if action == 3 and check_collision:
                    for s in np.linspace(0, 1, self.du + 2):
                        _x = x * s + (1 - s) * x_new
                        _y = y * s + (1 - s) * y_new
                        not_collided = not_collided and traversible[int(_y),
                                                                    int(_x)]
                        if not_collided is False:
                            break
                if not_collided:
                    collision_reward = 0
                    x, y, t = x_new, y_new, t_new
                    _new_state = [x, y, t]
            out_states.append(_new_state)

        cost_end = goal_dist[int(y), int(x)]
        reward_near_goal = 0.
        if cost_end < self.du:
            reward_near_goal = 1.
        costs = (cost_end - cost_start)
        reward = -costs + reward_near_goal + collision_reward
        return reward, (out_states)

    def find_best_action_set(self, state, spacious=False, multi_act=0):
        goal_dist = self.fmm_dist
        traversible = self.traversible
        action_list = self.action_list
        best_list = [3]
        max_margin = 0
        obst_dist = []
        best_reward, state_list = self._virtual_steps(best_list, state)
        best_reward = best_reward + 0.1
        max_margin_state = state_list
        max_margin_act = [0]
        feasible_acts = []
        feasible_states = []
        sm_cut_reward, sm_state_list = self._virtual_steps([3], state)
        sm_cut_reward_zero, sm_state_list = self._virtual_steps([0], state)
        sm_cut_reward = max(sm_cut_reward, sm_cut_reward_zero)
        smarter_acts = []
        smarter_states = []
        st_lsts, rews = [], []
        for a_list in action_list:
            rew, st_lst = self._virtual_steps(a_list, state)
            # Prefer shorter action sequences.
            rew = rew - len(st_lst) * 0.1
            rews.append(rew)
            st_lsts.append(st_lst)

            if rew > best_reward:
                best_list = a_list
                best_reward = rew
                state_list = (st_lst)
            if False:  # rew > 4: #self.env.dilation_cutoff:
                current_margin = self.get_obst_dist(st_lst[-1])
                if current_margin > max_margin:
                    max_margin = current_margin
                    max_margin_state = st_lst
                    max_margin_act = a_list
            if rew > 0:
                feasible_acts.append(a_list)
                feasible_states.append(st_lst)
            if rew >= max(sm_cut_reward, 0):
                if a_list == [0] and rew < 1:
                    continue
                smarter_acts.append(a_list)
                smarter_states.append(st_lst)

        if not (len(best_list) == len(state_list)):
            print(len(best_list), len(state_list))
        if not spacious or (len(max_margin_act) == 1
                            and max_margin_act[0] == 0):
            # print(0, best_list, best_reward, np.array(rews))
            return best_list, state_list
        else:
            # print(1, max_margin_act, max_margin_state)
            return max_margin_act, max_margin_state

    def compare_goal(self, a, goal_dist):
        goal_dist = self.fmm_dist
        x, y, t = a
        cost_end = goal_dist[int(y), int(x)]
        dist = cost_end * 1.
        if dist < self.du * 1:
            return True
        return False

    def get_action(self, state):
        _ = self.find_best_action_set(state, False, 0)
        return _[0][0], _[1][0], _[0]
