import numpy as np
import numpy.linalg as LA
import cv2
import math
import matplotlib.patches as patches
import networkx as nx
import random
import habitat
import habitat_sim
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from .baseline_utils import convertInsSegToSSeg
import matplotlib.pyplot as plt


def change_brightness(img, flag, value=30):
    """ change brightness of the img at the area with flag=True. """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    #lim = 255 - value
    #v[v > lim] = 255
    #v[v <= lim] += value

    v[np.logical_and(flag == False, v > value)] -= value
    v[np.logical_and(flag == False, v <= value)] = 0

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


class SimpleRLEnv(habitat.RLEnv):
    """ simple RL environment to initialize habitat navigation episodes."""

    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def get_scene_name(episode):
    """ extract the episode name from the long directory. """
    idx_right_most_slash = episode.scene_id.rfind('/')
    return episode.scene_id[idx_right_most_slash + 1:-4]


def verify_img(img):
    """ verify if the image 'img' has blank pixels. """
    sum_img = np.sum((img[:, :, 0] > 0))
    h, w = img.shape[:2]
    return sum_img > h * w * 0.75


def get_obs_and_pose(env, agent_pos, heading_angle, keep=True):
    """ get observation 'obs' at agent pose 'agent_pos' and orientation 'heading_angle' at current scene 'env'."""
    agent_rot = habitat_sim.utils.common.quat_from_angle_axis(
        heading_angle, habitat_sim.geo.GRAVITY)
    #print(f'agent_pos = {agent_pos}, agent_rot = {agent_rot}')
    obs = env.get_observations_at(agent_pos,
                                  agent_rot,
                                  keep_agent_at_new_pose=keep)

    #============= get allocentric pose ================
    agent_pos = env.get_agent_state().position
    agent_rot = env.get_agent_state().rotation
    #print(f'agent_pos = {agent_pos}, agent_rot = {agent_rot}')
    heading_vector = quaternion_rotate_vector(agent_rot.inverse(),
                                              np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    angle = phi
    allo_pose = (agent_pos[0], agent_pos[2], angle)

    #================ compute relative pose =================
    rel_pose = (0, 0, 0)

    '''
	rgb_img = obs['rgb']
	depth_img = 5. * obs['depth']
	depth_img = cv2.blur(depth_img, (3, 3))
	#print(f'depth_img.shape = {depth_img.shape}')
	InsSeg_img = obs["semantic"]
	#sseg_img = convertInsSegToSSeg(InsSeg_img, self.ins2cat_dict)

	if True:
		fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
		ax[0].imshow(rgb_img)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title("rgb")
		ax[1].imshow(InsSeg_img)
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title("sseg")
		ax[2].imshow(depth_img)
		ax[2].get_xaxis().set_visible(False)
		ax[2].get_yaxis().set_visible(False)
		ax[2].set_title("depth")
		fig.tight_layout()
		plt.show()
	'''

    return obs, rel_pose, allo_pose


def get_obs_and_pose_by_action(env, act, prev_allo_pose=(0,0,0)):
    obs = env.step(act)

    #============= get allocentric pose ================
    agent_pos = env.get_agent_state().position
    agent_rot = env.get_agent_state().rotation
    #print(f'agent_pos = {agent_pos}, agent_rot = {agent_rot}')
    heading_vector = quaternion_rotate_vector(agent_rot.inverse(),
                                              np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    angle = phi
    allo_pose = (agent_pos[0], agent_pos[2], angle)

    #================ compute relative pose =================
    rel_pose = tuple(np.array(allo_pose) - np.array(prev_allo_pose))

    return obs, rel_pose, allo_pose

def get_obs_and_pose_from_habitat_lab_env(env, obs_and_info, prev_allo_pose=None):
    obs = {}
    obs['rgb'] = obs_and_info['rgb']
    obs['depth'] = obs_and_info['depth']
    obs['semantic'] = obs_and_info['semantic']

    #rel_pose = tuple(obs_and_info['gps'])

    #============= get allocentric pose ================
    agent_pos = env.get_agent_state().position
    agent_rot = env.get_agent_state().rotation
    #print(f'agent_pos = {agent_pos}, agent_rot = {agent_rot}')
    heading_vector = quaternion_rotate_vector(agent_rot.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    angle = phi
    allo_pose = (agent_pos[0], agent_pos[2], angle)

    #================ compute relative pose =================
    if not prev_allo_pose:
        rel_pose = (0, 0, 0)
    else:
        rel_pose = tuple(np.array(allo_pose) - np.array(prev_allo_pose))

    return obs, rel_pose, allo_pose



# Return success, SPL, soft_SPL, distance_to_goal measures
def get_metrics(env, episode_goal_positions, success_distance,
                start_end_episode_distance, agent_episode_distance,
                stop_signal):

    curr_pos = env.get_agent_state().position

    # returns distance to the closest goal position, make sure goal height is same as current pose height
    distance_to_goal = env.geodesic_distance(
        curr_pos,
        [[episode_goal_positions[0], curr_pos[1], episode_goal_positions[2]]])

    if distance_to_goal <= success_distance and stop_signal:
        success = 1.0
    else:
        success = 0.0

    spl = success * (start_end_episode_distance /
                     max(start_end_episode_distance, agent_episode_distance))

    ep_soft_success = max(0,
                          (1 - distance_to_goal / start_end_episode_distance))
    soft_spl = ep_soft_success * (start_end_episode_distance / max(
        start_end_episode_distance, agent_episode_distance))

    nav_metrics = {
        'distance_to_goal': distance_to_goal,
        'success': success,
        'spl': spl,
        'softspl': soft_spl
    }

    print(f'========> {nav_metrics}')
    return nav_metrics
