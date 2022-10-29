import lsp
import numpy as np
import vertexnav


def get_grid_data_from_obs(updated_frontiers, pose, goal,
                           max_range, num_range, num_bearing,
                           visibility_mask=None):
    """Get grid-projected data to train the vertexnav+lsp neural net."""

    # Initialize some vectors
    is_frontier_mat = np.zeros([num_range, num_bearing], dtype=np.float32)
    is_feasible_mat = np.zeros([num_range, num_bearing], dtype=np.float32)
    delta_success_cost_mat = np.zeros([num_range, num_bearing], dtype=np.float32)
    exploration_cost_mat = np.zeros([num_range, num_bearing], dtype=np.float32)
    positive_weighting_mat = np.zeros([num_range, num_bearing], dtype=np.float32)
    negative_weighting_mat = np.zeros([num_range, num_bearing], dtype=np.float32)

    pose = vertexnav.Pose(pose.x, pose.y, pose.yaw)

    # Lookup vectors
    vec_range, vec_bearing = lsp.utils.learning_vision.get_range_bearing_vecs(
        max_range, num_range, num_bearing)

    # Compute goal vecs
    goal_loc_x_vec, goal_loc_y_vec = lsp.utils.learning_vision.get_rel_goal_loc_vecs(
        pose, goal, num_bearing
    )

    for frontier in updated_frontiers:
        for f_point in frontier.points.T:
            f_point_center = f_point + 0.5
            is_inside, ind_range, ind_bearing = lsp.utils.learning_vision.get_range_bearing_indices(
                obs_pose=pose, lookup_point=f_point_center,
                vec_bearing=vec_bearing, vec_range=vec_range
            )

            # If out of range, ignore
            if not is_inside:
                continue

            if visibility_mask is not None and visibility_mask[f_point[0], f_point[1]] < 0:
                continue

            is_frontier_mat[ind_range, ind_bearing] = 1
            is_feasible_mat[ind_range, ind_bearing] = frontier.prob_feasible
            delta_success_cost_mat[ind_range, ind_bearing] = frontier.delta_success_cost
            exploration_cost_mat[ind_range, ind_bearing] = frontier.exploration_cost
            positive_weighting_mat[ind_range, ind_bearing] = frontier.positive_weighting
            negative_weighting_mat[ind_range, ind_bearing] = frontier.negative_weighting

    return {
        'is_frontier': is_frontier_mat,
        'is_feasible': is_feasible_mat,
        'delta_success_cost': delta_success_cost_mat,
        'exploration_cost': exploration_cost_mat,
        'positive_weighting': positive_weighting_mat,
        'negative_weighting': negative_weighting_mat,
        'goal_loc_x': goal_loc_x_vec,
        'goal_loc_y': goal_loc_y_vec,
    }
