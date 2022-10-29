import random
import string
from organization.utilities import utils
from itertools import permutations

# GROUND_NAME = 'grey'
REGIONS = ["yellow", "green", "violet", "white"]
COLORS = ["red", "orange", "yellow", "green", "blue", "violet", "white", "black"]
COLOR_FROM_NAME = {
    "stove": "red",
    "table": "brown",
    "shelf": "grey",
}


def _draw_block(self, x, y, width, height, name="", color="blue"):
    """
    This function overrides the draw_block() from PDDLStream. The only change here is
    there is no name assigned to the block.
    """

    self.dynamic.extend(
        [
            self.canvas.create_rectangle(
                self.scale_x(x - width / 2.0),
                self.scale_y(y),
                self.scale_x(x + width / 2.0),
                self.scale_y(y + height),
                fill=color,
                outline="black",
                width=2,
            )
        ]
    )


def _draw_region(self, region, name=""):
    """
    This function overrides the draw_region() from PDDLStream. The only change here is
    there is no name assigned to the block.
    """

    x1, x2 = map(self.scale_x, region)
    y1, y2 = self.ground_height, self.height
    color = COLOR_FROM_NAME.get(name, name)
    self.static.extend(
        [
            self.canvas.create_rectangle(
                x1, y1, x2, y2, fill=color, outline="black", width=2
            )
        ]
    )


def generate_regions(env_num):
    """
    This generate random regions based on the seed env_num
    """

    random.seed(env_num)
    num_of_regions = random.randint(0, len(REGIONS))
    regions_sampled = random.sample(REGIONS, num_of_regions)
    # regions_sampled.insert(random.randint(0, len(regions_sampled)), 'red')
    regions_sampled.insert(0, "grey")
    regions_sampled.insert(1, "red")
    regions_sampled.insert(1, "blue")
    random.shuffle(regions_sampled)
    regions_pos = []
    region_dim = 20 / len(regions_sampled)
    y = -10 + region_dim
    for i in range(len(regions_sampled)):
        if regions_sampled[i] == "grey":
            region_pos_tuple = (-10, 10)
        else:
            region_pos_tuple = (-10 + (i * region_dim), y + (i * region_dim))
        regions_pos.append(region_pos_tuple)
    regions = dict(zip(regions_sampled, regions_pos))
    return regions


def gen_block_states_based_on_regions(region_pos):
    """
    This generate block states in which each state is when block is in the region
    """

    num_blocks = len(region_pos)
    blocks = [string.ascii_uppercase[j] for j in range(num_blocks)]
    block_pos_tuple_list = []
    region_pos = list(region_pos)
    for i in range(len(blocks)):
        curr_reg_pos = list(region_pos[i])

        if curr_reg_pos[0] < 0:
            curr_reg_pos[0] = curr_reg_pos[0] + 1
        else:
            curr_reg_pos[0] = curr_reg_pos[0] + 1

        if curr_reg_pos[1] < 0:
            curr_reg_pos[1] = curr_reg_pos[1] - 1
        else:
            curr_reg_pos[1] = curr_reg_pos[1] - 1

        block_pos = (random.uniform(curr_reg_pos[0], curr_reg_pos[1]), 0)
        block_pos_tuple_list.append(block_pos)

    initial_block_poses = dict(zip(blocks, block_pos_tuple_list))

    return initial_block_poses


def generate_init_block_poses(env_num, num_blocks, state_num, curr_block=None):
    """
    Generates random initial block poses. If there is overlap between block pose,
    correct_overlaps() is called.
    """

    random.seed(env_num)
    blocks = [string.ascii_uppercase[i] for i in range(num_blocks)]
    blocks_pos_x = utils.sample_with_minimum_distance(k=num_blocks, d=2)
    block_pos_tuple_list = []
    for i in range(len(blocks_pos_x)):
        block_pos = (blocks_pos_x[i], 0)
        block_pos_tuple_list.append(block_pos)

    initial_block_poses = dict(zip(blocks, block_pos_tuple_list))
    if state_num > 0:
        block_pose = (-9 + state_num / 2, 0)
        initial_block_poses[list(initial_block_poses.keys())[i]] = block_pose
        if curr_block:
            new_vals = correct_overlaps(initial_block_poses, curr_block)
            initial_block_poses = dict(zip(blocks, new_vals))

    return initial_block_poses


def gen_task(regions_list, num_blocks):
    """
    Generate task based on regions and number of blocks.
    """

    task = []
    blocks = [string.ascii_uppercase[i] for i in range(num_blocks)]
    target_nums = random.randint(1, len(blocks))
    target_blocks = random.sample(blocks, target_nums)
    if "grey" in regions_list:
        regions_list.remove("grey")

    target_regions = random.sample(regions_list, len(regions_list))
    for i in target_blocks:
        if len(regions_list) < 3:
            tau = (i, random.choice(target_regions))
            task.append(tau)
        elif len(regions_list) == 3:
            region = random.choice(target_regions)
            tau = (i, region)
            task.append(tau)
            count = 0
            for t1 in task:
                for t2 in task:
                    if t1 == t2:
                        continue
                    else:
                        if t1[1] == t2[1]:
                            count += 1
            if count > 2:
                task.remove(tau)
                target_regions.remove(region)
                region = random.choice(target_regions)
                tau = (i, region)
                task.append(tau)

        else:
            region = random.choice(target_regions)
            tau = (i, region)
            task.append(tau)
            target_regions.remove(region)

    return task


def get_region_dims(regions, num_blocks):
    """
    This extracts the all the region poses for blocks to be in
    to associate them to the states based on regions.
    """

    region_len = 20 / len(regions)
    region_vals = list(regions.values())
    grey_indx = region_vals.index((-10, 10))
    y = -10 + region_len
    region_vals[grey_indx] = (
        -10 + (grey_indx * region_len),
        y + (grey_indx * region_len),
    )
    region_poses = list(permutations(region_vals, num_blocks))
    return region_poses


def gen_simple_task_distribution():
    task1_prob = 0.5
    task2_prob = 0.5
    task_distribution = [[task1_prob, [("A", "red")]], [task2_prob, [("A", "blue")]]]

    return task_distribution


def correct_overlaps(block_pose_dict, block_num):
    """
    This function checks if the positions of the blocks overlap. If they do, correct the overlap.
    """

    list_of_vals = list(block_pose_dict.values())
    new_list = []
    if len(list_of_vals) == 1:
        new_list = list_of_vals
    else:
        for i in range(len(list_of_vals)):
            if i == block_num:
                continue
            else:
                diff = abs(list_of_vals[block_num][0] - list_of_vals[i][0])
                if diff <= 2:
                    if list_of_vals[i][0] <= 7:
                        new_tuple = (list_of_vals[i][0] + diff, 0)
                        new_list.append(new_tuple)
                    else:
                        new_tuple = (list_of_vals[i][0] - diff, 0)
                        new_list.append(new_tuple)
                else:
                    new_list.append(list_of_vals[i])

        new_list.insert(block_num, list_of_vals[block_num])
    return new_list


def return_image(view):
    """
    This function returns the image from PDDLStream viewer since it doesn't have
    it.
    """

    try:
        import pyscreenshot as ImageGrab
    except ImportError:
        print("Unable to load pyscreenshot")
        return None
    x, y = view.top.winfo_x(), 2 * view.top.winfo_y()
    width, height = view.top.winfo_width(), view.top.winfo_height()
    img = ImageGrab.grab((x, y, x + width, y + height))
    return img
