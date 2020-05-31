from causal_rl_bench.task_generators.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
import numpy as np
import copy
from itertools import compress
from causal_rl_bench.utils.state_utils import get_intersection


class StackedBlocksGeneratorTask(BaseTask):
    def __init__(self, **kwargs):
        """
        This task generator
        will most probably deal with camera data if u want to use the
        sample goal function
        :param kwargs:
        """
        super().__init__(task_name="stacked_blocks",
                         intervention_split=kwargs.get("intervention_split",
                                                       False),
                         training=kwargs.get("training", True),
                         sparse_reward_weight=
                         kwargs.get("sparse_reward_weight", 1),
                         dense_reward_weights=
                         kwargs.get("dense_reward_weights",
                                    np.array([])))
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "action_joint_positions",
                                            "end_effector_positions"]

        #for this task the stage observation keys will be set with the
        #goal/structure building
        self.task_params["tool_block_mass"] = \
            kwargs.get("tool_block_mass", 0.08)
        self.task_params["joint_positions"] = \
            kwargs.get("joint_positions", None)
        self.task_params["blocks_min_size"] = \
            kwargs.get("blocks_min_size", np.array([0.035, 0.035, 0.035]))
        self.task_params["num_of_levels"] = \
            kwargs.get("num_of_levels", 8)
        self.task_params["max_level_width"] = \
            kwargs.get("max_level_width", 0.12)
        self.previous_end_effector_positions = None
        self.previous_object_position = None
        self.previous_object_orientation = None

    def get_description(self):
        return "Task where the goal is to stack arbitrary shapes of cuboids"

    def _set_up_stage_arena(self):
        number_of_blocks_per_level = int(self.task_params["max_level_width"] \
                                     / self.task_params["blocks_min_size"][0])
        default_start_position = -(number_of_blocks_per_level *
                                  self.task_params["blocks_min_size"][0])/2
        default_start_position += self.task_params["blocks_min_size"][0]/2
        curr_height = 0.01 - self.task_params["blocks_min_size"][-1]/2
        change_per_level = 0.005
        rigid_block_side = 0.1
        for level in range(self.task_params["num_of_levels"]):
            change_per_level *= -1
            curr_height += self.task_params["blocks_min_size"][-1]
            start_position = default_start_position + change_per_level
            rigid_block_side *= -1
            for i in range(number_of_blocks_per_level):
                self.stage.add_silhoutte_general_object(name="goal_"+"level_"+
                                                             str(level)+"_num_"+
                                                             str(i),
                                                        shape="cube",
                                                        position=[start_position,
                                                                  0, curr_height],
                                                        orientation=[0, 0, 0, 1],
                                                        size=self.task_params
                                                        ["blocks_min_size"])
                self.stage.add_rigid_general_object(name="tool_" + "level_" +
                                                          str(level) + "_num_" +
                                                          str(i),
                                                    shape="cube",
                                                    position=[start_position,
                                                              rigid_block_side,
                                                              curr_height],
                                                    orientation=[0, 0, 0, 1],
                                                    size=self.task_params
                                                    ["blocks_min_size"])
                start_position += self.task_params["blocks_min_size"][0]
        if self.task_params["joint_positions"] is not None:
            self.initial_state['joint_positions'] = \
                self.task_params["joint_positions"]
        return

    # def generate_new_goal(self):
    #
    #     self._create_new_challenge()
    #     # self.reset_task()
    #     return

    def _create_new_challenge(self):
        self.stage.remove_everything()
        self.current_stack_levels = self.task_params["num_of_levels"]
        block_sizes, positions, chosen_y = self._generate_random_target(
            num_of_levels=self.task_params["num_of_levels"],
            min_size=self.task_params["blocks_min_size"])
        for level_num in range(len(block_sizes)):
            for i in range(len(block_sizes[level_num])):
                self.stage.add_rigid_general_object(
                    name="tool_" + "level_" + str(level_num) + "_num_" + str(
                        i),
                    shape="cube",
                    size=block_sizes[level_num][i],
                    color=np.random.uniform(0, 1, size=[3]),
                    mass=self.task_params["tool_block_mass"])
                block_position = self.stage.random_position(
                    height_limits=0.0425)
                block_orientation = euler_to_quaternion(
                    [0, 0, np.random.uniform(-np.pi, np.pi)])
                self.stage.set_objects_pose(names=[
                    "tool_" + "level_" + str(level_num) + "_num_" + str(i)],
                    positions=[block_position],
                    orientations=[block_orientation])
                trial_index = 0
                while not self.stage.check_feasiblity_of_stage() and \
                        trial_index < 10:
                    block_position = self.stage.random_position(
                        height_limits=[0.0425, 0.15])
                    block_orientation = euler_to_quaternion(
                        [0, 0, np.random.uniform(-np.pi, np.pi)])
                    self.stage.set_objects_pose(names=[
                        "tool_" + "level_" + str(level_num) + "_num_" + str(
                            i)],
                        positions=[block_position],
                        orientations=[
                            block_orientation])
                    trial_index += 1
                silhouette_position = [positions[level_num][i], chosen_y,
                                       (level_num+1) *
                                       self.task_params["blocks_min_size"][-1]
                                       + (-self.task_params["blocks_min_size"]
                                          [-1]/2+0.01)]
                print(silhouette_position)
                self.stage.add_silhoutte_general_object(name="goal_" +
                                                             "level_" +
                                                             str(level_num)+
                                                             "_num_"+str(i),
                                                             shape="cube",
                                                        position=
                                                        np.array(silhouette_position),
                                                        size=
                                                        np.array(block_sizes[level_num][i]))
        return

    def _generate_random_block(self, allowed_boundaries,
                               start_z, min_size=np.array([0.035, 0.035,
                                                           0.035]),
                               max_level_width=0.12):
        """
        This function will return a random position and size of a block
        while respecting the allowed boundaries passed
        :param allowed_boundaries:
        :param start_z:
        :param min_size:
        :return:
        """
        allowed_boundaries[0][0] = max(self.stage.floor_inner_bounding_box[0][0]
                                       + min_size[0],
                                       allowed_boundaries[0][0])
        allowed_boundaries[1][0] = min(self.stage.floor_inner_bounding_box[1][0]
                                       - min_size[0],
                                       allowed_boundaries[1][0])

        allowed_boundaries[0][1] = max(self.stage.floor_inner_bounding_box[0][1]
                                       + min_size[1],
                                       allowed_boundaries[0][1])
        allowed_boundaries[1][1] = min(self.stage.floor_inner_bounding_box[1][1]
                                       - min_size[1],
                                       allowed_boundaries[1][1])
        position_x_y = np.random.uniform(allowed_boundaries[0][:2],
                                         allowed_boundaries[1][:2])
        # choose size width, depth, height
        allowed_max_width = min(
            self.stage.floor_inner_bounding_box[1][0] - position_x_y[0],
            position_x_y[0] - self.stage.floor_inner_bounding_box[0][0]) * 2
        allowed_max_width = min(allowed_max_width, max_level_width)
        size = np.random.uniform(min_size,
                                 [allowed_max_width, min_size[1], min_size[2]])
        position_z = start_z + size[-1]/2
        position = np.array([position_x_y[0], position_x_y[1], position_z])
        return size, position

    def _generate_random_target(self, num_of_levels=4,
                                min_size=np.array([0.035, 0.035, 0.035])):
        """
        This function generated a sampled target, should be modified to new sample goal
        :param levels_num:
        :param min_size:
        :return:
        """
        level_blocks = []
        current_boundaries = np.array([self.stage.floor_inner_bounding_box[0]
                                       [:2],
                                       self.stage.floor_inner_bounding_box[1][
                                       :2]])
        start_z = self.stage.floor_height
        level_index = 0
        size, position = self._generate_random_block(
            allowed_boundaries=current_boundaries, start_z=start_z,
            min_size=min_size)
        level_blocks.append([[size[0], position[0]]])
        for level_index in range(1, num_of_levels):
            start_z = start_z + size[-1]
            new_allowed_boundaries = [position[:2] - size[:2] / 2,
                                      position[:2] + size[:2] / 2]
            current_boundaries = [np.maximum(current_boundaries[0],
                                             new_allowed_boundaries[0]),
                                  np.minimum(current_boundaries[1],
                                             new_allowed_boundaries[1])]
            size, position = self._generate_random_block(
                allowed_boundaries=current_boundaries,
                start_z=start_z, min_size=min_size)
            level_blocks.append([[size[0], position[0]]])
        chosen_y = position[1]
        new_level_blocks = \
            self._generate_blocks_to_use(level_blocks,
                                         min_size=min_size[0])
        new_level_blocks = \
            self._generate_blocks_to_use(new_level_blocks,
                                         min_size=min_size[0])
        new_level_blocks = \
            self._generate_blocks_to_use(new_level_blocks,
                                         min_size=min_size[0])
        #now return the actual block sizes that needs to be created
        block_sizes, positions = self._get_block_sizes(new_level_blocks,
                                                       min_size)

        return block_sizes, positions, chosen_y

    def _generate_blocks_to_use(self, level_blocks,
                                min_size):
        new_level_blocks = list(level_blocks)
        for i in range(len(level_blocks)):
            current_level_blocks = level_blocks[i]
            # try splitting randomly 4 times if the big block at
            # least has size of twice the min size
            for j in range(len(current_level_blocks)):
                if current_level_blocks[j][0] > min_size * 2:
                    # try splitting to half the size
                    # TODO: split a random split and
                    # check for stability instead
                    block_1_center = current_level_blocks[j][1] + \
                                     current_level_blocks[j][0] / 4
                    block_1_size = current_level_blocks[j][0] / 2
                    block_2_size = current_level_blocks[j][
                                       0] - block_1_size
                    block_2_center = block_1_center - block_1_size / 2 - \
                                     block_2_size / 2
                    # now check for stability of
                    # structure with the previous levels
                    stability_levels_check = copy.deepcopy(
                        new_level_blocks[:i + 1])
                    stability_levels_check[i][j] = [block_1_size,
                                                    block_1_center]
                    stability_levels_check[i].append(
                        [block_2_size, block_2_center])
                    if self._is_stable_structure(stability_levels_check):
                        new_level_blocks[:i + 1] = stability_levels_check
        return new_level_blocks

    def _is_stable_structure(self, level_blocks):
        # [[[width, center][width, center]]]
        current_min = -0.5
        current_max = 0.5
        for i in range(len(level_blocks)):
            current_level_blocks = level_blocks[i]
            new_min = 0.5
            new_max = -0.5
            # now check that for each block its
            # center lies within the current min and current max
            for block in current_level_blocks:
                new_min = min(new_min, block[1] - block[0] / 2)
                new_max = max(new_max, block[1] + block[0] / 2)
                if block[1] > current_max or block[1] < current_min:
                    return False
            current_min = new_min
            current_max = new_max
        return True

    def _get_block_sizes(self, level_blocks, min_size):
        block_sizes = []
        positions = []
        for i in range(len(level_blocks)):
            block_sizes.append([])
            positions.append([])
            current_level_blocks = level_blocks[i]
            for block in current_level_blocks:
                block_sizes[-1].append([block[0], min_size[1], min_size[2]])
                positions[-1].append(block[1])
        return block_sizes, positions

    def sample_new_goal(self, training=True):
        return self._create_new_challenge()
