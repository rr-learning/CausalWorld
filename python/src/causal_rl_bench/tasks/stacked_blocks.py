from causal_rl_bench.tasks.base_task import BaseTask
from causal_rl_bench.utils.rotation_utils import euler_to_quaternion
import numpy as np
from causal_rl_bench.utils.state_utils import get_iou


class StackedBlocksTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="picking")
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "action_joint_positions",
                                            "end_effector_positions"]
        # self.task_stage_observation_keys = ["block_position"]
        # self.task_params["block_mass"] = kwargs.get("block_mass", 0.02)
        self.task_params["randomize_joint_positions"] = kwargs.get(
            "randomize_joint_positions", True)
        # self.task_params["randomize_block_pose"] = kwargs.get(
        #     "randomize_block_pose", True)
        # self.task_params["goal_height"] = kwargs.get("goal_height", 0.15)
        # self.task_params["reward_weight_1"] = kwargs.get("reward_weight_1", 1)
        # self.task_params["reward_weight_2"] = kwargs.get("reward_weight_2", 1)
        # self.previous_object_position = None

    # def _generate_random_level_block(self, allowed_base_section):
    #     #first generate random size block
    #     width = allowed_base_section[1][0] - allowed_base_section[0][0]
    #     depth = allowed_base_section[1][1] - allowed_base_section[0][1]
    #     height = allowed_base_section[1][2] - allowed_base_section[0][2]
    #     size_upper_bound = [width/2, depth/2, height/2]
    #     size = np.random.uniform([0.035, 0.035, 0.035], size_upper_bound)
    #     #TODO: clip the height maybe
    #     size[-1] = min(size[-1], 0.065)
    #     #now choose the position and orientation (restrict to 0. 0. 0 for now)
    #     new_low_limits = allowed_base_section[0] - size/2
    #     new_upper_limits = allowed_base_section[1] - size / 2
    #     position = np.random.uniform(new_low_limits, new_upper_limits)
    #     position[-1] = allowed_base_section[0][2] + size[-2]/2
    #     return size, position

    def _generate_random_level_block(self, allowed_center_position):
        #modify allowed center position to fit the stage and minimum size
        allowed_center_position[0][0] = max(-0.1475, allowed_center_position[0][0])
        allowed_center_position[1][0] = min(0.1475, allowed_center_position[1][0])

        allowed_center_position[0][1] = max(-0.1475, allowed_center_position[0][1])
        allowed_center_position[1][1] = min(0.1475, allowed_center_position[1][1])
        #first choose the center position
        position_x_y = np.random.uniform(allowed_center_position[0][:2], allowed_center_position[1][:2])
        #choose size width, depth, height
        allowed_max_width = min(0.1825 - position_x_y[0], position_x_y[0] + 0.1825) * 2
        allowed_max_depth = min(0.1825 - position_x_y[1], position_x_y[1] + 0.1825) * 2
        allowed_max_height = allowed_center_position[1][2] - allowed_center_position[0][2]
        # size = np.random.uniform([0.035, 0.035, 0.035], [allowed_max_width, allowed_max_depth,
        #                                                  allowed_max_height])
        size = np.random.uniform([0.035, 0.035, 0.035], [allowed_max_width, 0.035,
                                                         0.035])
        position_z = allowed_center_position[0][2] + size[-1]/2
        position = np.array([position_x_y[0], position_x_y[1], position_z])
        return size, position

    def _set_up_stage_arena(self):
        # self.stage.add_rigid_general_object(name="block",
        #                                     shape="cube",
        #                                     mass=self.task_params["block_mass"])
        #generate new level block
        size, position = self._generate_random_level_block(allowed_center_position=[[-0.1825, -0.1825, 0.01],
                                                                                   [0.1825, 0.1825, 0.1]])
        self.stage.add_silhoutte_general_object(name="goal_position",
                                                shape="cube",
                                                position=position,
                                                size=size)
        #next level
        new_limits = [position - size/2, position + size/2]
        new_limits[0][-1] = new_limits[1][-1]
        new_limits[1][-1] = new_limits[0][-1] + 0.1
        size, position = self._generate_random_level_block(allowed_center_position=new_limits)
        self.stage.add_silhoutte_general_object(name="goal_position_2",
                                                shape="cube",
                                                position=position,
                                                size=size)
        # next level
        new_limits = [position - size / 2, position + size / 2]
        new_limits[0][-1] = new_limits[1][-1]
        new_limits[1][-1] = new_limits[0][-1] + 0.1
        size, position = self._generate_random_level_block(allowed_center_position=new_limits)
        self.stage.add_silhoutte_general_object(name="goal_position_3",
                                                shape="cube",
                                                position=position,
                                                size=size)
        # next level
        new_limits = [position - size / 2, position + size / 2]
        new_limits[0][-1] = new_limits[1][-1]
        new_limits[1][-1] = new_limits[0][-1] + 0.1
        size, position = self._generate_random_level_block(allowed_center_position=new_limits)
        self.stage.add_silhoutte_general_object(name="goal_position_4",
                                                shape="cube",
                                                position=position,
                                                size=size)
        # next level
        new_limits = [position - size / 2, position + size / 2]
        new_limits[0][-1] = new_limits[1][-1]
        new_limits[1][-1] = new_limits[0][-1] + 0.1
        size, position = self._generate_random_level_block(allowed_center_position=new_limits)
        self.stage.add_silhoutte_general_object(name="goal_position_5",
                                                shape="cube",
                                                position=position,
                                                size=size)
        # next level
        new_limits = [position - size / 2, position + size / 2]
        new_limits[0][-1] = new_limits[1][-1]
        new_limits[1][-1] = new_limits[0][-1] + 0.1
        size, position = self._generate_random_level_block(allowed_center_position=new_limits)
        self.stage.add_silhoutte_general_object(name="goal_position_6",
                                                shape="cube",
                                                position=position,
                                                size=size)
        # next level
        new_limits = [position - size / 2, position + size / 2]
        new_limits[0][-1] = new_limits[1][-1]
        new_limits[1][-1] = new_limits[0][-1] + 0.1
        size, position = self._generate_random_level_block(allowed_center_position=new_limits)
        self.stage.add_silhoutte_general_object(name="goal_position_7",
                                                shape="cube",
                                                position=position,
                                                size=size)
        # next level
        new_limits = [position - size / 2, position + size / 2]
        new_limits[0][-1] = new_limits[1][-1]
        new_limits[1][-1] = new_limits[0][-1] + 0.1
        size, position = self._generate_random_level_block(allowed_center_position=new_limits)
        self.stage.add_silhoutte_general_object(name="goal_position_8",
                                                shape="cube",
                                                position=position,
                                                size=size)
        return

    def _reset_task(self):
        if self.task_params["randomize_joint_positions"]:
            positions = self.robot.sample_joint_positions()
        else:
            positions = self.robot.get_rest_pose()[0]
        self.robot.set_full_state(np.append(positions,
                                            np.zeros(9)))

        # # reset stage next
        # if self.task_params["randomize_block_pose"]:
        #     block_position = self.stage.random_position(height_limits=0.0425)
        #     block_orientation = euler_to_quaternion([0, 0,
        #                                              np.random.uniform(-np.pi,
        #                                                                np.pi)])
        # else:
        #     block_position = [0, 0, 0.0425]
        #     block_orientation = euler_to_quaternion([0, 0, 0])
        # goal_position = [0, 0, self.task_params["goal_height"]]
        # goal_orientation = euler_to_quaternion([0, 0, 0])
        # self.stage.set_objects_pose(names=["block", "goal_position"],
        #                             positions=[block_position, goal_position],
        #                             orientations=[block_orientation, goal_orientation])
        # self.previous_object_position = block_position
        return

    def get_description(self):
        return "Task where the goal is to pick a " \
               "cube towards a goal height"

    def get_reward(self):
        # block_position = self.stage.get_object_state('block', 'position')
        # target_height = self.task_params["goal_height"]
        #
        # #reward term one
        # previous_block_to_goal = -abs(self.previous_object_position[2] -
        #                               target_height)
        # current_block_to_goal = -abs(block_position[2] - target_height)
        # reward_term_1 = previous_block_to_goal - current_block_to_goal
        #
        # # reward term two
        # previous_block_to_center = -(self.previous_object_position[0]**2 +
        #                             self.previous_object_position[1]**2)
        # current_block_to_center = -(block_position[0] ** 2 +
        #                             block_position[1] ** 2)
        # reward_term_2 = previous_block_to_center - current_block_to_center
        #
        # reward = self.task_params["reward_weight_1"] * reward_term_1 + \
        #          self.task_params["reward_weight_2"] * reward_term_2
        # #TODO: discuss termination conditions
        # # if abs(z - target_height) < 0.02:
        # #     self.task_solved = True
        # self.previous_object_position = block_position
        return 0

    def is_done(self):
        return self.task_solved

    def do_random_intervention(self):
        interventions_dict = dict()
        new_block_position = self.stage.random_position(height_limits=0.0425)
        new_colour = np.random.uniform([0], [1], size=[3, ])
        interventions_dict["position"] = new_block_position
        interventions_dict["colour"] = new_colour
        new_size = np.random.uniform([0.065], [0.15], size=[3, ])
        interventions_dict["size"] = new_size
        self.stage.object_intervention("block", interventions_dict)
        self.previous_object_position = new_block_position
        return

    def do_intervention(self, **kwargs):
        raise Exception("not yet implemented")

