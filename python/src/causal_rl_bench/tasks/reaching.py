from causal_rl_bench.tasks.base_task import BaseTask
import numpy as np
import math


class ReachingTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__(task_name="reaching")
        self.task_robot_observation_keys = ["joint_positions",
                                            "joint_velocities",
                                            "end_effector_positions",
                                            "action_joint_positions",
                                            "end_effector_positions_goal"]
        self.task_params["sparse_reward_weight"] = \
            kwargs.get("sparse_reward_weight", 0)
        self.task_params["dense_reward_weights"] = \
            kwargs.get("dense_reward_weights", np.array([4, 4, 4, 4]))
        default_goal = np.zeros([9, ])
        default_goal[2] = 0.1
        default_goal[5] = 0.15
        default_goal[-1] = 0.2
        self.end_effector_positions_goal = default_goal
        self.previous_end_effector_positions = None
        self.previous_joint_velocities = None

    def _set_up_stage_arena(self):
        self.stage.add_silhoutte_general_object(name="goal_1",
                                                shape="sphere",
                                                colour=np.array([1, 0, 0]))
        self.stage.add_silhoutte_general_object(name="goal_2",
                                                shape="sphere",
                                                colour=np.array([0, 1, 0]))
        self.stage.add_silhoutte_general_object(name="goal_3",
                                                shape="sphere",
                                                colour=np.array([0, 0, 1]))
        self.initial_state = dict()
        self.initial_state['joint_positions'] = self.robot.get_rest_pose()[0]
        self.initial_state['goal_positions'] = self.end_effector_positions_goal
        return

    def _set_up_non_default_observations(self):
        self._setup_non_default_robot_observation_key(
            observation_key="end_effector_positions_goal",
            observation_function=self._set_end_effector_positions_goal,
            lower_bound=[-0.5, -0.5, 0]*3,
            upper_bound=[0.5, 0.5, 0.5]*3)
        return

    def _set_end_effector_positions_goal(self):
        return self.end_effector_positions_goal

    def _reset_task(self, interventions_dict=None):
        #if there is not interventions dict passed then I
        # just apply the default one
        interventions_dict_copy = interventions_dict
        if interventions_dict_copy is not None:
            non_changed_variables = \
                set(self.initial_state) - set(interventions_dict_copy)
            if len(non_changed_variables) > 0:
                interventions_dict_copy = dict(interventions_dict)
            for non_changed_variable in non_changed_variables:
                interventions_dict_copy[non_changed_variable] = \
                    self.initial_state[non_changed_variable]

            self._apply_interventions(interventions_dict_copy,
                                      initial_state_latch=True)
        else:
            self._apply_interventions(self.initial_state,
                                      initial_state_latch=False)
        self.previous_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        self.previous_joint_velocities = np.copy(
            self.robot.latest_full_state.velocity)
        self.stage.set_objects_pose(
            names=["goal_1", "goal_2", "goal_3"],
            positions=[self.end_effector_positions_goal[:3],
                       self.end_effector_positions_goal[3:6],
                       self.end_effector_positions_goal[6:]],
            orientations=[None, None, None])
        return

    def get_description(self):
        return \
            "Task where the goal is to reach a point for each finger"

    def get_reward(self):
        current_end_effector_positions = \
            self.robot.compute_end_effector_positions(
                self.robot.latest_full_state.position)
        previous_dist_to_goal = np.linalg.norm(
            self.end_effector_positions_goal -
            self.previous_end_effector_positions)
        current_dist_to_goal = np.linalg.norm(self.end_effector_positions_goal
                                              - current_end_effector_positions)
        #TODO: now you need to call this to compute the done flag,
        # maybe discuss this
        sparse_reward = self._compute_sparse_reward(
            achieved_goal=current_end_effector_positions,
            desired_goal=self.end_effector_positions_goal,
            info=self.get_info())
        rewards = list()
        rewards.append(previous_dist_to_goal - current_dist_to_goal)
        rewards.append(-current_dist_to_goal)
        rewards.append(-np.linalg.norm(self.robot.latest_full_state.torque))
        rewards.append(-np.linalg.norm(np.abs(
            self.robot.latest_full_state.velocity - self.previous_joint_velocities),
                                        ord=2))
        reward = np.array(rewards) * self.task_params["dense_reward_weights"] \
                 + sparse_reward * self.task_params["sparse_reward_weight"]
        self.previous_end_effector_positions = current_end_effector_positions
        self.previous_joint_velocities = np.copy(
            self.robot.latest_full_state.velocity)
        return reward

    def _set_training_intervention_spaces(self):
        self.training_intervention_spaces = dict()
        self.training_intervention_spaces['joint_positions'] = \
            np.array([[-math.radians(70), -math.radians(70),
                       -math.radians(160)] * 3,
                       [math.radians(40), -math.radians(20),
                        -math.radians(30)] * 3])
        self.training_intervention_spaces['goal_positions'] = \
            np.array([[-0.5, -0.5, 0.0] * 3,
                      [0, 0, 0.2] * 3])

    def _set_testing_intervention_spaces(self):
        self.testing_intervention_spaces = dict()
        self.testing_intervention_spaces['joint_positions'] = \
            np.array([[math.radians(40), -math.radians(20),
                       -math.radians(30)] * 3,
                       [math.radians(70), 0,
                        math.radians(-2)] * 3])
        self.testing_intervention_spaces['goal_positions'] = \
            np.array([[0, 0, 0.2] * 3,
                      [0.5, 0.5, 0.5] * 3])

    def do_intervention(self, variable_name, variable_value,
                        sub_variable_name=None):
        #TODO: maybe allow intervention on joint velocities
        #TODO:check for feasibility of intervention here
        if variable_name == "joint_positions":
            self.robot.set_full_state(np.append(variable_value,
                                                np.zeros(9)))
        elif variable_name == "goal_positions":
            self.end_effector_positions_goal = variable_value
        else:
            raise Exception("This variable is not allowed for "
                            "interventions")
        return

    def get_current_student_params(self):
        student_params = dict()
        student_params['joint_positions'] = \
            np.array(self.robot.latest_full_state.position)
        return student_params

    def get_current_teacher_params(self):
        teacher_params = dict()
        teacher_params['goal_positions'] = \
            np.array(self.end_effector_positions_goal)
        return teacher_params


