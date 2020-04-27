from counterfactual.python.src.causal_rl_bench.tasks.task import Task


class Pushing(Task):
    def __init__(self, unit_length=0.65, shape="cube", orientation_reward=False, reward_mode="dense",
                 fixed_goal_position=False, fixed_goal_orientation=False, precision=None):
        super().__init__()

        self.unit_length = unit_length
        self.reward_mode = reward_mode
        self.orientation_reward = orientation_reward
        self.shape = shape
        self.precision = precision

        # Instantiate "pushing_object" of shape and unit length
        # Add collision object to scene objects

        # Instantiate "goal_silhouette", Sample initial goal position and goal orientation
        # Add silhouette object to scene objects

    def reset_task(self):
        pass

    def get_description(self):
        return "Task where the goal is to push an object towards a goal position"

    def get_reward(self):
        reward = 0.0
        done = False
        dist_pos_obj_goal = self.scene_objects["pushing_object"].position \
                            - self.scene_objects["goal_silhouette"].position
        dist_ori_obj_goal = self.scene_objects["pushing_object"].orientation \
                            - self.scene_objects["goal_silhouette"].orientation


    def get_counterfactual_variant(self):
        pass

    def reset_scene_objects(self):
        pass
