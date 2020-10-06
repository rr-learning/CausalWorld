.. _intervention_spaces:

============================
Variables Spaces Definition
============================

-------------------------
Reaching Variable Spaces
-------------------------

.. |br| raw:: html

   <br />

.. list-table:: Table of Reaching Intervention Spaces
   :widths: 25 25 40 40
   :header-rows: 1

   * - Variable
     - SubVariable
     - Space A
     - Space B
   * - joint_positions
     -  None
     - [[-1.57, -1.2, -3.0] * 3, |br| [-0.69, 0, 0] * 3]]
     - [[-0.69, 0, 0] * 3, |br| [1.0, 1.57, 3.0] * 3]
   * - goal_60 |br| or goal_120 |br| or goal_300
     - cylindrical_position
     - [[0.0, - math.pi, 0.0075], |br| [0.11, math.pi, 0.15]]
     - [[0.11, - math.pi, 0.0075] |br| [0.15, math.pi, 0.3]]
   * - goal_60 |br| or goal_120 |br| or goal_300
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - goal_60 |br| or goal_120 |br| or goal_300
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - floor_color
     -  None
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - stage_color
     -  None
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
     - [[0.5, 0.5, 0.5], |br| [1, 1, 1]]
   * - floor_friction
     -  None
     - [0.3, 0.6]
     - [0.6, 0.8]
   * - stage_friction
     -  None
     - [0.3, 0.6]
     - [0.6, 0.8]
   * - robot_finger_60_link_0 |br| or robot_finger_60_link_1 |br| or robot_finger_60_link_2 |br| or robot_finger_60_link_3 |br| or robot_finger_120_link_0 |br| or robot_finger_120_link_1 |br| or robot_finger_120_link_2 |br| or robot_finger_120_link_3 |br| or robot_finger_300_link_0 |br| or robot_finger_300_link_1 |br| or robot_finger_300_link_2 |br| or robot_finger_300_link_3
     -  color
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
     - [[0.5, 0.5, 0.5], |br| [1, 1, 1]]
   * - robot_finger_60_link_0 |br| or robot_finger_60_link_1 |br| or robot_finger_60_link_2 |br| or robot_finger_60_link_3 |br| or robot_finger_120_link_0 |br| or robot_finger_120_link_1 |br| or robot_finger_120_link_2 |br| or robot_finger_120_link_3 |br| or robot_finger_300_link_0 |br| or robot_finger_300_link_1 |br| or robot_finger_300_link_2 |br| or robot_finger_300_link_3
     -  mass
     - [0.015, 0.045]
     - [0.045, 0.1]
   * - number_of_obstacles
     -  None
     - [1, 5]
     - [1, 5]

-----------------------
Pushing Variable Spaces
-----------------------

- Notation: h refers to the height of the block.

.. list-table:: Table of Pushing Intervention Spaces
   :widths: 25 25 40 40
   :header-rows: 1

   * - Variable
     - SubVariable
     - Space A
     - Space B
   * - joint_positions
     -  None
     - [[-1.57, -1.2, -3.0] * 3, |br| [-0.69, 0, 0] * 3]]
     - [[-0.69, 0, 0] * 3, |br| [1.0, 1.57, 3.0] * 3]
   * - tool_block
     - cylindrical_position
     - [[0.0, - math.pi, h/2], |br| [0.11, math.pi, h/2]]
     - [[0.11, - math.pi, h/2] |br| [0.15, math.pi, h/2]]
   * - tool_block
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - tool_block
     -  size
     - [[0.055, 0.055, 0.055], |br| [0.075, 0.075, 0.075]]
     - [[0.075, 0.075, 0.075], |br| [0.095, 0.095, 0.095]]
   * - tool_block
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - goal_block
     - cylindrical_position
     - [[0.0, - math.pi, h/2], |br| [0.11, math.pi, h/2]]
     - [[0.11, - math.pi, h/2] |br| [0.15, math.pi, h/2]]
   * - goal_block
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - goal_block
     -  size
     - [[0.055, 0.055, 0.055], |br| [0.075, 0.075, 0.075]]
     - [[0.075, 0.075, 0.075], |br| [0.095, 0.095, 0.095]]
   * - goal_block
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - floor_color
     -  None
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - stage_color
     -  None
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
     - [[0.5, 0.5, 0.5], |br| [1, 1, 1]]
   * - floor_friction
     -  None
     - [0.3, 0.6]
     - [0.6, 0.8]
   * - stage_friction
     -  None
     - [0.3, 0.6]
     - [0.6, 0.8]
   * - robot_finger_60_link_0 |br| or robot_finger_60_link_1 |br| or robot_finger_60_link_2 |br| or robot_finger_60_link_3 |br| or robot_finger_120_link_0 |br| or robot_finger_120_link_1 |br| or robot_finger_120_link_2 |br| or robot_finger_120_link_3 |br| or robot_finger_300_link_0 |br| or robot_finger_300_link_1 |br| or robot_finger_300_link_2 |br| or robot_finger_300_link_3
     -  color
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
     - [[0.5, 0.5, 0.5], |br| [1, 1, 1]]
   * - robot_finger_60_link_0 |br| or robot_finger_60_link_1 |br| or robot_finger_60_link_2 |br| or robot_finger_60_link_3 |br| or robot_finger_120_link_0 |br| or robot_finger_120_link_1 |br| or robot_finger_120_link_2 |br| or robot_finger_120_link_3 |br| or robot_finger_300_link_0 |br| or robot_finger_300_link_1 |br| or robot_finger_300_link_2 |br| or robot_finger_300_link_3
     -  mass
     - [0.015, 0.045]
     - [0.045, 0.1]

-----------------------
Picking Variable Spaces
-----------------------

- Notation: h refers to the height of the block.

.. list-table:: Table of Picking Intervention Spaces
   :widths: 25 25 40 40
   :header-rows: 1

   * - Variable
     - SubVariable
     - Space A
     - Space B
   * - joint_positions
     -  None
     - [[-1.57, -1.2, -3.0] * 3, |br| [-0.69, 0, 0] * 3]]
     - [[-0.69, 0, 0] * 3, |br| [1.0, 1.57, 3.0] * 3]
   * - tool_block
     - cylindrical_position
     - [[0.0, - math.pi, h/2], |br| [0.11, math.pi, 0.15]]
     - [[0.11, - math.pi, 0.15] |br| [0.15, math.pi, 0.3]]
   * - tool_block
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - tool_block
     -  size
     - [[0.055, 0.055, 0.055], |br| [0.075, 0.075, 0.075]]
     - [[0.075, 0.075, 0.075], |br| [0.095, 0.095, 0.095]]
   * - tool_block
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - goal_block
     - cylindrical_position
     - [[0.0, - math.pi, 0.08], |br| [0.11, math.pi, 0.20]]
     - [[0.11, - math.pi, 0.20] |br| [0.15, math.pi, 0.25]]
   * - goal_block
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - goal_block
     -  size
     - [[0.055, 0.055, 0.055], |br| [0.075, 0.075, 0.075]]
     - [[0.075, 0.075, 0.075], |br| [0.095, 0.095, 0.095]]
   * - goal_block
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - floor_color
     -  None
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - stage_color
     -  None
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
     - [[0.5, 0.5, 0.5], |br| [1, 1, 1]]
   * - floor_friction
     -  None
     - [0.3, 0.6]
     - [0.6, 0.8]
   * - stage_friction
     -  None
     - [0.3, 0.6]
     - [0.6, 0.8]
   * - robot_finger_60_link_0 |br| or robot_finger_60_link_1 |br| or robot_finger_60_link_2 |br| or robot_finger_60_link_3 |br| or robot_finger_120_link_0 |br| or robot_finger_120_link_1 |br| or robot_finger_120_link_2 |br| or robot_finger_120_link_3 |br| or robot_finger_300_link_0 |br| or robot_finger_300_link_1 |br| or robot_finger_300_link_2 |br| or robot_finger_300_link_3
     -  color
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
     - [[0.5, 0.5, 0.5], |br| [1, 1, 1]]
   * - robot_finger_60_link_0 |br| or robot_finger_60_link_1 |br| or robot_finger_60_link_2 |br| or robot_finger_60_link_3 |br| or robot_finger_120_link_0 |br| or robot_finger_120_link_1 |br| or robot_finger_120_link_2 |br| or robot_finger_120_link_3 |br| or robot_finger_300_link_0 |br| or robot_finger_300_link_1 |br| or robot_finger_300_link_2 |br| or robot_finger_300_link_3
     -  mass
     - [0.015, 0.045]
     - [0.045, 0.1]

----------------------------
PickAndPlace Variable Spaces
----------------------------

- Notation: h refers to the height of the block.

.. list-table:: Table of PickAndPlace Intervention Spaces
   :widths: 25 25 40 40
   :header-rows: 1

   * - Variable
     - SubVariable
     - Space A
     - Space B
   * - joint_positions
     -  None
     - [[-1.57, -1.2, -3.0] * 3, |br| [-0.69, 0, 0] * 3]]
     - [[-0.69, 0, 0] * 3, |br| [1.0, 1.57, 3.0] * 3]
   * - tool_block
     - cylindrical_position
     - [[0.07, np.pi/6, h/2], |br| [0.12, (5 / 6.0) * np.pi, h/2]]
     - [[0.12, np.pi / 6, h/2] |br| [0.15, (5 / 6.0) * np.pi, h/2]]
   * - tool_block
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - tool_block
     -  size
     - [[0.055, 0.055, 0.055], |br| [0.075, 0.075, 0.075]]
     - [[0.075, 0.075, 0.075], |br| [0.095, 0.095, 0.095]]
   * - tool_block
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - goal_block
     - cylindrical_position
     - [[0.07, np.pi/6, h/2], |br| [0.12, (5 / 6.0) * np.pi, h/2]]
     - [[0.12, np.pi / 6, h/2] |br| [0.15, (5 / 6.0) * np.pi, h/2]]
   * - goal_block
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - goal_block
     -  size
     - [[0.055, 0.055, 0.055], |br| [0.075, 0.075, 0.075]]
     - [[0.075, 0.075, 0.075], |br| [0.095, 0.095, 0.095]]
   * - goal_block
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - obstacle
     -  size
     - [[0.5, 0.015, 0.02] |br|, [0.5, 0.015, 0.065]]
     - [[0.5, 0.015, 0.065] |br|, [0.5, 0.015, 0.1]]
   * - floor_color
     -  None
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - stage_color
     -  None
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
     - [[0.5, 0.5, 0.5], |br| [1, 1, 1]]
   * - floor_friction
     -  None
     - [0.3, 0.6]
     - [0.6, 0.8]
   * - stage_friction
     -  None
     - [0.3, 0.6]
     - [0.6, 0.8]
   * - robot_finger_60_link_0 |br| or robot_finger_60_link_1 |br| or robot_finger_60_link_2 |br| or robot_finger_60_link_3 |br| or robot_finger_120_link_0 |br| or robot_finger_120_link_1 |br| or robot_finger_120_link_2 |br| or robot_finger_120_link_3 |br| or robot_finger_300_link_0 |br| or robot_finger_300_link_1 |br| or robot_finger_300_link_2 |br| or robot_finger_300_link_3
     -  color
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
     - [[0.5, 0.5, 0.5], |br| [1, 1, 1]]
   * - robot_finger_60_link_0 |br| or robot_finger_60_link_1 |br| or robot_finger_60_link_2 |br| or robot_finger_60_link_3 |br| or robot_finger_120_link_0 |br| or robot_finger_120_link_1 |br| or robot_finger_120_link_2 |br| or robot_finger_120_link_3 |br| or robot_finger_300_link_0 |br| or robot_finger_300_link_1 |br| or robot_finger_300_link_2 |br| or robot_finger_300_link_3
     -  mass
     - [0.015, 0.045]
     - [0.045, 0.1]

-------------------------
Stacking2 Variable Spaces
-------------------------

- Notation: h refers to the height of the block.

.. list-table:: Table of Stacking2 Intervention Spaces
   :widths: 25 25 40 40
   :header-rows: 1

   * - Variable
     - SubVariable
     - Space A
     - Space B
   * - joint_positions
     -  None
     - [[-1.57, -1.2, -3.0] * 3, |br| [-0.69, 0, 0] * 3]]
     - [[-0.69, 0, 0] * 3, |br| [1.0, 1.57, 3.0] * 3]
   * - tool_block_1 |br| or tool_block_2
     - cylindrical_position
     - [[0.0, - math.pi, h/2], |br| [0.11, math.pi, h/2]]
     - [[0.11, - math.pi, h/2] |br| [0.15, math.pi, h/2]]
   * - tool_block_1 |br| or tool_block_2
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - tool_block_1 |br| or tool_block_2
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - goal_block_1 |br|  or goal_block_2
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - goal_tower
     -  cylindrical_position
     - [[0.07, np.pi/6, h/2], |br| [0.12, (5 / 6.0) * np.pi, h/2]]
     - [[0.12, np.pi / 6, h/2] |br| [0.15, (5 / 6.0) * np.pi, h/2]]
   * - goal_tower
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - floor_color
     -  None
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - stage_color
     -  None
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
     - [[0.5, 0.5, 0.5], |br| [1, 1, 1]]
   * - floor_friction
     -  None
     - [0.3, 0.6]
     - [0.6, 0.8]
   * - stage_friction
     -  None
     - [0.3, 0.6]
     - [0.6, 0.8]
   * - robot_finger_60_link_0 |br| or robot_finger_60_link_1 |br| or robot_finger_60_link_2 |br| or robot_finger_60_link_3 |br| or robot_finger_120_link_0 |br| or robot_finger_120_link_1 |br| or robot_finger_120_link_2 |br| or robot_finger_120_link_3 |br| or robot_finger_300_link_0 |br| or robot_finger_300_link_1 |br| or robot_finger_300_link_2 |br| or robot_finger_300_link_3
     -  color
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
     - [[0.5, 0.5, 0.5], |br| [1, 1, 1]]
   * - robot_finger_60_link_0 |br| or robot_finger_60_link_1 |br| or robot_finger_60_link_2 |br| or robot_finger_60_link_3 |br| or robot_finger_120_link_0 |br| or robot_finger_120_link_1 |br| or robot_finger_120_link_2 |br| or robot_finger_120_link_3 |br| or robot_finger_300_link_0 |br| or robot_finger_300_link_1 |br| or robot_finger_300_link_2 |br| or robot_finger_300_link_3
     -  mass
     - [0.015, 0.045]
     - [0.045, 0.1]