.. _intervention_spaces:

============================
Variables Spaces Definition
============================

**Please note that the spaces below are still preliminary, not refined yet, subject to changing.**

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
     - mass
     - [0.015, 0.045]
     - [0.045, 0.1]
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
     - mass
     - [0.015, 0.045]
     - [0.045, 0.1]
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
     - mass
     - [0.015, 0.045]
     - [0.045, 0.1]
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
     - mass
     - [0.015, 0.045]
     - [0.045, 0.1]
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


-----------------------
Towers Variable Spaces
-----------------------

- Notation: h refers to the height of the block.

.. list-table:: Table of Towers Intervention Spaces
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
   * - tool_level_l_col_c_row_r |br| where l is the level number |br| c is the col number |br| r is the row number
     - cylindrical_position
     - [[0.0, - math.pi, h/2], |br| [0.11, math.pi, 0.15]]
     - [[0.11, - math.pi, 0.15] |br| [0.15, math.pi, 0.3]]
   * - tool_level_l_col_c_row_r |br| where l is the level number |br| c is the col number |br| r is the row number
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - tool_level_l_col_c_row_r |br| where l is the level number |br| c is the col number |br| r is the row number
     - mass
     - [0.015, 0.045]
     - [0.045, 0.1]
   * - tool_level_l_col_c_row_r |br| where l is the level number |br| c is the col number |br| r is the row number
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - number_of_blocks_in_tower
     -  None
     - [[1, 1, 1], |br| [4, 4, 4]]
     - [[4, 4, 4], |br| [6, 6, 6]]
   * - blocks_mass
     -  None
     - [0.02, 0.06]
     - [0.06, 0.08]
   * - tower_dims
     -  None
     - [[0.08, 0.08, 0.08], |br| [0.12, 0.12, 0.12]]
     - [[0.12, 0.12, 0.12], |br| [0.20, 0.20, 0.20]]
   * - tower_center
     -  None
     - [[-0.1, -0.1], |br| [0.05, 0.05]]
     - [[0.05, 0.05], |br| [0.1, 0.1]]
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

-----------------------------
StackedBlocks Variable Spaces
-----------------------------

- Notation: h refers to the height of the block.

.. list-table:: Table of StackedBlocks Intervention Spaces
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
   * - tool_level_l_num_n |br| where l is the level number |br| n is the number of block |br| in the level
     - cylindrical_position
     - [[0.0, - math.pi, h/2], |br| [0.11, math.pi, 0.15]]
     - [[0.11, - math.pi, 0.15] |br| [0.15, math.pi, 0.3]]
   * - tool_level_l_num_n |br| where l is the level number |br| n is the number of block |br| in the level
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - tool_level_l_num_n |br| where l is the level number |br| n is the number of block |br| in the level
     - mass
     - [0.015, 0.045]
     - [0.045, 0.1]
   * - tool_level_l_num_n |br| where l is the level number |br| n is the number of block |br| in the level
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - stack_levels
     -  None
     - [1, 5]
     - [6, 8]
   * - blocks_mass
     -  None
     - [0.02, 0.06]
     - [0.06, 0.08]
   * - blocks_min_size
     -  None
     - [0.035, 0.065]
     - [0.065, 0.075]
   * - max_level_width
     -  None
     - [0.035, 0.12]
     - [0.12, 0.15]
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

-------------------------------------
CreativeStackedBlocks Variable Spaces
-------------------------------------

- Notation: h refers to the height of the block.

.. list-table:: Table of CreativeStackedBlocks Intervention Spaces
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
   * - tool_level_l_num_n |br| where l is the level number |br| n is the number of block |br| in the level
     - cylindrical_position
     - [[0.0, - math.pi, h/2], |br| [0.11, math.pi, 0.15]]
     - [[0.11, - math.pi, 0.15] |br| [0.15, math.pi, 0.3]]
   * - tool_level_l_num_n |br| where l is the level number |br| n is the number of block |br| in the level
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - tool_level_l_num_n |br| where l is the level number |br| n is the number of block |br| in the level
     - mass
     - [0.015, 0.045]
     - [0.045, 0.1]
   * - tool_level_l_num_n |br| where l is the level number |br| n is the number of block |br| in the level
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - stack_levels
     -  None
     - [1, 5]
     - [6, 8]
   * - blocks_mass
     -  None
     - [0.02, 0.06]
     - [0.06, 0.08]
   * - blocks_min_size
     -  None
     - [0.035, 0.065]
     - [0.065, 0.075]
   * - max_level_width
     -  None
     - [0.035, 0.12]
     - [0.12, 0.15]
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
General Variable Spaces
-----------------------

- Notation: h refers to the height of the block.

.. list-table:: Table of General Intervention Spaces
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
   * - tool_n |br| where n is the number of block
     - cylindrical_position
     - [[0.0, - math.pi, h/2], |br| [0.11, math.pi, 0.15]]
     - [[0.11, - math.pi, 0.15] |br| [0.15, math.pi, 0.3]]
   * - tool_n |br| where n is the number of block
     -  euler_orientation
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
     - [[0, 0, -math.pi], |br| [0, 0, math.pi]]
   * - tool_n |br| where n is the number of block
     - mass
     - [0.015, 0.045]
     - [0.045, 0.1]
   * - tool_n |br| where n is the number of block
     -  color
     - [[0.5, 0.5, 0.5] |br|, [1, 1, 1]]
     - [[0, 0, 0], |br| [0.5, 0.5, 0.5]]
   * - nums_objects
     -  None
     - [1, 5]
     - [6, 9]
   * - blocks_mass
     -  None
     - [0.02, 0.06]
     - [0.06, 0.08]
   * - tool_block_size
     -  None
     - [0.05, 0.07]
     - [0.04, 0.05]
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
Variables Without Spaces Yet
----------------------------

- robot_height
- gravity
- velocities in general
- others in a followup version including shape for instance.