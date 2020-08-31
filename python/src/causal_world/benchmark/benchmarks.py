import causal_world.evaluation.protocols as protocols

REACHING_BENCHMARK = dict(task_generator_id='reaching',
                          evaluation_protocols=[
                              protocols.ProtocolGenerator(name=
                                                          'goal_poses_space_a',
                                                          first_level_regex=
                                                          'goal_.*',
                                                          second_level_regex=
                                                          'cylindrical_position',
                                                          variable_space='a'),
                              protocols.ProtocolGenerator(name=
                                                          'goal_poses_space_b',
                                                          first_level_regex=
                                                          'goal_.*',
                                                          second_level_regex=
                                                          'cylindrical_position',
                                                          variable_space='b'),
                              protocols.ProtocolGenerator(name=
                                                          'default_task',
                                                          first_level_regex=
                                                          '',
                                                          second_level_regex=
                                                          '',
                                                          variable_space='a_b')
                          ])

PUSHING_BENCHMARK = dict(task_generator_id='pushing',
                         evaluation_protocols=[
                             protocols.GoalPosesSpaceA(),
                             protocols.GoalPosesSpaceB(),
                             protocols.InitialPosesSpaceA(),
                             protocols.InitialPosesSpaceB(),
                             protocols.ObjectMassesSpaceA(),
                             protocols.ObjectMassesSpaceB(),
                             protocols.ObjectSizeSpaceA(),
                             protocols.ObjectSizeSpaceB(),
                             protocols.FloorFrictionSpaceA(),
                             protocols.FloorFrictionSpaceB(),
                             protocols.InEpisodePosesChangeSpaceA(),
                             protocols.DefaultTask()
                         ])

PICKING_BENCHMARK = dict(task_generator_id='picking',
                         evaluation_protocols=[
                             protocols.GoalPosesSpaceA(),
                             protocols.GoalPosesSpaceB(),
                             protocols.InitialPosesSpaceA(),
                             protocols.InitialPosesSpaceB(),
                             protocols.ObjectMassesSpaceA(),
                             protocols.ObjectMassesSpaceB(),
                             protocols.ObjectColorsSpaceA(),
                             protocols.ObjectColorsSpaceB(),
                             protocols.ObjectSizeSpaceA(),
                             protocols.ObjectSizeSpaceB(),
                             protocols.FloorFrictionSpaceA(),
                             protocols.FloorFrictionSpaceB(),
                             protocols.InEpisodePosesChangeSpaceA(),
                             protocols.DefaultTask()
                         ])

PICK_AND_PLACE_BENCHMARK = dict(task_generator_id='pick_and_place',
                                evaluation_protocols=[
                                    protocols.GoalPosesSpaceA(),
                                    protocols.GoalPosesSpaceB(),
                                    protocols.InitialPosesSpaceA(),
                                    protocols.InitialPosesSpaceB(),
                                    protocols.ObjectMassesSpaceA(),
                                    protocols.ObjectMassesSpaceB(),
                                    protocols.ObjectColorsSpaceA(),
                                    protocols.ObjectColorsSpaceB(),
                                    protocols.ObjectSizeSpaceA(),
                                    protocols.ObjectSizeSpaceB(),
                                    protocols.FloorFrictionSpaceA(),
                                    protocols.FloorFrictionSpaceB(),
                                    protocols.InEpisodePosesChangeSpaceA(),
                                    protocols.DefaultTask()
                                ])

TOWER_2_BENCHMARK = dict(task_generator_id='towers',
                         evaluation_protocols=[
                             protocols.GoalPosesSpaceA(),
                             protocols.GoalPosesSpaceB(),
                             protocols.InitialPosesSpaceA(),
                             protocols.InitialPosesSpaceB(),
                             protocols.ObjectMassesSpaceA(),
                             protocols.ObjectMassesSpaceB(),
                             protocols.ObjectColorsSpaceA(),
                             protocols.ObjectColorsSpaceB(),
                             protocols.ObjectSizeSpaceA(),
                             protocols.ObjectSizeSpaceB(),
                             protocols.FloorFrictionSpaceA(),
                             protocols.FloorFrictionSpaceB(),
                             protocols.InEpisodePosesChangeSpaceA(),
                             protocols.DefaultTask()
                         ])
