import causal_world.evaluation.protocols as protocols

REACHING_BENCHMARK = dict(task_generator_id='reaching',
                          evaluation_protocols=[
                              protocols.GoalPosesTrainSpace(),
                              protocols.GoalPosesTestSpace(),
                              protocols.InitialPosesTrainSpace(),
                              protocols.InitialPosesTestSpace(),
                              protocols.DefaultTask()
                          ])

PUSHING_BENCHMARK = dict(task_generator_id='pushing',
                         evaluation_protocols=[
                             protocols.GoalPosesTrainSpace(),
                             protocols.GoalPosesTestSpace(),
                             protocols.InitialPosesTrainSpace(),
                             protocols.InitialPosesTestSpace(),
                             protocols.ObjectMassesTrainSpace(),
                             protocols.ObjectMassesTestSpace(),
                             protocols.ObjectColorsTrainSpace(),
                             protocols.ObjectColorsTestSpace(),
                             protocols.ObjectSizeTrainSpace(),
                             protocols.ObjectSizeTestSpace(),
                             protocols.FloorFrictionTrainSpace(),
                             protocols.FloorFrictionTestSpace(),
                             protocols.InEpisodePosesChange(),
                             protocols.DefaultTask()
                         ])

PICKING_BENCHMARK = dict(task_generator_id='picking',
                         evaluation_protocols=[
                             protocols.GoalPosesTrainSpace(),
                             protocols.GoalPosesTestSpace(),
                             protocols.InitialPosesTrainSpace(),
                             protocols.InitialPosesTestSpace(),
                             protocols.ObjectMassesTrainSpace(),
                             protocols.ObjectMassesTestSpace(),
                             protocols.ObjectColorsTrainSpace(),
                             protocols.ObjectColorsTestSpace(),
                             protocols.ObjectSizeTrainSpace(),
                             protocols.ObjectSizeTestSpace(),
                             protocols.FloorFrictionTrainSpace(),
                             protocols.FloorFrictionTestSpace(),
                             protocols.InEpisodePosesChange(),
                             protocols.DefaultTask()
                         ])

PICK_AND_PLACE_BENCHMARK = dict(task_generator_id='pick_and_place',
                                evaluation_protocols=[
                                    protocols.GoalPosesTrainSpace(),
                                    protocols.GoalPosesTestSpace(),
                                    protocols.InitialPosesTrainSpace(),
                                    protocols.InitialPosesTestSpace(),
                                    protocols.ObjectMassesTrainSpace(),
                                    protocols.ObjectMassesTestSpace(),
                                    protocols.ObjectColorsTrainSpace(),
                                    protocols.ObjectColorsTestSpace(),
                                    protocols.ObjectSizeTrainSpace(),
                                    protocols.ObjectSizeTestSpace(),
                                    protocols.FloorFrictionTrainSpace(),
                                    protocols.FloorFrictionTestSpace(),
                                    protocols.InEpisodePosesChange(),
                                    protocols.DefaultTask()
                                ])

TOWER_2_BENCHMARK = dict(task_generator_id='towers',
                         evaluation_protocols=[
                             protocols.GoalPosesTrainSpace(),
                             protocols.GoalPosesTestSpace(),
                             protocols.InitialPosesTrainSpace(),
                             protocols.InitialPosesTestSpace(),
                             protocols.ObjectMassesTrainSpace(),
                             protocols.ObjectMassesTestSpace(),
                             protocols.ObjectColorsTrainSpace(),
                             protocols.ObjectColorsTestSpace(),
                             protocols.ObjectSizeTrainSpace(),
                             protocols.ObjectSizeTestSpace(),
                             protocols.FloorFrictionTrainSpace(),
                             protocols.FloorFrictionTestSpace(),
                             protocols.InEpisodePosesChange(),
                             protocols.DefaultTask()
                         ])
