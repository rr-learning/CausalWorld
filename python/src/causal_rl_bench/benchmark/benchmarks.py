import causal_rl_bench.evaluation.protocols as protocols

PUSHING_BENCHMARK = dict(task_name='pushing',
                         evaluation_protocols=[protocols.GoalPosesOOD(),
                                               protocols.InitialPosesOOD(),
                                               protocols.SameMassesOOD(),
                                               protocols.SameColorsOOD(),
                                               protocols.ObjectSizesOOD(),
                                               protocols.FloorFrictionOOD(),
                                               protocols.RandomInTrainSet(),
                                               protocols.InEpisodePosesChange(),
                                               protocols.DefaultTask()])

REACHING_BENCHMARK = dict()
