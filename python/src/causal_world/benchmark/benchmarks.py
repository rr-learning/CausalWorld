import causal_world.evaluation.protocols as protocols

REACHING_BENCHMARK = dict(task_generator_id='reaching',
                          evaluation_protocols=[
                              protocols.Protocol5(),
                              protocols.GoalPosesSpaceB(),
                              protocols.Protocol4(),
                              protocols.InitialPosesSpaceB(),
                              protocols.Protocol0()
                          ])

PUSHING_BENCHMARK = dict(task_generator_id='pushing',
                         evaluation_protocols=[
                             protocols.Protocol0(),
                             protocols.Protocol1(),
                             protocols.Protocol2(),
                             protocols.Protocol3(),
                             protocols.Protocol4(),
                             protocols.Protocol5(),
                             protocols.Protocol6(),
                             protocols.Protocol7(),
                             protocols.Protocol8(),
                             protocols.Protocol9(),
                             protocols.Protocol10(),
                             protocols.Protocol11()
                         ])

PICKING_BENCHMARK = dict(task_generator_id='picking',
                         evaluation_protocols=[
                             protocols.Protocol0(),
                             protocols.Protocol1(),
                             protocols.Protocol2(),
                             protocols.Protocol3(),
                             protocols.Protocol4(),
                             protocols.Protocol5(),
                             protocols.Protocol6(),
                             protocols.Protocol7(),
                             protocols.Protocol8(),
                             protocols.Protocol9(),
                             protocols.Protocol10(),
                             protocols.Protocol11()
                         ])

PICK_AND_PLACE_BENCHMARK = dict(task_generator_id='pick_and_place',
                                evaluation_protocols=[
                                    protocols.Protocol0(),
                                    protocols.Protocol1(),
                                    protocols.Protocol2(),
                                    protocols.Protocol3(),
                                    protocols.Protocol4(),
                                    protocols.Protocol5(),
                                    protocols.Protocol6(),
                                    protocols.Protocol7(),
                                    protocols.Protocol8(),
                                    protocols.Protocol9(),
                                    protocols.Protocol10(),
                                    protocols.Protocol11()
                                ])

STACKING_TWO_BENCHMARK = dict(task_generator_id='stacking2',
                              evaluation_protocols=[
                                  protocols.Protocol0(),
                                  protocols.Protocol1(),
                                  protocols.Protocol2(),
                                  protocols.Protocol3(),
                                  protocols.Protocol4(),
                                  protocols.Protocol5(),
                                  protocols.Protocol6(),
                                  protocols.Protocol7(),
                                  protocols.Protocol8(),
                                  protocols.Protocol9(),
                                  protocols.Protocol10(),
                                  protocols.Protocol11()
                              ])

TOWER_2_BENCHMARK = dict(task_generator_id='towers',
                         evaluation_protocols=[
                             protocols.Protocol0(),
                             protocols.Protocol1(),
                             protocols.Protocol2(),
                             protocols.Protocol3(),
                             protocols.Protocol4(),
                             protocols.Protocol5(),
                             protocols.Protocol6(),
                             protocols.Protocol7(),
                             protocols.Protocol8(),
                             protocols.Protocol9(),
                             protocols.Protocol10(),
                             protocols.Protocol11()
                         ])
