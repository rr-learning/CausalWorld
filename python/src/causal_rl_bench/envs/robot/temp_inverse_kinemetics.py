from time import sleep
import pybullet as p
import numpy as np
import math


def getJointRanges(pybullet_client, bodyId, includeFixed=False):
    """
    Parameters
    ----------
    bodyId : int
    includeFixed : bool
    Returns
    -------
    lowerLimits : [ float ] * numDofs
    upperLimits : [ float ] * numDofs
    jointRanges : [ float ] * numDofs
    restPoses : [ float ] * numDofs
    """

    lowerLimits, upperLimits, jointRanges, restPoses = [], [], [], []

    numJoints = pybullet_client.getNumJoints(bodyId)

    for i in range(numJoints):
        jointInfo = pybullet_client.getJointInfo(bodyId, i)

        if includeFixed or jointInfo[3] > -1:
            ll, ul = jointInfo[8:10]
            jr = ul - ll

            # For simplicity, assume resting state == initial state
            rp = pybullet_client.getJointState(bodyId, i)[0]

            lowerLimits.append(-2)
            upperLimits.append(2)
            jointRanges.append(2)
            restPoses.append(rp)

    return lowerLimits, upperLimits, jointRanges, restPoses


def accurateIK(pybullet_client, bodyId, endEffectorId, targetPosition, lowerLimits, upperLimits, jointRanges, restPoses,
               useNullSpace=False, maxIter=10, threshold=1e-4):
    """
    Parameters
    ----------
    bodyId : int
    endEffectorId : int
    targetPosition : [float, float, float]
    lowerLimits : [float]
    upperLimits : [float]
    jointRanges : [float]
    restPoses : [float]
    useNullSpace : bool
    maxIter : int
    threshold : float
    Returns
    -------
    jointPoses : [float] * numDofs
    """
    if useNullSpace:
        jointPoses = pybullet_client.calculateInverseKinematics(bodyId, endEffectorId, targetPosition,
                                                                lowerLimits=lowerLimits, upperLimits=upperLimits,
                                                                jointRanges=jointRanges,
                                                                restPoses=restPoses)
    else:
        jointPoses = pybullet_client.calculateInverseKinematics(bodyId,
                                                                endEffectorId,
                                                                targetPosition,)
    return jointPoses


# def setMotors(pybullet_client, bodyId, jointPoses):
#     """
#     Parameters
#     ----------
#     bodyId : int
#     jointPoses : [float] * numDofs
#     """
#     numJoints = pybullet_client.getNumJoints(bodyId)
#
#     for i in range(numJoints):
#         jointInfo = p.getJointInfo(bodyId, i)
#         # print(jointInfo)
#         qIndex = jointInfo[3]
#         if qIndex > -1:
#             pybullet_client.setJointMotorControl2(bodyIndex=bodyId, jointIndex=i, controlMode=pybullet_client.POSITION_CONTROL,
#                                                   targetPosition=jointPoses[qIndex - 7])


def apply_inverse_kinemetics(pybullet_clint, trifinger_id, finger_tip_ids, target_position,
                             rest_pose):
    lowerLimits, upperLimits, jointRanges, restPoses = getJointRanges(pybullet_clint,
                                                                      trifinger_id,
                                                                      includeFixed=False)
    useNullSpace = True
    # for i in range(3):
    # lower limits for null space
    ll = [-math.radians(70), -math.radians(70), -math.radians(160)] * 3
    ul = [math.radians(70), 0, math.radians(-2)] * 3
    # ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    # # upper limits for null space
    # ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    # joint ranges for null space
    jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]

    # jr = [2, 2, 2, 2, 2, 2, 2, 2, 2]

    # restposes for null space
    rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    rp = rest_pose
    # joint damping coefficents
    jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    final_joint_pose = []
    for i in range(3):
        jointPoses = accurateIK(pybullet_clint, trifinger_id,
                                finger_tip_ids[i],
                                target_position[i*3:(i+1)*3],
                                ll,
                                ul,
                                jr,
                                rp,
                                useNullSpace=useNullSpace)
        final_joint_pose.extend(jointPoses[i*3:(i+1)*3])
        # setMotors(baxterId, jointPoses)

        # sleep(0.1)
    return final_joint_pose