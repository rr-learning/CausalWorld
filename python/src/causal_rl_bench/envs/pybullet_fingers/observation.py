#!/usr/bin/env python3


class Observation:
    """
    The observation structure
    """

    def __init__(self):
        position = []
        velocity = []
        torque = []
        camera_60 = []
        camera_180 = []
        camera_300 = []

        self.position = position
        self.velocity = velocity
        self.torque = torque
        self.camera_60 = camera_60
        self.camera_180 = camera_180
        self.camera_300 = camera_300
