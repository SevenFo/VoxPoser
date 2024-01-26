import numpy as np

from pyrep.objects import Object
from pyrep.backend import sim, utils
from pyrep.const import ObjectType


class Quadcopter(Object):
    def __init__(
        self,
        quadcopter_object_name="Quadricopter",
        quadcopter_target_object_name="Quadricopter_target",
    ):
        super().__init__(quadcopter_object_name)
        self.target_object = Object.get_object(quadcopter_target_object_name)

    def get_distance_between(self, position1, position2):
        return np.linalg.norm(np.array(position1) - np.array(position2))

    def _goto(self, position):
        self.target_object.set_position(position)
        print("set target position: ", position)

    def goto(self, position, ignor_distance=False):
        # the target position must less than 0.2m away from the current position, otherwise the quadcopter maybe unstable
        # if the target position is too far away, will decompose the target position into several small steps
        # and set the first step as the target position and return the rest steps
        current_position = self.get_position()
        distance = self.get_distance_between(current_position, position)
        if distance > 0.2 and not ignor_distance:
            # decompose the target position into several small steps
            step_count = int(distance / 0.2)
            step_vector = (np.array(position) - np.array(current_position)) / step_count
            # set the first step as the target position
            self.target_object.set_position(current_position + step_vector)
            # return the rest steps
            print("set target position: ", current_position + step_vector)
            return [current_position + step_vector * i for i in range(1, step_count)]
        else:
            # if the target position is less than 0.2m away from the current position, just set the target position
            self._goto(position)

    def action_is_reached(self, position, tolerance=0.05):
        # print("distance: ", self.get_distance_between(self.get_position(), position))
        return self.get_distance_between(self.get_position(), position) < tolerance

    def action_is_failed(self, position, tolerance=10):
        return self.get_distance_between(self.get_position(), position) > tolerance

    def _get_requested_type(self):
        return ObjectType(sim.simGetObjectType(self.get_handle()))
