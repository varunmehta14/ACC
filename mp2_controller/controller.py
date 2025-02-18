# """
# This file is the main controller file

# Here, you will design the controller for your for the adaptive cruise control system.
# """

# from mp2_simulator.simulator import Observation
# import logging


# # NOTE: Very important that the class name remains the same
# class Controller:
#     def __init__(self, target_speed: float, distance_threshold: float):
#         self.target_speed = target_speed
#         self.distance_threshold = distance_threshold

#     def run_step(self, obs: Observation, estimate_dist) -> float:
#         """This is the main run step of the controller.

#         Here, you will have to read in the observations `obs`, process it, and output an
#         acceleration value. The acceleration value must be some value between -10.0 and 10.0.

#         Note that the acceleration value is really some control input that is used
#         internally to compute the throttle and brake values for the car.

#         Below is some example code where the car just outputs the control value 10.0
#         """

#         ego_velocity = obs.ego_velocity
#         desired_speed = obs.desired_speed
#         dist_to_lead = estimate_dist
#         # Do your magic...

#         return 10.0
    


# """This file is the main controller file.

# Here, you will design the controller for your for the adaptive cruise control system.
# """

# from mp2_simulator.simulator import Observation
# # NOTE: Very important that the class name remains the same
# class Controller:
#     def __init__(self, target_speed, distance_threshold):
#         self.target_speed = target_speed
#         self.distance_threshold = distance_threshold
#         self.distance_safe = 1.97 * max((self.target_speed ** 2) / 20, self.distance_threshold)

#     def run_step(self, obs: Observation, estimate_dist: float = None) -> float:
#         """This is the main run step of the controller.
#         """
#         ego_velocity = obs.ego_velocity
#         desired_speed = obs.desired_speed
#         dist_to_lead = estimate_dist if estimate_dist is not None else obs.distance_to_lead
    
#         # Do your magic...
#         if dist_to_lead <= self.distance_safe:
#             if ego_velocity < 10 and dist_to_lead > self.distance_safe * 0.48:
#                 acc = 10  # Max acceleration
#             else:
#                 acc = -10 * ego_velocity  # Decelerate based on current velocity
#         else:
#             if ego_velocity == desired_speed:
#                 acc = 0  # Maintain current speed
#             else:
#                 acc = 10 * (desired_speed - ego_velocity)  # Accelerate towards desired speed
    
#         # Clip the acceleration to be within the range -10 to 10
#         acc = max(-10, min(10, acc))
    
#         return acc


"""This file is the main controller file.

Here, you will design the controller for your for the adaptive cruise control system.
"""

from mp2_simulator.simulator import Observation
# NOTE: Very important that the class name remains the same
class Controller:
    def __init__(self, target_speed, distance_threshold):
        self.target_speed = target_speed
        self.distance_threshold = distance_threshold
        self.distance_safe = 2 * max((self.target_speed ** 2) / 20, self.distance_threshold)

    def run_step(self, obs: Observation, estimate_dist: float = None) -> float:
        """This is the main run step of the controller.
        """
        ego_velocity = obs.ego_velocity
        desired_speed = obs.desired_speed
        dist_to_lead = estimate_dist if estimate_dist is not None else obs.distance_to_lead
    
        # Do your magic...
        if dist_to_lead <= self.distance_safe:
            if ego_velocity < 10 and dist_to_lead > self.distance_safe * 0.5:
                acc = 10  # Max acceleration
            else:
                acc = -10 * ego_velocity  # Decelerate based on current velocity
        else:
            if ego_velocity == desired_speed:
                acc = 0  # Maintain current speed
            else:
                acc = 10 * (desired_speed - ego_velocity)  # Accelerate towards desired speed
    
        # Clip the acceleration to be within the range -10 to 10
        acc = max(-10, min(10, acc))
    
        return acc
