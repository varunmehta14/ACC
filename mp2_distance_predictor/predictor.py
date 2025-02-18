# # # code reuse partial of https://github.com/harshilpatel312/KITTI-distance-estimation
# # """This file contains the NN-based distance predictor.

# # Here, you will design the NN module for distance prediction
# # """
# # from mp2_distance_predictor.inference_distance import infer_dist
# # from mp2_distance_predictor.detect import detect_cars

# # from pathlib import Path
# # from keras.models import load_model
# # from keras.models import model_from_json


# # # NOTE: Very important that the class name remains the same
# # class Predictor:
# #     def __init__(self):
# #         self.detect_model = None
# #         self.distance_model = None

# #     def initialize(self):

# #         self.detect_model = load_model('mp2_distance_predictor/yolo_model.h5')
# #         self.distance_model = self.load_inference_model()

# #     def load_inference_model(self):
# #         MODEL = 'distance_model'
# #         WEIGHTS = 'distance_model'

# #         # load json and create model
# #         json_file = open('mp2_distance_predictor/distance_model_weights/{}.json'.format(MODEL), 'r')
# #         loaded_model_json = json_file.read()
# #         json_file.close()
# #         loaded_model = model_from_json(loaded_model_json)

# #         # load weights into new model
# #         loaded_model.load_weights("mp2_distance_predictor/distance_model_weights/{}.h5".format(WEIGHTS))
# #         print("Loaded model from disk")

# #         # evaluate loaded model on test data
# #         loaded_model.compile(loss='mean_squared_error', optimizer='adam')
# #         return loaded_model


# #     def predict(self, obs) -> float:
# #         """This is the main predict step of the NN.

# #         Here, the provided observation is an Image. Your goal is to train a NN that can
# #         use this image to predict distance to the lead car.

# #         """
# #         data = obs
# #         image_name = 'camera_images/vision_input.png'

# #         # load a trained yolov3 model
# #         car_bounding_box = detect_cars(self.detect_model, image_name)  # return the bounding box of car
# #         dist_test = data.distance_to_lead
# #         # Different dist_test will have effect on the prediction
# #         # You can play with the number of dist_test
# #         if car_bounding_box is not None:
# #             dist = infer_dist(self.distance_model, car_bounding_box, [[dist_test]])
# #         else:
# #             print("No car detected")
# #             # If no car detected what would you do for the distance prediction
# #             # Do your magic...


# #             dist = 30

# #         print("estimated distance: ", dist)
# #         return dist



# ######### My codeeee

# from mp2_distance_predictor.inference_distance import infer_dist
# from mp2_distance_predictor.detect import detect_cars
# from pathlib import Path
# from keras.models import load_model, model_from_json

# # NOTE: Very important that the class name remains the same
# class Predictor:
#     def __init__(self):
#         self.detect_model = None
#         self.distance_model = None

#     def initialize(self):
#         """Initialize the models for detection and distance prediction."""
#         try:
#             self.detect_model = load_model('mp2_distance_predictor/yolo_model.h5')
#             self.distance_model = self.load_inference_model()
#             print("Models loaded successfully.")
#         except Exception as e:
#             print(f"Error during initialization: {e}")

#     def load_inference_model(self):
#         """Load the inference model for distance prediction from disk."""
#         try:
#             model_path = 'mp2_distance_predictor/distance_model_weights'
#             with open(f'{model_path}/distance_model.json', 'r') as json_file:
#                 loaded_model_json = json_file.read()

#             loaded_model = model_from_json(loaded_model_json)
#             loaded_model.load_weights(f"{model_path}/distance_model.h5")
#             loaded_model.compile(loss='mean_squared_error', optimizer='adam')
#             print("Loaded distance model from disk.")

#             return loaded_model
#         except Exception as e:
#             print(f"Error loading inference model: {e}")
#             return None

#     def predict(self, obs) -> float:
#         """Predict the distance to the lead car based on the given observation."""
#         try:
#             image_name = 'camera_images/vision_input.png'
#             car_bounding_box = detect_cars(self.detect_model, image_name)  # Detect car bounding box

#             dist_test = obs.distance_to_lead  # Placeholder for additional input features

#             if car_bounding_box is not None:
#                 dist = infer_dist(self.distance_model, car_bounding_box, [[dist_test]])
#             else:
#                 print("No car detected. Using default distance estimate.")
#                 dist = self.handle_no_detection(obs)

#             print("Estimated distance:", dist)
#             return dist

#         except Exception as e:
#             print(f"Error during prediction: {e}")
#             return -1  # Return a sentinel value indicating failure

#     def handle_no_detection(self, obs):
#         """Handle cases where no car is detected."""
#         # Implement a fallback mechanism, e.g., default distance, prior estimates, etc.
#         # Example: Default distance based on input observation
#         return max(obs.distance_to_lead, 30)  # Ensure a reasonable default value


import numpy as np
from mp2_distance_predictor.inference_distance import infer_dist
from mp2_distance_predictor.detect import detect_cars
from keras.models import load_model, model_from_json

class Predictor:
    def __init__(self):
        self.detect_model = None
        self.distance_model = None
        self.previous_distances = []
        self.max_history = 5

    def initialize(self):
        try:
            self.detect_model = load_model('mp2_distance_predictor/yolo_model.h5')
            self.distance_model = self.load_inference_model()
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

    def load_inference_model(self):
        try:
            model_path = 'mp2_distance_predictor/distance_model_weights'
            with open(f'{model_path}/distance_model.json', 'r') as json_file:
                loaded_model_json = json_file.read()

            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(f"{model_path}/distance_model.h5")
            loaded_model.compile(loss='mean_squared_error', optimizer='adam')
            print("Loaded distance model from disk.")
            return loaded_model
        except Exception as e:
            print(f"Error loading inference model: {e}")
            raise

    def predict(self, obs) -> float:
        try:
            image_name = 'camera_images/vision_input.png'
            car_bounding_box = detect_cars(self.detect_model, image_name)

            dist_test = obs.distance_to_lead

            if car_bounding_box is not None:
                dist = infer_dist(self.distance_model, car_bounding_box, [[dist_test]])
                self.update_distance_history(dist)
            else:
                print("No car detected. Using fallback estimation.")
                dist = self.handle_no_detection(obs)

            print(f"Estimated distance: {dist:.2f}")
            return dist

        except Exception as e:
            print(f"Error during prediction: {e}")
            return self.handle_prediction_error(obs)

    def handle_no_detection(self, obs):
        if self.previous_distances:
            # Use exponential moving average of previous distances
            weights = np.exp(np.linspace(-1, 0, len(self.previous_distances)))
            weighted_avg = np.average(self.previous_distances, weights=weights)
            return max(weighted_avg, obs.distance_to_lead, 30)
        else:
            # If no history, use a combination of observation and default value
            return max(obs.distance_to_lead * 1.1, 30)  # Add 10% margin to observed distance

    def update_distance_history(self, dist):
        self.previous_distances.append(dist)
        if len(self.previous_distances) > self.max_history:
            self.previous_distances.pop(0)

    def handle_prediction_error(self, obs):
        print("Prediction error occurred. Using last known distance or observation.")
        if self.previous_distances:
            return self.previous_distances[-1]
        else:
            return max(obs.distance_to_lead, 30)
