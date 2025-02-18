# code reuse partial of https://github.com/harshilpatel312/KITTI-distance-estimation
"""This file contains the NN-based distance predictor.

Here, you will design the NN module for distance prediction
"""
from mp2_distance_predictor.inference_distance import infer_dist
from mp2_distance_predictor.detect import detect_cars

from pathlib import Path
from keras.models import load_model
from keras.models import model_from_json


# NOTE: Very important that the class name remains the same
class Predictor:
    def __init__(self):
        self.detect_model = None
        self.distance_model = None

    def initialize(self):

        self.detect_model = load_model('/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/yolo_model.h5')
        self.distance_model = self.load_inference_model()

    def load_inference_model(self):
        MODEL = 'distance_model'
        WEIGHTS = 'distance_model'
    
        # load json and create model
        json_file = open('/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{}.json'.format(MODEL), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights("/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{}.h5".format(WEIGHTS))
        print("Loaded model from disk")

        # evaluate loaded model on test data
        loaded_model.compile(loss='mean_squared_error', optimizer='adam')
        return loaded_model


    def predict(self, obs) -> float:
        """
        Predicting based on the NNtrained. 
        """
        # Extract distance from observation
        dist_test = obs.distance_to_lead
        #print(f"Received observation with distance_to_lead: {dist_test}")
    
        # Save the observation image to a temporary file
        image_name = 'camera_images/vision_input.png'
        if hasattr(obs, "image") and obs.image is not None:
            obs.image.save(image_name)
        else:
            #print("No image found in observation. Returning fallback distance.")
            return dist_test  # Fallback to observation's distance if no image is present
    
        # Detecting cars in the image and then estimating distance
        car_bounding_box = detect_cars(self.detect_model, image_name)  # Pre-trained YOLO
        if car_bounding_box is not None:
            # Use bounding box to estimate distance
            dist = infer_dist(self.distance_model, car_bounding_box, [[dist_test]])
        else:
            print("No car detected. Using fallback distance.")
            dist = dist_test  # Use observation's distance as a fallback
    
        print("Estimated distance: ", dist)
        return dist
    
    
