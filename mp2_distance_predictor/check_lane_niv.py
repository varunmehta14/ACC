# import os
# import pandas as pd
# import numpy as np
# from keras.models import Sequential, load_model
# from keras.layers import Dense, Dropout
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from keras.callbacks import ModelCheckpoint
# from keras.layers import Dense, Dropout, BatchNormalization
# from mp2_distance_predictor.detect import detect_cars
# from keras.models import Sequential, model_from_json
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# class LeadCarLanePredictor:
#     def __init__(self):
#         self.bb_detect_model = load_model("/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/yolo_model.h5")
#         self.carlane_model = None

#     def create_dataset(self):
#         """
#         Create a dataset of bounding boxes and labels for lane detection.
#         Saves progress to a temporary file and generates train/test datasets at the end.
#         Skips this step if the dataset already exists.
#         """
#         temp_file = "/home1/varunjay/csci513-miniproject2/data/lane_temp.csv"
#         train_file = "/home1/varunjay/csci513-miniproject2/data/lane_train.csv"
#         test_file = "/home1/varunjay/csci513-miniproject2/data/lane_test.csv"

#         if os.path.exists(train_file) and os.path.exists(test_file):
#             print(f"Datasets already exist: {train_file} and {test_file}. Skipping dataset creation.")
#             return

#         lane_data_path = "/home1/varunjay/csci513-miniproject2/lane_data"
#         data = []
#         total_images = 0

#         for folder in ["ado_in_lane", "ado_not_in_lane"]:
#             folder_path = os.path.join(lane_data_path, folder)
#             if os.path.exists(folder_path):
#                 total_images += len(os.listdir(folder_path))
#             else:
#                 print(f"Folder not found: {folder_path}")

#         print(f"Total images to process: {total_images}")

#         for label, folder in enumerate(["ado_in_lane", "ado_not_in_lane"]):
#             folder_path = os.path.join(lane_data_path, folder)
#             if not os.path.exists(folder_path):
#                 continue

#             for img_name in os.listdir(folder_path):
#                 img_path = os.path.join(folder_path, img_name)
#                 unique_name = f"{folder}/{img_name}"
#                 print(f"Processing image: {img_path}")

#                 try:
#                     car_bounding_box = detect_cars(self.bb_detect_model, img_path)
#                     if car_bounding_box is not None and len(car_bounding_box) > 0:
#                         if isinstance(car_bounding_box[0], list):
#                             car_bounding_box = car_bounding_box[0]

#                         xloc = (car_bounding_box[0] + car_bounding_box[2]) / 2
#                         yloc = (car_bounding_box[1] + car_bounding_box[3]) / 2
#                         zloc = 0

#                         record = {
#                             "filename": unique_name,
#                             "xmin": car_bounding_box[0],
#                             "ymin": car_bounding_box[1],
#                             "xmax": car_bounding_box[2],
#                             "ymax": car_bounding_box[3],
#                             "xloc": xloc,
#                             "yloc": yloc,
#                             "zloc": zloc,
#                             "label": label,
#                         }
#                         data.append(record)
#                 except Exception as e:
#                     print(f"Error processing image {img_path}: {e}")

#         df = pd.DataFrame(data)
#         df = df.sample(frac=1, random_state=42).reset_index(drop=True)

#         train_df = df.sample(frac=0.8, random_state=42)
#         test_df = df.drop(train_df.index)

#         train_df.to_csv(train_file, index=False)
#         test_df.to_csv(test_file, index=False)
#         print(f"Dataset creation complete. Saved to `{train_file}` and `{test_file}`.")

#     def check_lane_model(self):
#         """
#         Train a neural network model for lane detection using `lane_train.csv`.
#         """
#         train_file = "/home1/varunjay/csci513-miniproject2/data/lane_train.csv"
#         model_name = "car_lane_model"
    
#         if not os.path.exists(train_file):
#             print("Training dataset not found. Please create the dataset first.")
#             return
    
#         # Load the training data
#         df = pd.read_csv(train_file)
#         X = df[["xmin", "ymin", "xmax", "ymax"]].values
#         y = df["label"].values
    
#         # Creating the model with more layers and neurons
#         model = Sequential([
#             Dense(128, activation="relu", input_dim=4),
#             Dropout(0.5),
#             Dense(64, activation="relu"),
#             Dropout(0.5),
#             Dense(32, activation="relu"),
#             Dropout(0.5),
#             Dense(1, activation="sigmoid")
#         ])
    
#         model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
#         # Training the model with increased epochs
#         checkpoint = ModelCheckpoint(f"mp2_distance_predictor/carlane_model_weights/{model_name}.h5", save_best_only=True)
#         model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2, callbacks=[checkpoint])
    
#         # Save the model
#         model_json = model.to_json()
#         with open(f"mp2_distance_predictor/carlane_model_weights/{model_name}.json", "w") as json_file:
#             json_file.write(model_json)
    
#         print("Saved car_lane model to disk.")
#         self.carlane_model = model

#     def load_trained_model(self):
#             """
#             Load the trained car lane model from disk.
#             """
#             model_name = "car_lane_model"
#             json_file = open(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{model_name}.json", "r")
#             loaded_model_json = json_file.read()
#             json_file.close()
    
#             model = Sequential()
#             model = model_from_json(loaded_model_json)
#             model.load_weights(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{model_name}.h5")
#             print("Loaded car_lane model from disk.")
    
#             self.carlane_model = model
    

#     def check_lane_model_test(self):
#             """
#             Evaluate the trained model on the test dataset (`lane_test.csv`).
#             """
#             test_file = "/home1/varunjay/csci513-miniproject2/data/lane_test.csv"
#             model_name = "car_lane_model"
    
#             if not os.path.exists(test_file):
#                 print("Test dataset not found. Please create the dataset first.")
#                 return
    
#             if not self.carlane_model:
#                 self.load_trained_model()
    
#             df = pd.read_csv(test_file)
#             X = df[["xmin", "ymin", "xmax", "ymax"]].values
#             y_true = df["label"].values
    
#             y_pred = (self.carlane_model.predict(X) > 0.5).astype(int)
#             accuracy = accuracy_score(y_true, y_pred)
    
#             print(f"Model accuracy on test data: {accuracy * 100:.2f}%")
#     def predict(self, img):
#         """
#         Predict if the car in the image is in the same lane using the trained model.
#         """
#         if not self.carlane_model:
#             self.load_trained_model()
    
#         # Debug: Log the input image details
#         print(f"Received input for prediction. Type: {type(img)}, Shape: {getattr(img, 'shape', 'Unknown')}")
    
#         try:
#             # Handle numpy array input by saving it as an image file
#             if isinstance(img, np.ndarray):
#                 temp_dir = "/home1/varunjay/csci513-miniproject2/tmp"
#                 temp_image_path = os.path.join(temp_dir, "temp_image.png")
                
#                 # Ensure the temp directory exists
#                 os.makedirs(temp_dir, exist_ok=True)
    
#                 # Save the numpy array as an image
#                 plt.imsave(temp_image_path, img)
    
#                 # Update img to be the path to the temporary file
#                 img = temp_image_path
    
#             # Use the image path to detect cars
#             car_bounding_box = detect_cars(self.bb_detect_model, img)
#             if car_bounding_box is None or len(car_bounding_box) == 0:
#                 print("No bounding box detected.")
#                 return "Not in lane"
    
#             # Debug: Log bounding box details
#             print(f"Detected bounding box: {car_bounding_box}")
    
#             if isinstance(car_bounding_box[0], list):
#                 car_bounding_box = car_bounding_box[0]
    
#             X = np.array([[car_bounding_box[0], car_bounding_box[1], car_bounding_box[2], car_bounding_box[3]]])
    
#             # Debug: Log the input to the car lane model
#             print(f"Input to car lane model: {X}")
    
#             prediction = self.carlane_model.predict(X)
#             print(f"Prediction result: {prediction}")
    
#             return "In lane" if prediction > 0.5 else "Not in lane"
    
#         except Exception as e:
#             print(f"Error during prediction: {e}")
#             return "Error"
    
    
    
    
# from PIL import Image
# import numpy as np

# def main():
#     # Correctly initialize the LeadCarLanePredictor object
#     lead_carlane_predictor = LeadCarLanePredictor()

#     # Create a dataset using the images provided
#     lead_carlane_predictor.create_dataset()

#     # Train the model to predict if the car is in the same lane or not
#     lead_carlane_predictor.check_lane_model()

#     # Test the model using the `lane_test.csv`
#     lead_carlane_predictor.check_lane_model_test()

#     #img_path = "/home1/varunjay/csci513-miniproject2/lane_data/ado_in_lane/camera_image_1.png"
#     #result = lead_carlane_predictor.predict(img_path)
#     #print(f"Prediction for {img_path}: {result}")
    

# if __name__ == "__main__":
#     main()


import os
import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, BatchNormalization
from mp2_distance_predictor.detect import detect_cars
from keras.models import Sequential, model_from_json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
class LeadCarLanePredictor:
    def __init__(self):
        self.bb_detect_model = load_model("/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/yolo_model.h5")
        self.carlane_model = None

    def create_dataset(self):
        """
        Create a dataset of bounding boxes and labels for lane detection.
        Saves progress to a temporary file and generates train/test datasets at the end.
        Skips this step if the dataset already exists.
        """
        temp_file = "/home1/varunjay/csci513-miniproject2/data/lane_temp.csv"
        train_file = "/home1/varunjay/csci513-miniproject2/data/lane_train.csv"
        test_file = "/home1/varunjay/csci513-miniproject2/data/lane_test.csv"

        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f"Datasets already exist: {train_file} and {test_file}. Skipping dataset creation.")
            return

        lane_data_path = "/home1/varunjay/csci513-miniproject2/lane_data"
        data = []
        total_images = 0

        for folder in ["ado_in_lane", "ado_not_in_lane"]:
            folder_path = os.path.join(lane_data_path, folder)
            if os.path.exists(folder_path):
                total_images += len(os.listdir(folder_path))
            else:
                print(f"Folder not found: {folder_path}")

        print(f"Total images to process: {total_images}")

        for label, folder in enumerate(["ado_in_lane", "ado_not_in_lane"]):
            folder_path = os.path.join(lane_data_path, folder)
            if not os.path.exists(folder_path):
                continue

            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                unique_name = f"{folder}/{img_name}"
                print(f"Processing image: {img_path}")

                try:
                    car_bounding_box = detect_cars(self.bb_detect_model, img_path)
                    if car_bounding_box is not None and len(car_bounding_box) > 0:
                        if isinstance(car_bounding_box[0], list):
                            car_bounding_box = car_bounding_box[0]

                        xloc = (car_bounding_box[0] + car_bounding_box[2]) / 2
                        yloc = (car_bounding_box[1] + car_bounding_box[3]) / 2
                        zloc = 0

                        record = {
                            "filename": unique_name,
                            "xmin": car_bounding_box[0],
                            "ymin": car_bounding_box[1],
                            "xmax": car_bounding_box[2],
                            "ymax": car_bounding_box[3],
                            "xloc": xloc,
                            "yloc": yloc,
                            "zloc": zloc,
                            "label": label,
                        }
                        data.append(record)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

        df = pd.DataFrame(data)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        print(f"Dataset creation complete. Saved to `{train_file}` and `{test_file}`.")

    def check_lane_model(self):
        """
        Train a neural network model for lane detection using `lane_train.csv`.
        """
        train_file = "/home1/varunjay/csci513-miniproject2/data/lane_train.csv"
        model_name = "car_lane_model"
    
        if not os.path.exists(train_file):
            print("Training dataset not found. Please create the dataset first.")
            return
    
        # Load the training data
        df = pd.read_csv(train_file)
        X = df[["xmin", "ymin", "xmax", "ymax"]].values
        y = df["label"].values
    
        # Creating the model with more layers and neurons
        model = Sequential([
            Dense(128, activation="relu", input_dim=4),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid")
        ])
    
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
        # Training the model with increased epochs
        checkpoint = ModelCheckpoint(f"mp2_distance_predictor/carlane_model_weights/{model_name}.h5", save_best_only=True)
        model.fit(X, y, epochs=40, batch_size=32, validation_split=0.2, callbacks=[checkpoint])
    
        # Save the model
        model_json = model.to_json()
        with open(f"mp2_distance_predictor/carlane_model_weights/{model_name}.json", "w") as json_file:
            json_file.write(model_json)
    
        print("Saved car_lane model to disk.")
        self.carlane_model = model

    def load_trained_model(self):
            """
            Load the trained car lane model from disk.
            """
            model_name = "car_lane_model"
            json_file = open(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{model_name}.json", "r")
            loaded_model_json = json_file.read()
            json_file.close()
    
            model = Sequential()
            model = model_from_json(loaded_model_json)
            model.load_weights(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{model_name}.h5")
            print("Loaded car_lane model from disk.")
    
            self.carlane_model = model
    

    def check_lane_model_test(self):
            """
            Evaluate the trained model on the test dataset (`lane_test.csv`).
            """
            test_file = "/home1/varunjay/csci513-miniproject2/data/lane_test.csv"
            model_name = "car_lane_model"
    
            if not os.path.exists(test_file):
                print("Test dataset not found. Please create the dataset first.")
                return
    
            if not self.carlane_model:
                self.load_trained_model()
    
            df = pd.read_csv(test_file)
            X = df[["xmin", "ymin", "xmax", "ymax"]].values
            y_true = df["label"].values
    
            y_pred = (self.carlane_model.predict(X) > 0.5).astype(int)
            accuracy = accuracy_score(y_true, y_pred)
    
            print(f"Model accuracy on test data: {accuracy * 100:.2f}%")
    def predict(self, img):
        """
        Predict if the car in the image is in the same lane using the trained model.
        """
        if not self.carlane_model:
            self.load_trained_model()
    
        # Debug: Log the input image details
        print(f"Received input for prediction. Type: {type(img)}, Shape: {getattr(img, 'shape', 'Unknown')}")
    
        try:
            # Handle numpy array input by saving it as an image file
            if isinstance(img, np.ndarray):
                temp_dir = "/home1/varunjay/csci513-miniproject2/tmp"
                temp_image_path = os.path.join(temp_dir, "temp_image.png")
                
                # Ensure the temp directory exists
                os.makedirs(temp_dir, exist_ok=True)
    
                # Save the numpy array as an image
                plt.imsave(temp_image_path, img)
    
                # Update img to be the path to the temporary file
                img = temp_image_path
    
            # Use the image path to detect cars
            car_bounding_box = detect_cars(self.bb_detect_model, img)
            if car_bounding_box is None or len(car_bounding_box) == 0:
                print("No bounding box detected.")
                return "Not in lane"
    
            # Debug: Log bounding box details
            print(f"Detected bounding box: {car_bounding_box}")
    
            if isinstance(car_bounding_box[0], list):
                car_bounding_box = car_bounding_box[0]
    
            X = np.array([[car_bounding_box[0], car_bounding_box[1], car_bounding_box[2], car_bounding_box[3]]])
    
            # Debug: Log the input to the car lane model
            print(f"Input to car lane model: {X}")
    
            prediction = self.carlane_model.predict(X)
            print(f"Prediction result: {prediction}")
    
            return "In lane" if prediction > 0.5 else "Not in lane"
    
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Error"
    
    
    
    
from PIL import Image
import numpy as np

def main():
    # Correctly initialize the LeadCarLanePredictor object
    lead_carlane_predictor = LeadCarLanePredictor()

    # Create a dataset using the images provided
    lead_carlane_predictor.create_dataset()

    # Train the model to predict if the car is in the same lane or not
    lead_carlane_predictor.check_lane_model()

    # Test the model using the `lane_test.csv`
    lead_carlane_predictor.check_lane_model_test()

    #img_path = "/home1/varunjay/csci513-miniproject2/lane_data/ado_in_lane/camera_image_1.png"
    #result = lead_carlane_predictor.predict(img_path)
    #print(f"Prediction for {img_path}: {result}")
    

if __name__ == "__main__":
    main()


