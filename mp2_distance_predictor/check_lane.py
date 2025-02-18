
# # # import keras
# # # from keras.models import load_model
# # # from keras.models import Sequential
# # # # from keras.layers import Dense
# # # # from sklearn.preprocessing import StandardScaler
# # # # from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# # # class LeadCarLanePredictor:
# # #     def __init__(self):
# # #         self.bb_detect_model = load_model('mp2_distance_predictor/yolo_model.h5')
# # #         self.carlane_model = None


# # #     def create_dataset(self):
# # #         # The image data is stored in `lane_data` directory.
# # #         # The `lane_data` directory has 2 folders `ado_in_lane` and `ado_not_in_lane`.

# # #         # Use a pre-trained YOLOv3 model for detecting bounding boxes for all images in these folders.
# # #         # Assign class 1 for images in `ado_in_lane` and class 0 for `ado_not_in_lane`
# # #         # Create a dataset similar to `data/train.csv` and `data/test.csv` which contains bounding box from YOLOv3 and the corresponding class.
# # #         # Split this newly created dataset in `data/lane_train.csv` and `data/lane_test.csv`
        
# # #         pass


# # #     def check_lane_model(self):
# # #         # Your goal is to train a NN that can use the bounding box to predict if the lead car is in the same lane or not.
# # #         # Train a model that takes the newly created dataset i.e. bounding box of a car [[xmin, ymin, xmax, ymax]] 
        

# # #         modelname = "car_lane_model"
        
# # #         # Create a model and train it
# # #         # Do your magic
# # #         model = Sequential()











# # #         model_json = model.to_json()
# # #         with open("mp2_distance_predictor/carlane_model_weights/{}.json".format(modelname), "w+") as json_file:
# # #             json_file.write(model_json)
            
# # #         model.save_weights("mp2_distance_predictor/carlane_model_weights/{}.h5".format(modelname))
# # #         print("Saved car_lane model to disk")


# # #     def check_lane_model_test(self):
# # #         # Return True if ego car is in the same lane else return False
# # #         # Do your magic
        
        
# # #         return True
    

# # #     def predict(self, img):
# # #         # Use a pre-trained YOLOv3 model for detecting bounding boxes 
# # #         # Predict if the car is in lane or not using the `car_lane_model`
# # #         # Return True if ego car is in the same lane else return False

# # #         return True
         
         





# # # def main():
# # #     leadcarlanepredictor = LeadCarLanePredictor()
    
# # #     # Create a dataset using the images provided
# # #     leadcarlanepredictor.create_dataset()

# # #     # Train the model to predict if the car is the same lane or not
# # #     leadcarlanepredictor.check_lane_model()


# # #     #Test the model using `data/lane_test.csv`
# # #     leadcarlanepredictor.check_lane_model_test()

# # # if __name__ == '__main__':
# # # 	main()



# # # import keras
# # # from keras.models import load_model
# # # from keras.models import Sequential
# # # from keras.layers import Dense, Flatten, Input
# # # from keras.optimizers import Adam
# # # from sklearn.model_selection import train_test_split
# # # import pandas as pd
# # # import numpy as np
# # # import os
# # # from PIL import Image


# # # class LeadCarLanePredictor:
# # #     def __init__(self):
# # #         self.bb_detect_model = load_model('mp2_distance_predictor/yolo_model.h5')
# # #         self.carlane_model = None

# # #     def parse_yolo_output(self, predictions):
# # #         """
# # #         Parse YOLO output to extract bounding box with the highest confidence.
# # #         Assumes YOLO format: (batch_size, grid_size, grid_size, anchors, [x, y, w, h, confidence, classes...]).
# # #         """
# # #         # Flatten grid predictions
# # #         predictions = predictions.reshape((-1, predictions.shape[-1]))  # (anchors * grid_size^2, 255)

# # #         # Extract confidence scores and find the index of the highest confidence
# # #         confidence_scores = predictions[:, 4]  # Assuming confidence is at index 4
# # #         best_idx = np.argmax(confidence_scores)

# # #         # Extract bounding box for the highest confidence
# # #         best_box = predictions[best_idx][:4]  # [xmin, ymin, xmax, ymax]
# # #         return best_box

# # #     def create_dataset(self):
# # #         data = []
# # #         labels = []
# # #         batch = []
# # #         batch_labels = []
# # #         batch_size = 32

# # #         for label, folder in enumerate(["ado_in_lane", "ado_not_in_lane"]):
# # #             folder_path = os.path.join("lane_data", folder)
# # #             for img_file in os.listdir(folder_path):
# # #                 img_path = os.path.join(folder_path, img_file)

# # #                 # Load and preprocess the image
# # #                 img = Image.open(img_path).convert("RGB")
# # #                 img = img.resize((416, 416))
# # #                 img_array = np.array(img) / 255.0
# # #                 batch.append(img_array)
# # #                 batch_labels.append(label)

# # #                 # Process batch when full
# # #                 if len(batch) == batch_size:
# # #                     batch_array = np.array(batch)
# # #                     predictions = self.bb_detect_model.predict(batch_array)

# # #                     for pred, lbl in zip(predictions, batch_labels):
# # #                         try:
# # #                             bounding_box = self.parse_yolo_output(pred)
# # #                             data.append(bounding_box)
# # #                             labels.append(lbl)
# # #                         except IndexError:
# # #                             print("Skipping image with no valid bounding box.")

# # #                     batch = []
# # #                     batch_labels = []

# # #         # Process remaining images in the last batch
# # #         if batch:
# # #             batch_array = np.array(batch)
# # #             predictions = self.bb_detect_model.predict(batch_array)

# # #             for pred, lbl in zip(predictions, batch_labels):
# # #                 try:
# # #                     bounding_box = self.parse_yolo_output(pred)
# # #                     data.append(bounding_box)
# # #                     labels.append(lbl)
# # #                 except IndexError:
# # #                     print("Skipping image with no valid bounding box.")

# # #         # Convert to DataFrame
# # #         df = pd.DataFrame(data, columns=["xmin", "ymin", "xmax", "ymax"])
# # #         df["class"] = labels

# # #         # Split into train and test
# # #         train, test = train_test_split(df, test_size=0.2, random_state=42)
# # #         os.makedirs("data", exist_ok=True)
# # #         train.to_csv("data/lane_train.csv", index=False)
# # #         test.to_csv("data/lane_test.csv", index=False)

# # #     def check_lane_model(self):
# # #         modelname = "car_lane_model"

# # #         # Load dataset
# # #         train_data = pd.read_csv("data/lane_train.csv")
# # #         X_train = train_data[["xmin", "ymin", "xmax", "ymax"]].values
# # #         y_train = train_data["class"].values

# # #         # Define model
# # #         model = Sequential([
# # #             Input(shape=(4,)),
# # #             Dense(64, activation="relu"),
# # #             Dense(32, activation="relu"),
# # #             Dense(1, activation="sigmoid")
# # #         ])

# # #         model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

# # #         # Train model
# # #         model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# # #         # Save model
# # #         model_json = model.to_json()
# # #         with open(f"mp2_distance_predictor/carlane_model_weights/{modelname}.json", "w") as json_file:
# # #             json_file.write(model_json)

# # #         model.save_weights(f"mp2_distance_predictor/carlane_model_weights/{modelname}.h5")
# # #         print("Saved car_lane model to disk")

# # #     def check_lane_model_test(self):
# # #         # Load test dataset
# # #         test_data = pd.read_csv("data/lane_test.csv")
# # #         X_test = test_data[["xmin", "ymin", "xmax", "ymax"]].values
# # #         y_test = test_data["class"].values
    
# # #         # Load model
# # #         modelname = "car_lane_model"
# # #         with open(f"mp2_distance_predictor/carlane_model_weights/{modelname}.json", "r") as json_file:
# # #             model_json = json_file.read()
# # #         self.carlane_model = keras.models.model_from_json(model_json)
# # #         self.carlane_model.load_weights(f"mp2_distance_predictor/carlane_model_weights/{modelname}.h5")
    
# # #         # Compile the model before evaluating
# # #         self.carlane_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    
# # #         # Evaluate the model on the test data
# # #         loss, accuracy = self.carlane_model.evaluate(X_test, y_test, verbose=0)
    
# # #         # Print and return results
# # #         print(f"Test Accuracy: {accuracy * 100:.2f}%")
# # #         return accuracy > 0.8  # Return True if accuracy meets the threshold


# # #     def predict(self, img):
# # #         img = img.resize((416, 416))
# # #         img_array = np.array(img) / 255.0
# # #         img_array = np.expand_dims(img_array, axis=0)

# # #         bounding_box = self.parse_yolo_output(self.bb_detect_model.predict(img_array)[0])
# # #         prediction = self.carlane_model.predict(np.array([bounding_box]))
# # #         return prediction[0] > 0.5


# # # def main():
# # #     leadcarlanepredictor = LeadCarLanePredictor()
# # #     leadcarlanepredictor.create_dataset()
# # #     leadcarlanepredictor.check_lane_model()
# # #     leadcarlanepredictor.check_lane_model_test()


# # # if __name__ == '__main__':
# # #     main()



# # #### My codee

# # # import keras
# # # from keras.models import load_model, Sequential
# # # from keras.layers import Dense, Flatten, Input
# # # from keras.optimizers import Adam
# # # from sklearn.model_selection import train_test_split
# # # import pandas as pd
# # # import numpy as np
# # # import os
# # # from PIL import Image


# # # class LeadCarLanePredictor:
# # #     def __init__(self):
# # #         self.bb_detect_model = load_model('mp2_distance_predictor/yolo_model.h5')
# # #         self.carlane_model = None  # Initialize as None, will be loaded later

# # #     def parse_yolo_output(self, predictions):
# # #         predictions = predictions.reshape((-1, predictions.shape[-1]))  # Flatten grid predictions
# # #         confidence_scores = predictions[:, 4]  # Assuming confidence is at index 4
# # #         best_idx = np.argmax(confidence_scores)
# # #         best_box = predictions[best_idx][:4]  # Extract bounding box for the highest confidence
# # #         return best_box

# # #     def create_dataset(self):
# # #         data = []
# # #         labels = []
# # #         batch = []
# # #         batch_labels = []
# # #         batch_size = 32

# # #         for label, folder in enumerate(["ado_in_lane", "ado_not_in_lane"]):
# # #             folder_path = os.path.join("lane_data", folder)
# # #             for img_file in os.listdir(folder_path):
# # #                 img_path = os.path.join(folder_path, img_file)

# # #                 img = Image.open(img_path).convert("RGB")
# # #                 img = img.resize((416, 416))
# # #                 img_array = np.array(img) / 255.0
# # #                 batch.append(img_array)
# # #                 batch_labels.append(label)

# # #                 if len(batch) == batch_size:
# # #                     batch_array = np.array(batch)
# # #                     predictions = self.bb_detect_model.predict(batch_array)

# # #                     for pred, lbl in zip(predictions, batch_labels):
# # #                         try:
# # #                             bounding_box = self.parse_yolo_output(pred)
# # #                             data.append(bounding_box)
# # #                             labels.append(lbl)
# # #                         except IndexError:
# # #                             print("Skipping image with no valid bounding box.")

# # #                     batch = []
# # #                     batch_labels = []

# # #         if batch:
# # #             batch_array = np.array(batch)
# # #             predictions = self.bb_detect_model.predict(batch_array)

# # #             for pred, lbl in zip(predictions, batch_labels):
# # #                 try:
# # #                     bounding_box = self.parse_yolo_output(pred)
# # #                     data.append(bounding_box)
# # #                     labels.append(lbl)
# # #                 except IndexError:
# # #                     print("Skipping image with no valid bounding box.")

# # #         df = pd.DataFrame(data, columns=["xmin", "ymin", "xmax", "ymax"])
# # #         df["class"] = labels

# # #         train, test = train_test_split(df, test_size=0.2, random_state=42)
# # #         os.makedirs("data", exist_ok=True)
# # #         train.to_csv("data/lane_train.csv", index=False)
# # #         test.to_csv("data/lane_test.csv", index=False)

# # #     def check_lane_model(self):
# # #         modelname = "car_lane_model"

# # #         train_data = pd.read_csv("data/lane_train.csv")
# # #         X_train = train_data[["xmin", "ymin", "xmax", "ymax"]].values
# # #         y_train = train_data["class"].values

# # #         model = Sequential([
# # #             Input(shape=(4,)),
# # #             Dense(64, activation="relu"),
# # #             Dense(32, activation="relu"),
# # #             Dense(1, activation="sigmoid")
# # #         ])

# # #         model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
# # #         model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# # #         model_json = model.to_json()
# # #         with open(f"mp2_distance_predictor/carlane_model_weights/{modelname}.json", "w") as json_file:
# # #             json_file.write(model_json)

# # #         model.save_weights(f"mp2_distance_predictor/carlane_model_weights/{modelname}.h5")
# # #         print("Saved car_lane model to disk")

# # #     def load_carlane_model(self):
# # #         """Load the car lane model if it hasn't been loaded yet."""
# # #         if self.carlane_model is None:
# # #             modelname = "car_lane_model"
# # #             with open(f"mp2_distance_predictor/carlane_model_weights/{modelname}.json", "r") as json_file:
# # #                 model_json = json_file.read()
# # #             self.carlane_model = keras.models.model_from_json(model_json)
# # #             self.carlane_model.load_weights(f"mp2_distance_predictor/carlane_model_weights/{modelname}.h5")
# # #             self.carlane_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# # #             print("Car Lane Model Loaded Successfully.")

# # #     def check_lane_model_test(self):
# # #         self.load_carlane_model()
# # #         test_data = pd.read_csv("data/lane_test.csv")
# # #         X_test = test_data[["xmin", "ymin", "xmax", "ymax"]].values
# # #         y_test = test_data["class"].values
# # #         loss, accuracy = self.carlane_model.evaluate(X_test, y_test, verbose=0)
# # #         print(f"Test Accuracy: {accuracy * 100:.2f}%")
# # #         return accuracy > 0.8

# # #     def predict(self, img):
# # #         if isinstance(img, np.ndarray):
# # #             img = Image.fromarray(img)
# # #         img = img.resize((416, 416))
# # #         img_array = np.array(img) / 255.0
# # #         img_array = np.expand_dims(img_array, axis=0)

# # #         yolo_prediction = self.bb_detect_model.predict(img_array)[0]
# # #         bounding_box = self.parse_yolo_output(yolo_prediction)

# # #         self.load_carlane_model()  # Ensure the car lane model is loaded
# # #         prediction = self.carlane_model.predict(np.array([bounding_box]))
# # #         return prediction[0] > 0.5


# # # def main():
# # #     leadcarlanepredictor = LeadCarLanePredictor()
# # #     leadcarlanepredictor.create_dataset()
# # #     leadcarlanepredictor.check_lane_model()
# # #     leadcarlanepredictor.check_lane_model_test()


# # # if __name__ == '__main__':
# # #     main()


# # #### My code 2 accuracy 65.19%

# # import keras
# # from keras.models import load_model, Sequential
# # from keras.layers import Dense, Dropout, BatchNormalization, Input
# # from keras.optimizers import Adam
# # from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import StandardScaler
# # import pandas as pd
# # import numpy as np
# # import os
# # from PIL import Image
# # import joblib

# # class LeadCarLanePredictor:
# #     def __init__(self):
# #         self.bb_detect_model = load_model('mp2_distance_predictor/yolo_model.h5')
# #         self.carlane_model = None
# #         self.scaler = StandardScaler()

# #     def parse_yolo_output(self, predictions):
# #         predictions = predictions.reshape((-1, predictions.shape[-1]))
# #         confidence_scores = predictions[:, 4]
# #         best_idx = np.argmax(confidence_scores)
# #         best_box = predictions[best_idx][:4]
# #         return best_box

# #     def create_dataset(self):
# #         data = []
# #         labels = []
# #         batch = []
# #         batch_labels = []
# #         batch_size = 32

# #         for label, folder in enumerate(["ado_in_lane", "ado_not_in_lane"]):
# #             folder_path = os.path.join("lane_data", folder)
# #             for img_file in os.listdir(folder_path):
# #                 img_path = os.path.join(folder_path, img_file)

# #                 img = Image.open(img_path).convert("RGB")
# #                 img = img.resize((416, 416))
# #                 img_array = np.array(img) / 255.0
# #                 batch.append(img_array)
# #                 batch_labels.append(label)

# #                 if len(batch) == batch_size:
# #                     batch_array = np.array(batch)
# #                     predictions = self.bb_detect_model.predict(batch_array)

# #                     for pred, lbl in zip(predictions, batch_labels):
# #                         try:
# #                             bounding_box = self.parse_yolo_output(pred)
# #                             data.append(bounding_box)
# #                             labels.append(lbl)
# #                         except IndexError:
# #                             print("Skipping image with no valid bounding box.")

# #                     batch = []
# #                     batch_labels = []

# #         if batch:
# #             batch_array = np.array(batch)
# #             predictions = self.bb_detect_model.predict(batch_array)

# #             for pred, lbl in zip(predictions, batch_labels):
# #                 try:
# #                     bounding_box = self.parse_yolo_output(pred)
# #                     data.append(bounding_box)
# #                     labels.append(lbl)
# #                 except IndexError:
# #                     print("Skipping image with no valid bounding box.")

# #         df = pd.DataFrame(data, columns=["xmin", "ymin", "xmax", "ymax"])
# #         df["class"] = labels

# #         train, test = train_test_split(df, test_size=0.2, random_state=42)
# #         os.makedirs("data", exist_ok=True)
# #         train.to_csv("data/lane_train.csv", index=False)
# #         test.to_csv("data/lane_test.csv", index=False)

# #     def check_lane_model(self):
# #         modelname = "car_lane_model_improved"

# #         train_data = pd.read_csv("data/lane_train.csv")
# #         X_train = train_data[["xmin", "ymin", "xmax", "ymax"]].values
# #         y_train = train_data["class"].values

# #         X_train = self.scaler.fit_transform(X_train)

# #         model = Sequential([
# #             Input(shape=(4,)),
# #             Dense(128, activation="relu"),
# #             BatchNormalization(),
# #             Dropout(0.3),
# #             Dense(64, activation="relu"),
# #             BatchNormalization(),
# #             Dropout(0.3),
# #             Dense(32, activation="relu"),
# #             BatchNormalization(),
# #             Dropout(0.3),
# #             Dense(1, activation="sigmoid")
# #         ])

# #         optimizer = Adam(learning_rate=0.001)
# #         model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# #         early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# #         reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# #         history = model.fit(
# #             X_train, y_train,
# #             epochs=100,
# #             batch_size=32,
# #             validation_split=0.2,
# #             callbacks=[early_stopping, reduce_lr]
# #         )

# #         model_json = model.to_json()
# #         with open(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}.json", "w") as json_file:
# #             json_file.write(model_json)

# #         model.save_weights(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}.h5")
# #         print("Saved improved car_lane model to disk")

# #         joblib.dump(self.scaler, f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}_scaler.pkl")

# #     def load_carlane_model(self):
# #         if self.carlane_model is None:
# #             modelname = "car_lane_model_improved"
# #             with open(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}.json", "r") as json_file:
# #                 model_json = json_file.read()
# #             self.carlane_model = keras.models.model_from_json(model_json)
# #             self.carlane_model.load_weights(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}.h5")
# #             self.carlane_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            
# #             self.scaler = joblib.load(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}_scaler.pkl")
            
# #             print("Improved Car Lane Model Loaded Successfully.")

# #     def check_lane_model_test(self):
# #         self.load_carlane_model()
# #         test_data = pd.read_csv("data/lane_test.csv")
# #         X_test = test_data[["xmin", "ymin", "xmax", "ymax"]].values
# #         y_test = test_data["class"].values
        
# #         X_test = self.scaler.transform(X_test)
        
# #         loss, accuracy = self.carlane_model.evaluate(X_test, y_test, verbose=0)
# #         print(f"Test Accuracy: {accuracy * 100:.2f}%")
# #         return accuracy > 0.8

# #     def predict(self, img):
# #         if isinstance(img, np.ndarray):
# #             img = Image.fromarray(img)
# #         img = img.resize((416, 416))
# #         img_array = np.array(img) / 255.0
# #         img_array = np.expand_dims(img_array, axis=0)

# #         yolo_prediction = self.bb_detect_model.predict(img_array)[0]
# #         bounding_box = self.parse_yolo_output(yolo_prediction)

# #         self.load_carlane_model()
        
# #         scaled_bounding_box = self.scaler.transform(np.array([bounding_box]))
        
# #         prediction = self.carlane_model.predict(scaled_bounding_box)
# #         return prediction[0] > 0.5

# # def main():
# #     leadcarlanepredictor = LeadCarLanePredictor()
# #     leadcarlanepredictor.create_dataset()
# #     leadcarlanepredictor.check_lane_model()
# #     leadcarlanepredictor.check_lane_model_test()

# # if __name__ == '__main__':
# #     main()

# import keras
# from keras.models import load_model, Sequential
# from keras.layers import Dense, Dropout, BatchNormalization, Input
# from keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
# import numpy as np
# import os
# from PIL import Image
# import joblib

# class LeadCarLanePredictor:
#     def __init__(self):
#         self.bb_detect_model = load_model('mp2_distance_predictor/yolo_model.h5')
#         self.carlane_model = None
#         self.scaler = StandardScaler()

#     def parse_yolo_output(self, predictions):
#         predictions = predictions.reshape((-1, predictions.shape[-1]))
#         confidence_scores = predictions[:, 4]
#         best_idx = np.argmax(confidence_scores)
#         best_box = predictions[best_idx][:4]
#         return best_box

#     def create_dataset(self):
#         data = []
#         labels = []
#         batch = []
#         batch_labels = []
#         batch_size = 32

#         for label, folder in enumerate(["ado_in_lane", "ado_not_in_lane"]):
#             folder_path = os.path.join("lane_data", folder)
#             for img_file in os.listdir(folder_path):
#                 img_path = os.path.join(folder_path, img_file)

#                 img = Image.open(img_path).convert("RGB")
#                 img = img.resize((416, 416))
#                 img_array = np.array(img) / 255.0
#                 batch.append(img_array)
#                 batch_labels.append(label)

#                 if len(batch) == batch_size:
#                     batch_array = np.array(batch)
#                     predictions = self.bb_detect_model.predict(batch_array)

#                     for pred, lbl in zip(predictions, batch_labels):
#                         try:
#                             bounding_box = self.parse_yolo_output(pred)
#                             data.append(bounding_box)
#                             labels.append(lbl)
#                         except IndexError:
#                             print("Skipping image with no valid bounding box.")

#                     batch = []
#                     batch_labels = []

#         if batch:
#             batch_array = np.array(batch)
#             predictions = self.bb_detect_model.predict(batch_array)

#             for pred, lbl in zip(predictions, batch_labels):
#                 try:
#                     bounding_box = self.parse_yolo_output(pred)
#                     data.append(bounding_box)
#                     labels.append(lbl)
#                 except IndexError:
#                     print("Skipping image with no valid bounding box.")

#         df = pd.DataFrame(data, columns=["xmin", "ymin", "xmax", "ymax"])
#         df["class"] = labels

#         train, test = train_test_split(df, test_size=0.2, random_state=42)
#         os.makedirs("data", exist_ok=True)
#         train.to_csv("data/lane_train.csv", index=False)
#         test.to_csv("data/lane_test.csv", index=False)

#     def check_lane_model(self):
#         modelname = "car_lane_model_improved"

#         train_data = pd.read_csv("data/lane_train.csv")
#         X_train = train_data[["xmin", "ymin", "xmax", "ymax"]].values
#         y_train = train_data["class"].values

#         X_train = self.scaler.fit_transform(X_train)

#         model = Sequential([
#             Input(shape=(4,)),
#             Dense(128, activation="relu"),
#             BatchNormalization(),
#             Dropout(0.3),
#             Dense(64, activation="relu"),
#             BatchNormalization(),
#             Dropout(0.3),
#             Dense(32, activation="relu"),
#             BatchNormalization(),
#             Dropout(0.3),
#             Dense(1, activation="sigmoid")
#         ])

#         optimizer = Adam(learning_rate=0.001)
#         model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

#         early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#         reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

#         history = model.fit(
#             X_train, y_train,
#             epochs=100,
#             batch_size=32,
#             validation_split=0.2,
#             callbacks=[early_stopping, reduce_lr]
#         )

#         model_json = model.to_json()
#         with open(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}.json", "w") as json_file:
#             json_file.write(model_json)

#         model.save_weights(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}.h5")
#         print("Saved improved car_lane model to disk")

#         joblib.dump(self.scaler, f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}_scaler.pkl")

#     def load_carlane_model(self):
#         if self.carlane_model is None:
#             modelname = "car_lane_model_improved"
#             with open(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}.json", "r") as json_file:
#                 model_json = json_file.read()
#             self.carlane_model = keras.models.model_from_json(model_json)
#             self.carlane_model.load_weights(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}.h5")
#             self.carlane_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            
#             self.scaler = joblib.load(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}_scaler.pkl")
            
#             print("Improved Car Lane Model Loaded Successfully.")

#     def check_lane_model_test(self):
#         self.load_carlane_model()
#         test_data = pd.read_csv("data/lane_test.csv")
#         X_test = test_data[["xmin", "ymin", "xmax", "ymax"]].values
#         y_test = test_data["class"].values
        
#         X_test = self.scaler.transform(X_test)
        
#         loss, accuracy = self.carlane_model.evaluate(X_test, y_test, verbose=0)
#         print(f"Test Accuracy: {accuracy * 100:.2f}%")
#         return accuracy > 0.8

#     def predict(self, img):
#         if isinstance(img, np.ndarray):
#             img = Image.fromarray(img)
#         img = img.resize((416, 416))
#         img_array = np.array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         yolo_prediction = self.bb_detect_model.predict(img_array)[0]
#         bounding_box = self.parse_yolo_output(yolo_prediction)

#         self.load_carlane_model()
        
#         scaled_bounding_box = self.scaler.transform(np.array([bounding_box]))
        
#         prediction = self.carlane_model.predict(scaled_bounding_box)
#         return prediction[0] > 0.5

# def main():
#     leadcarlanepredictor = LeadCarLanePredictor()
#     leadcarlanepredictor.create_dataset()
#     leadcarlanepredictor.check_lane_model()
#     leadcarlanepredictor.check_lane_model_test()

# if __name__ == '__main__':
#     main()


import keras
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os
from PIL import Image
import joblib

class LeadCarLanePredictor:
    def __init__(self):
        self.bb_detect_model = load_model('mp2_distance_predictor/yolo_model.h5')
        self.carlane_model = None
        self.scaler = StandardScaler()

    def parse_yolo_output(self, predictions):
        predictions = predictions.reshape((-1, predictions.shape[-1]))
        confidence_scores = predictions[:, 4]
        best_idx = np.argmax(confidence_scores)
        best_box = predictions[best_idx][:4]
        return best_box

    def create_dataset(self):
        data = []
        labels = []
        batch = []
        batch_labels = []
        batch_size = 32

        for label, folder in enumerate(["ado_in_lane", "ado_not_in_lane"]):
            folder_path = os.path.join("lane_data", folder)
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)

                img = Image.open(img_path).convert("RGB")
                img = img.resize((416, 416))
                img_array = np.array(img) / 255.0
                batch.append(img_array)
                batch_labels.append(label)

                if len(batch) == batch_size:
                    batch_array = np.array(batch)
                    predictions = self.bb_detect_model.predict(batch_array)

                    for pred, lbl in zip(predictions, batch_labels):
                        try:
                            bounding_box = self.parse_yolo_output(pred)
                            data.append(bounding_box)
                            labels.append(lbl)
                        except IndexError:
                            print("Skipping image with no valid bounding box.")

                    batch = []
                    batch_labels = []

        if batch:
            batch_array = np.array(batch)
            predictions = self.bb_detect_model.predict(batch_array)

            for pred, lbl in zip(predictions, batch_labels):
                try:
                    bounding_box = self.parse_yolo_output(pred)
                    data.append(bounding_box)
                    labels.append(lbl)
                except IndexError:
                    print("Skipping image with no valid bounding box.")

        df = pd.DataFrame(data, columns=["xmin", "ymin", "xmax", "ymax"])
        df["class"] = labels

        train, test = train_test_split(df, test_size=0.2, random_state=42)
        os.makedirs("data", exist_ok=True)
        train.to_csv("data/lane_train.csv", index=False)
        test.to_csv("data/lane_test.csv", index=False)

    def check_lane_model(self):
        modelname = "car_lane_model_improved"

        train_data = pd.read_csv("data/lane_train.csv")
        X_train = train_data[["xmin", "ymin", "xmax", "ymax"]].values
        y_train = train_data["class"].values

        X_train = self.scaler.fit_transform(X_train)

        model = Sequential([
            Input(shape=(4,)),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation="sigmoid")
        ])

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr]
        )

        model_json = model.to_json()
        with open(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}.json", "w") as json_file:
            json_file.write(model_json)

        model.save_weights(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}.h5")
        print("Saved improved car_lane model to disk")

        joblib.dump(self.scaler, f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}_scaler.pkl")

    def load_carlane_model(self):
        if self.carlane_model is None:
            modelname = "car_lane_model_improved"
            with open(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}.json", "r") as json_file:
                model_json = json_file.read()
            self.carlane_model = keras.models.model_from_json(model_json)
            self.carlane_model.load_weights(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}.h5")
            self.carlane_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            
            self.scaler = joblib.load(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/carlane_model_weights/{modelname}_scaler.pkl")
            
            print("Improved Car Lane Model Loaded Successfully.")

    def check_lane_model_test(self):
        self.load_carlane_model()
        test_data = pd.read_csv("data/lane_test.csv")
        X_test = test_data[["xmin", "ymin", "xmax", "ymax"]].values
        y_test = test_data["class"].values
        
        X_test = self.scaler.transform(X_test)
        
        loss, accuracy = self.carlane_model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        return accuracy > 0.8

    def predict(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = img.resize((416, 416))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        yolo_prediction = self.bb_detect_model.predict(img_array)[0]
        bounding_box = self.parse_yolo_output(yolo_prediction)

        self.load_carlane_model()
        
        scaled_bounding_box = self.scaler.transform(np.array([bounding_box]))
        
        prediction = self.carlane_model.predict(scaled_bounding_box)
        return prediction[0] > 0.5

def main():
    leadcarlanepredictor = LeadCarLanePredictor()
    leadcarlanepredictor.create_dataset()
    leadcarlanepredictor.check_lane_model()
    leadcarlanepredictor.check_lane_model_test()

if __name__ == '__main__':
    main()
