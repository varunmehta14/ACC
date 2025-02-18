# import pandas as pd
# import numpy as np
# import time

# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.preprocessing import StandardScaler
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# def main():
# 	# ----------- import data and scaling ----------- #
# 	df_train = pd.read_csv('data/train.csv')
# 	df_test = pd.read_csv('data/test.csv')

# 	X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
# 	y_train = df_train[['zloc']].values

# 	X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
# 	y_test = df_test[['zloc']].values

# 	# standardized data
# 	scalar = StandardScaler()
# 	X_train = scalar.fit_transform(X_train)
# 	y_train = scalar.fit_transform(y_train)

# 	# ----------- create model ----------- #
#     ### Do your magic!
	
#     # This model takes in a bounding box of a car [[xmin, ymin, xmax, ymax]] and returns the 
#     # distance to the car. Simple models will perform well for this problem, but the below 
#     # single layer perceptron is too small to perform well. See the keras documentation 
#     # https://keras.io/api/layers/ for adding more layers. 
# 	model = Sequential()
# 	#model.add(Dense(1, input_dim=4, kernel_initializer='normal', activation='relu'))
# # ----------- create model ----------- #
	
# 	model.add(Dense(128, input_dim=4, activation='relu', kernel_initializer='he_normal'))
# 	model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
# 	model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
# 	model.add(Dense(1, activation='linear'))


# 	model.compile(loss='mean_squared_error', optimizer='adam')

# 	# ----------- define callbacks ----------- #
# 	earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
# 	reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
# 									   verbose=1, epsilon=1e-4, mode='min')
# 	modelname = "distance_model"
# 	tensorboard = TensorBoard(log_dir="training_logs/{}".format(modelname))

# 	# ----------- start training ----------- #
# 	history = model.fit(X_train, y_train,
# 	 							 validation_split=0.1, epochs=5000, batch_size=2048,
# 	 							 callbacks=[tensorboard], verbose=1)

# 	# ----------- save model and weights ----------- #
# 	model_json = model.to_json()
# 	with open("/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{}.json".format(modelname), "w+") as json_file:
# 		json_file.write(model_json)
		
# 	model.save_weights("/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{}.h5".format(modelname))
# 	print("Saved model to disk")

# if __name__ == '__main__':
# 	main()


import pandas as pd
import numpy as np
import time

import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

def main():
	# ----------- import data and scaling ----------- #
	df_train = pd.read_csv('data/train.csv')
	df_test = pd.read_csv('data/test.csv')

	X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
	y_train = df_train[['zloc']].values

	X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
	y_test = df_test[['zloc']].values

	# standardized data
	scalar = StandardScaler()
	X_train = scalar.fit_transform(X_train)
	y_train = scalar.fit_transform(y_train)

	# ----------- create model ----------- #
    ### Do your magic!
	
    # This model takes in a bounding box of a car [[xmin, ymin, xmax, ymax]] and returns the 
    # distance to the car. Simple models will perform well for this problem, but the below 
    # single layer perceptron is too small to perform well. See the keras documentation 
    # https://keras.io/api/layers/ for adding more layers. 
	model = Sequential()
	#model.add(Dense(1, input_dim=4, kernel_initializer='normal', activation='relu'))
# ----------- create model ----------- #
	
	model.add(Dense(128, input_dim=4, activation='relu', kernel_initializer='he_normal'))
	model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
	model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
	model.add(Dense(1, activation='linear'))


	model.compile(loss='mean_squared_error', optimizer='adam')

	# ----------- define callbacks ----------- #
	earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
	reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
									   verbose=1, epsilon=1e-4, mode='min')
	modelname = "distance_model"
	tensorboard = TensorBoard(log_dir="training_logs/{}".format(modelname))

	# ----------- start training ----------- #
	history = model.fit(X_train, y_train,
	 							 validation_split=0.1, epochs=5000, batch_size=2048,
	 							 callbacks=[tensorboard], verbose=1)

	# ----------- save model and weights ----------- #
	model_json = model.to_json()
	with open("/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{}.json".format(modelname), "w+") as json_file:
		json_file.write(model_json)
		
	model.save_weights("/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{}.h5".format(modelname))
	print("Saved model to disk")

if __name__ == '__main__':
	main()
