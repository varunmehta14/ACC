# # # import pandas as pd
# # # import numpy as np
# # # import time

# # # import keras
# # # from keras.models import Sequential
# # # from keras.layers import Dense
# # # from sklearn.preprocessing import StandardScaler
# # # from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# # # def main():
# # # 	# ----------- import data and scaling ----------- #
# # # 	df_train = pd.read_csv('data/train.csv')
# # # 	df_test = pd.read_csv('data/test.csv')

# # # 	X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
# # # 	y_train = df_train[['zloc']].values

# # # 	X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
# # # 	y_test = df_test[['zloc']].values

# # # 	# standardized data
# # # 	scalar = StandardScaler()
# # # 	X_train = scalar.fit_transform(X_train)
# # # 	y_train = scalar.fit_transform(y_train)

# # # 	# ----------- create model ----------- #
# # #     ### Do your magic!
	
# # #     # This model takes in a bounding box of a car [[xmin, ymin, xmax, ymax]] and returns the 
# # #     # distance to the car. Simple models will perform well for this problem, but the below 
# # #     # single layer perceptron is too small to perform well. See the keras documentation 
# # #     # https://keras.io/api/layers/ for adding more layers. 
# # # 	model = Sequential()
# # # 	model.add(Dense(1, input_dim=4, kernel_initializer='normal', activation='relu'))


# # # 	model.compile(loss='mean_squared_error', optimizer='adam')

# # # 	# ----------- define callbacks ----------- #
# # # 	earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
# # # 	reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
# # # 									   verbose=1, epsilon=1e-4, mode='min')
# # # 	modelname = "distance_model"
# # # 	tensorboard = TensorBoard(log_dir="training_logs/{}".format(modelname))

# # # 	# ----------- start training ----------- #
# # # 	history = model.fit(X_train, y_train,
# # # 	 							 validation_split=0.1, epochs=5000, batch_size=2048,
# # # 	 							 callbacks=[tensorboard], verbose=1)

# # # 	# ----------- save model and weights ----------- #
# # # 	model_json = model.to_json()
# # # 	with open("mp2_distance_predictor/distance_model_weights/{}.json".format(modelname), "w+") as json_file:
# # # 		json_file.write(model_json)
		
# # # 	model.save_weights("mp2_distance_predictor/distance_model_weights/{}.h5".format(modelname))
# # # 	print("Saved model to disk")

# # # if __name__ == '__main__':
# # # 	main()



# # ####### My code

# # import pandas as pd
# # import numpy as np
# # import time

# # import keras
# # from keras.models import Sequential
# # from keras.layers import Dense, Dropout, BatchNormalization
# # from sklearn.preprocessing import StandardScaler
# # from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# # def main():
# #     # ----------- import data and scaling ----------- #
# #     df_train = pd.read_csv('data/train.csv')
# #     df_test = pd.read_csv('data/test.csv')

# #     X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
# #     y_train = df_train[['zloc']].values

# #     X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
# #     y_test = df_test[['zloc']].values

# #     # standardized data
# #     scalar_X = StandardScaler()
# #     scalar_y = StandardScaler()
# #     X_train = scalar_X.fit_transform(X_train)
# #     y_train = scalar_y.fit_transform(y_train)
# #     X_test = scalar_X.transform(X_test)
# #     y_test = scalar_y.transform(y_test) 

# #     # ----------- create model ----------- #
# #     model = Sequential()
    
# #     # Input layer
# #     model.add(Dense(64, input_dim=4, kernel_initializer='he_uniform', activation='relu'))
# #     model.add(BatchNormalization())
# #     model.add(Dropout(0.2))

# #     # Hidden layers
# #     model.add(Dense(128, kernel_initializer='he_uniform', activation='relu'))
# #     model.add(BatchNormalization())
# #     model.add(Dropout(0.3))

# #     model.add(Dense(64, kernel_initializer='he_uniform', activation='relu'))
# #     model.add(BatchNormalization())
# #     model.add(Dropout(0.2))

# #     # Output layer
# #     model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# #     # Compile the model
# #     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

# #     # ----------- define callbacks ----------- #
# #     earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
# #     reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, 
# #                                        verbose=1, min_delta=1e-4, mode='min')
# #     modelname = "distance_model"
# #     tensorboard = TensorBoard(log_dir=f"training_logs/{modelname}")
# #     checkpoint = ModelCheckpoint(filepath=f"mp2_distance_predictor/distance_model_weights/{modelname}.h5",
# #                                  monitor='val_loss', save_best_only=True, verbose=1)

# #     # ----------- start training ----------- #
# #     history = model.fit(X_train, y_train,
# #                         validation_split=0.1,
# #                         epochs=500,
# #                         batch_size=2048,
# #                         callbacks=[tensorboard, earlyStopping, reduce_lr_loss, checkpoint],
# #                         verbose=1)

# #     # ----------- save model and weights ----------- #
# #     model_json = model.to_json()
# #     with open(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{modelname}.json", "w") as json_file:
# #         json_file.write(model_json)

# #     model.save_weights(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{modelname}.h5")
# #     print("Saved model to disk")

# #     # ----------- evaluate model ----------- #
# #     loss, mse = model.evaluate(X_test, y_test, verbose=0)
# #     print(f"Test Loss: {loss}, Test MSE: {mse}")

# # if __name__ == '__main__':
# #     main()
    
# # #Test Loss: 0.06306145340204239, Test MSE: 0.06306145340204239


# # import pandas as pd
# # import numpy as np
# # import time

# # import keras
# # from keras.models import Sequential
# # from keras.layers import Dense, Dropout, BatchNormalization
# # from sklearn.preprocessing import StandardScaler
# # from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# # def main():
# #     # ----------- import data and scaling ----------- #
# #     df_train = pd.read_csv('data/train.csv')
# #     df_test = pd.read_csv('data/test.csv')

# #     X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
# #     y_train = df_train[['zloc']].values

# #     X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
# #     y_test = df_test[['zloc']].values

# #     # standardized data
# #     scalar_X = StandardScaler()
# #     scalar_y = StandardScaler()
# #     X_train = scalar_X.fit_transform(X_train)
# #     y_train = scalar_y.fit_transform(y_train)
# #     X_test = scalar_X.transform(X_test)
# #     y_test = scalar_y.transform(y_test)

# #     # ----------- create model ----------- #
# #     model = Sequential()
    
# #     # Input layer
# #     model.add(Dense(128, input_dim=4, kernel_initializer='he_uniform', activation='relu'))
# #     model.add(BatchNormalization())
# #     model.add(Dropout(0.3))

# #     # Hidden layers
# #     model.add(Dense(256, kernel_initializer='he_uniform', activation='relu'))
# #     model.add(BatchNormalization())
# #     model.add(Dropout(0.4))

# #     model.add(Dense(128, kernel_initializer='he_uniform', activation='relu'))
# #     model.add(BatchNormalization())
# #     model.add(Dropout(0.3))

# #     model.add(Dense(64, kernel_initializer='he_uniform', activation='relu'))
# #     model.add(BatchNormalization())
# #     model.add(Dropout(0.2))

# #     # Output layer
# #     model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# #     # Compile the model
# #     model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

# #     # ----------- define callbacks ----------- #
# #     earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min')
# #     reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=15, 
# #                                        verbose=1, min_delta=1e-4, mode='min')
# #     modelname = "distance_model"
# #     tensorboard = TensorBoard(log_dir=f"training_logs/{modelname}")
# #     checkpoint = ModelCheckpoint(filepath=f"mp2_distance_predictor/distance_model_weights/{modelname}.h5",
# #                                  monitor='val_loss', save_best_only=True, verbose=1)

# #     # ----------- start training ----------- #
# #     history = model.fit(X_train, y_train,
# #                         validation_split=0.1,
# #                         epochs=1000,
# #                         batch_size=1024,
# #                         callbacks=[tensorboard, earlyStopping, reduce_lr_loss, checkpoint],
# #                         verbose=1)

# #     # ----------- save model and weights ----------- #
# #     model_json = model.to_json()
# #     with open(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{modelname}.json", "w") as json_file:
# #         json_file.write(model_json)

# #     model.save_weights(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{modelname}.h5")
# #     print("Saved model to disk")

# #     # ----------- evaluate model ----------- #
# #     loss, mse = model.evaluate(X_test, y_test, verbose=0)
# #     print(f"Test Loss: {loss}, Test MSE: {mse}")

# # if __name__ == '__main__':
# #     main()

# # #Test Loss: 0.06411547213792801, Test MSE: 0.0641154795885086



# import pandas as pd
# import numpy as np
# import time

# import keras
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, BatchNormalization
# from keras.regularizers import l2
# from keras.optimizers import Adam
# from sklearn.preprocessing import StandardScaler
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
# from sklearn.model_selection import train_test_split

# def create_model():
#     model = Sequential()
    
#     # Input layer
#     model.add(Dense(64, input_dim=4, kernel_initializer='he_uniform', activation='relu', kernel_regularizer=l2(0.01)))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.3))

#     # Hidden layers
#     model.add(Dense(128, kernel_initializer='he_uniform', activation='relu', kernel_regularizer=l2(0.01)))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.4))

#     model.add(Dense(64, kernel_initializer='he_uniform', activation='relu', kernel_regularizer=l2(0.01)))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.3))

#     # Output layer
#     model.add(Dense(1, kernel_initializer='normal', activation='linear'))

#     optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
#     model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    
#     return model

# def main():
#     # ----------- import data and scaling ----------- #
#     df_train = pd.read_csv('data/train.csv')
#     df_test = pd.read_csv('data/test.csv')

#     X = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
#     y = df_train[['zloc']].values

#     X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
#     y_test = df_test[['zloc']].values

#     # Split training data into train and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Standardize data
#     scalar_X = StandardScaler()
#     scalar_y = StandardScaler()
#     X_train = scalar_X.fit_transform(X_train)
#     y_train = scalar_y.fit_transform(y_train)
#     X_val = scalar_X.transform(X_val)
#     y_val = scalar_y.transform(y_val)
#     X_test = scalar_X.transform(X_test)
#     y_test = scalar_y.transform(y_test)

#     # ----------- create model ----------- #
#     model = create_model()

#     # ----------- define callbacks ----------- #
#     earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min', restore_best_weights=True)
#     reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_delta=1e-4, mode='min')
#     modelname = "distance_model_improved"
#     tensorboard = TensorBoard(log_dir=f"training_logs/{modelname}")
#     checkpoint = ModelCheckpoint(filepath=f"mp2_distance_predictor/distance_model_weights/{modelname}.h5",
#                                  monitor='val_loss', save_best_only=True, verbose=1)

#     # ----------- start training ----------- #
#     history = model.fit(X_train, y_train,
#                         validation_data=(X_val, y_val),
#                         epochs=1000,
#                         batch_size=512,
#                         callbacks=[tensorboard, earlyStopping, reduce_lr_loss, checkpoint],
#                         verbose=1)

#     # ----------- save model and weights ----------- #
#     model_json = model.to_json()
#     with open(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{modelname}.json", "w") as json_file:
#         json_file.write(model_json)

#     model.save_weights(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{modelname}.h5")
#     print("Saved model to disk")

#     # ----------- evaluate model ----------- #
#     loss, mse = model.evaluate(X_test, y_test, verbose=0)
#     print(f"Test Loss: {loss}, Test MSE: {mse}")

#     # ----------- make predictions ----------- #
#     y_pred = model.predict(X_test)
#     y_pred = scalar_y.inverse_transform(y_pred)
#     y_test = scalar_y.inverse_transform(y_test)

#     # Calculate RMSE
#     rmse = np.sqrt(np.mean((y_pred - y_test)**2))
#     print(f"Root Mean Squared Error: {rmse}")

# if __name__ == '__main__':
#     main()

import pandas as pd
import numpy as np
import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split

def create_model():
    model = Sequential()
    
    # Input layer
    model.add(Dense(64, input_dim=4, kernel_initializer='he_uniform', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Hidden layers
    model.add(Dense(128, kernel_initializer='he_uniform', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(64, kernel_initializer='he_uniform', activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    
    return model

def main():
    # ----------- import data and scaling ----------- #
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    X = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
    y = df_train[['zloc']].values

    X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_test = df_test[['zloc']].values

    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scalar_X = StandardScaler()
    scalar_y = StandardScaler()
    X_train = scalar_X.fit_transform(X_train)
    y_train = scalar_y.fit_transform(y_train)
    X_val = scalar_X.transform(X_val)
    y_val = scalar_y.transform(y_val)
    X_test = scalar_X.transform(X_test)
    y_test = scalar_y.transform(y_test)

    # ----------- create model ----------- #
    model = create_model()

    # ----------- define callbacks ----------- #
    earlyStopping = EarlyStopping(monitor='val_loss', patience=30, verbose=1, mode='min', restore_best_weights=True)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_delta=1e-4, mode='min')
    modelname = "distance_model_improved"
    tensorboard = TensorBoard(log_dir=f"training_logs/{modelname}")
    checkpoint = ModelCheckpoint(filepath=f"mp2_distance_predictor/distance_model_weights/{modelname}.h5",
                                 monitor='val_loss', save_best_only=True, verbose=1)

    # ----------- start training ----------- #
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=1000,
                        batch_size=512,
                        callbacks=[tensorboard, earlyStopping, reduce_lr_loss, checkpoint],
                        verbose=1)

    # ----------- save model and weights ----------- #
    model_json = model.to_json()
    with open(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{modelname}.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(f"/home1/varunjay/csci513-miniproject2/mp2_distance_predictor/distance_model_weights/{modelname}.h5")
    print("Saved model to disk")

    # ----------- evaluate model ----------- #
    loss, mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss}, Test MSE: {mse}")

    # ----------- make predictions ----------- #
    y_pred = model.predict(X_test)
    y_pred = scalar_y.inverse_transform(y_pred)
    y_test = scalar_y.inverse_transform(y_test)

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_pred - y_test)**2))
    print(f"Root Mean Squared Error: {rmse}")

if __name__ == '__main__':
    main()
