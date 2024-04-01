import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

#scikit-learn imports
from sklearn.metrics import mean_squared_error as mse

#tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def get_data():
    """
    This method reads data from weather archives and returns a pandas dataframe
    :return: df
    """
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    df = df[5::6] #extract data at every hour interval
    df.index = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')

    return df

def df_to_X_y(df, window_size):

    ### reshaping the data for LSTM in the following shape
    # [[1, 2, 3, 4, 5]] [6]
    # [[2, 3, 4, 5, 6]] [7]

    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - window_size):
        row = [a for a in df_as_np[i:i+window_size]]
        X.append(row)
        label = df_as_np[i+window_size][0]
        y.append(label)

    return np.array(X), np.array(y)

def preprocess(X):

  if(len(X.shape) > 1):
    x_mean = np.mean(X[:,:,0])
    x_std = np.std(X[:,:,0])
    X[:,:,0] = (X[:,:,0] - x_mean) / x_std
  else:
    y_mean = np.mean(X)
    y_std = np.std(X)
    X = (X - y_mean) / y_std
  return X

def plot_predictions1(model, X, y, start=0, end=100,plot=False):

    predictions = model.predict(X).flatten()
    df = pd.DataFrame(data={'Predictions': predictions, 'Actuals':y})
    if plot:
        plt.plot(df['Predictions'][start:end],label='prediction')
        plt.plot(df['Actuals'][start:end],label='actual')
        plt.legend()
        plt.show()
    return df, mse(predictions, y)

def plot_performance(train_results, start=0,end=100,plot=False):
    if plot:
        performance = (train_results[:end]['Predictions'] - train_results[:100]['Actuals'])
        plt.plot(performance[:end],label='error margin')
        plt.plot(train_results[:100]['Predictions'],label='predictions')
        plt.plot(train_results[:end]['Actuals'], label='actuals')
        plt.legend()
        plt.show()

if __name__ == '__main__':

    window_size = 5
    plot_perf = True
    df = get_data()
    temp_df = df[['T (degC)','rh (%)','p (mbar)','VPact (mbar)','VPact (mbar)','max. wv (m/s)']]
    df_temp_ = temp_df.copy()
    df_temp_ = pd.DataFrame(df_temp_).rename(columns={'T (degC)':'temperature','rh (%)':'rh_perc',\
                                                      'p (mbar)':'p_mbar','VPact (mbar)':'vpact_mbar','max. wv (m/s)':'max_wv_m_per_sec'})

    #adding more features to the dataframe

    X1, y1 = df_to_X_y(df_temp_, window_size)

    #Normalize the target
    X1 = preprocess(X1)
    y1 = preprocess(y1)

    x_train1, y_train1 = X1[:60000], y1[:60000]
    x_val1, y_val1 = X1[60000:65000], y1[60000:65000]
    x_test1, y_test1 = X1[65000:], y1[65000:]

    print(x_train1.shape, y_train1.shape, x_val1.shape, y_val1.shape, x_test1.shape, y_test1.shape)
    num_features = x_train1.shape[2]
    model1 = Sequential()
    model1.add(InputLayer((window_size, num_features)))
    model1.add(LSTM(64))
    model1.add(Dense(8, 'relu'))
    model1.add(Dense(1, 'linear'))

    model1.summary()
    cp1 = ModelCheckpoint('/c//Users//SIMON//Documents//trading//deeplearning//tensorflow_test1//models//model1.keras', save_best_only=True)
    model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])
    model1.fit(x_train1, y_train1, validation_data=(x_val1, y_val1), epochs=50, callbacks=[cp1])
    model1 = load_model('/c//Users//SIMON//Documents//trading//deeplearning//tensorflow_test1//models//model1.keras')

    test_results, perf = plot_predictions1(model1,x_test1, y_test1, plot=False)
    print(f"performance: {perf}")
    plot_performance(test_results)
    print("the end")