from keras.models import load_model
import keras.models
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

file_path = "D:/sem232/thesis/pythontest/my_thesis_db/mysite/polls/savedmodels/"
file_path_2 = "D:/sem232/thesis/pythontest/my_thesis_db/mysite/polls/savedmodels/mini/"


def loadnewmodel(id):
    global model
    model = load_model(file_path + f"smallmodel/smallmodel_{id}.h5")
    global threshold
    file_path_2 = (
        "D:/sem232/thesis/pythontest/my_thesis_db/mysite/polls/savedmodels/mini/"
        + f"variables_{id}.pkl"
    )
    with open(file_path_2, "rb") as file:
        loaded_var1, loaded_var2, loaded_var3, accu = pickle.load(file)
    threshold = loaded_var1
    hint = (f"{id}: ", threshold)
    return hint


def predict2(data):

    data = pd.DataFrame(data)
    data = data[["x", "y", "z"]].values
    noise = np.random.normal(loc=0, scale=0.05, size=data.shape)
    data = data + noise
    features = extract_features(data)

    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    pred = model.predict(features_tensor)
    mse = np.mean(np.power(features - pred, 2), axis=1)
    print(mse)
    if mse > 0.0003:
        return 1
    else:
        return 0


def extract_features(sample, scale=1):
    x = MinMaxScaler()
    features = []
    # sample = sample[0:max_measurements]
    sample = x.fit_transform(sample)
    # Scale sample
    sample = scale * sample

    #     # Variance
    features.append(np.var(sample, axis=0))

    return np.array(features)
