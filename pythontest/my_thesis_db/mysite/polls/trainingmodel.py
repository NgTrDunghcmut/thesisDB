from keras.models import load_model
from keras import layers, models, optimizers, regularizers
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
import pandas as pd
import keras
import pickle

win_len = 400


def training_model(data, id):
    result = []
    data = pd.DataFrame(data)
    x = MinMaxScaler()
    normal_op = x.fit_transform(data)
    filenames = pd.DataFrame(normal_op)
    windows_normal = filenames.values[
        np.arange(win_len)[None, :] + np.arange(normal_op.shape[0] - win_len)[:, None]
    ]
    np.random.shuffle(windows_normal)
    windows_normal_train = windows_normal[
        : int(np.floor(0.6 * windows_normal.shape[0]))
    ]
    windows_normal_val = windows_normal[
        int(np.floor(0.6 * windows_normal.shape[0])) : int(
            np.floor(0.8 * windows_normal.shape[0])
        )
    ]
    windows_normal_test = windows_normal[int(np.floor(0.8 * windows_normal.shape[0])) :]
    x_train = create_feature_set(windows_normal_train)

    x_val = create_feature_set(windows_normal_val)
    x_test = create_feature_set(windows_normal_test)
    sample_shape = x_train.shape[1:]
    encoding_dim = 2
    model = models.Sequential(
        [
            layers.InputLayer(input_shape=sample_shape),
            layers.Dense(*sample_shape, activation="leaky_relu"),
            layers.Dense(encoding_dim, activation="leaky_relu"),
            layers.Dense(encoding_dim, activation="leaky_relu"),
            layers.Dense(*sample_shape, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam", loss=tf.keras.losses.mean_squared_error, metrics=["accuracy"]
    )
    stop = keras.callbacks.EarlyStopping(
        monitor="loss",
        min_delta=0,
        patience=10,
        restore_best_weights=True,
    )

    history = model.fit(
        x_train,
        x_train,
        epochs=1300,
        batch_size=100,
        validation_data=(x_val, x_val),
        callbacks=stop,
        verbose=1,
    )
    loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]
    predictions = model.predict(x_val)
    normal_mse = np.mean(np.power(x_val - predictions, 2), axis=1)
    recommend = np.percentile(normal_mse, 99.98)
    testpred = model.predict(x_test)
    testmse = np.mean(np.power(x_test - testpred, 2), axis=1)
    for i in testmse:
        result.append(1 if i > recommend else 0)
    accu = (len(result) - sum(result)) / len(testpred)
    db_file_path = "D:/sem232/thesis/pythontest/my_thesis_db/mysite/polls/savedmodels/"
    model.save(db_file_path + f"/smallmodel/smallmodel_{id}.h5")
    with open(db_file_path + f"/mini/variables_{id}.pkl", "wb") as file:
        pickle.dump([recommend, loss, val_loss, accu, normal_mse], file)

    return


# Function: extract specified features (variances, MAD) from sample
def extract_features(sample, max_measurements=0, scale=1):

    features = []

    # Truncate sample
    if max_measurements == 0:
        max_measurements = sample.shape[0]

    # Scale sample
    sample = scale * sample

    #     # Variance
    features.append(np.var(sample, axis=0))

    return np.array(features).flatten()


def create_feature_set(filenames):
    x_out = []
    for file in filenames:

        features = extract_features(file, 1)

        x_out.append(features)

    return np.array(x_out)
