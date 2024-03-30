from keras.models import Model, Sequential
from keras.layers import (
    Dense,
    InputLayer,
    LeakyReLU,
    Input,
)
from keras.activations import sigmoid, relu
from keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf


def loader(files, w_size, batch):
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.cast(tf.reshape(files, (files.shape[0], w_size)), tf.float32)
    )
    dataset = dataset.batch(batch)
    print(dataset)
    return dataset


class USAD(Model):
    def __init__(self, w_size, latent_dim):
        super().__init__()
        self.encoder = Sequential(
            [
                InputLayer(input_shape=w_size),
                Dense(w_size / 2),
                LeakyReLU(),
                Dense(w_size / 4),
                LeakyReLU(),
                Dense(latent_dim),
                LeakyReLU(),
            ]
        )
        self.decoder = Sequential(
            [
                InputLayer(latent_dim),
                Dense(w_size / 4),
                LeakyReLU(),
                Dense(w_size / 2),
                LeakyReLU(),
                Dense(w_size, activation=sigmoid),
            ]
        )
        self.predictor = Sequential(
            [
                InputLayer(latent_dim),
                Dense(w_size / 4),
                LeakyReLU(),
                Dense(w_size / 2),
                LeakyReLU(),
                Dense(w_size, activation=sigmoid),
            ]
        )
        self.latent_dim = latent_dim

    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder(z)
        w2 = self.predictor(z)
        w3 = self.predictor(self.encoder(w1))
        loss1 = 1 / n * tf.reduce_mean(tf.square(batch - w1)) + (
            1 - 1 / n
        ) * tf.reduce_mean(tf.square(batch - w3))
        loss2 = 1 / n * tf.reduce_mean(tf.square(batch - w2)) - (
            1 - 1 / n
        ) * tf.reduce_mean(tf.square(batch - w3))
        return loss1, loss2

    def val_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder(z)
        w2 = self.predictor(z)
        w3 = self.predictor(self.encoder(w1))
        loss1 = 1 / n * tf.reduce_mean(tf.square(batch - w1)) + (
            1 - 1 / n
        ) * tf.reduce_mean(tf.square(batch - w3))
        loss2 = 1 / n * tf.reduce_mean(tf.square(batch - w2)) - (
            1 - 1 / n
        ) * tf.reduce_mean(tf.square(batch - w3))
        return {"val_loss1": loss1, "val_loss2": loss2}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x["val_loss1"] for x in outputs]
        epoch_loss1 = tf.reduce_mean(batch_losses1)
        batch_losses2 = [x["val_loss2"] for x in outputs]
        epoch_loss2 = tf.reduce_mean(batch_losses2)
        return {"val_loss1": epoch_loss1.numpy(), "val_loss2": epoch_loss2.numpy()}

    def epoch_end(self, epoch, result):
        print(
            f"Epoch [{epoch}], val_loss1: {result['val_loss1']:.4f}, val_loss2: {result['val_loss2']:.4f}"
        )


def evaluate(model, val_dataset, n):
    outputs = []
    for batch in val_dataset:
        output = model.val_step(batch, n)
        outputs.append(output)
    return model.validation_epoch_end(outputs)


def training(
    epochs, model, train_dataset, val_dataset, opt_func=tf.keras.optimizers.Adam
):

    history = []
    optimizer1 = opt_func()
    optimizer2 = opt_func()
    for epoch in range(epochs):
        i = 1
        for batch in train_dataset:
            with tf.GradientTape(persistent=True) as tape:
                loss1, loss2 = model.training_step(batch, epoch + 1)
            grads1 = tape.gradient(
                loss1,
                model.encoder.trainable_variables + model.decoder.trainable_variables,
            )
            grads2 = tape.gradient(
                loss2,
                model.encoder.trainable_variables + model.predictor.trainable_variables,
            )
            optimizer1.apply_gradients(
                zip(
                    grads1,
                    model.encoder.trainable_variables
                    + model.decoder.trainable_variables,
                )
            )
            optimizer2.apply_gradients(
                zip(
                    grads2,
                    model.encoder.trainable_variables
                    + model.predictor.trainable_variables,
                )
            )
            print("batch num:", i)
            i += 1
        result = evaluate(model, val_dataset, epoch + 1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def testing(model, test_dataset, alpha=0.5, beta=0.5):
    results = []
    for batch in test_dataset:
        w1 = model.decoder(model.encoder(batch))
        w2 = model.predictor(model.encoder(batch))
        results.append(
            alpha * tf.reduce_mean(tf.square(batch - w1), axis=1)
            + beta * tf.reduce_mean(tf.square(batch - w2), axis=1)
        )
    return results
