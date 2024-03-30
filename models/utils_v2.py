from keras.models import Model, Sequential
from keras.layers import (
    Dense,
    Layer,
    InputLayer,
    LeakyReLU,
    BatchNormalization,
    Activation,
    Dropout,
    Input,
    Flatten,
    Reshape,
)
from keras.callbacks import EarlyStopping
import numpy as np
import tensorflow as tf


def loader(files, w_size, batch):
    dataset = tf.data.Dataset.from_tensor_slices(
        tf.cast(tf.reshape(files, (files.shape[0], w_size)), tf.float32)
    )
    dataset = dataset.batch(batch)
    return dataset


class Encoder(Model):
    def __init__(self, input_size, encoded_size):
        super(Encoder, self).__init__()
        self.encoder = self.build_encoder(input_size, encoded_size)

    def build_encoder(self, input_size, encoded_size):
        inputs = Input(shape=(input_size,))
        x = Dense(input_size // 2)(inputs)
        x = LeakyReLU()(x)
        x = Dense(input_size // 4)(x)
        x = LeakyReLU()(x)
        x = Dense(encoded_size)(x)
        x = LeakyReLU()(x)
        return Model(inputs, x)

    def call(self, w):
        z = self.encoder(w)
        return z


class Decoder(Model):
    def __init__(self, encoded_size, output_size):
        super(Decoder, self).__init__()
        self.decoder = self.build_decoder(encoded_size, output_size)

    def build_decoder(self, encoded_size, output_size):
        inputs = Input(shape=(encoded_size,))
        x = Dense(output_size // 4)(inputs)
        x = LeakyReLU()(x)
        x = Dense(output_size // 2)(x)
        x = LeakyReLU()(x)
        x = Dense(output_size)(x)
        return Model(inputs, x)

    def call(self, encoded_w):
        w = self.decoder(encoded_w)
        return w


class USAD_model(Model):
    def __init__(self, w_size, z_size):
        super(USAD_model, self).__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder = Decoder(z_size, w_size)
        self.predictor = Decoder(z_size, w_size)

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
        epoch_loss1 = tf.stack(batch_losses1).mean()
        batch_losses2 = [x["val_loss2"] for x in outputs]
        epoch_loss2 = tf.stack(batch_losses2).mean()
        return {"val_loss1": epoch_loss1.item(), "val_loss2": epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(
                epoch, result["val_loss1"], result["val_loss2"]
            )
        )
