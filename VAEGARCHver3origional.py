

import os

import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.figsize'] = [9, 6]
import sp500

import garch
# Set random seed for reproducible results 

#from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from tensorflow import keras
import keras
import time

print(tf.__version__)
print(keras.__version__)
#                                                       #SET SEED

num_history = 90
num_predict = 30
num_latent = 15

Predict = True
if Predict:
    num_input = num_history
    num_output = num_history + num_predict
else:
    num_input = num_history
    num_output = num_history

# Load data and set up dataset ##ADD MARIO CODING HERE I'M GOING TO USE HIS SYNTHTIC VS REAL DATA WARNINGS FOR GARCH

data = sp500.sp500(num_output)
#data = garch.garch11(num_output)
r = tf.constant(data.r, dtype=tf.float32)
dataset = tf.data.Dataset.from_tensor_slices(r)
dataset = dataset.shuffle(dataset.cardinality().numpy()) ##NOT SURE HOW THE SHUFFLE WORKS AND DATES MARIOs CODE PARSES ON DATE AND THEN SORTS BY DATE
train_num = round(0.75*dataset.cardinality().numpy())
test_num = dataset.cardinality().numpy() - train_num
train_dataset = dataset.take(train_num).batch(32, drop_remainder=True)
test_dataset = dataset.skip(train_num).batch(32, drop_remainder=True)

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()





@keras.saving.register_keras_serializable()
class VAE(keras.Model):
    def __init__(self, input_dim, latent_dim, output_dim, predict, **kwargs): #        ##SOME SIMULARITIES TO MARIOs CODE
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.predict = predict

        if predict:
            dec1_size = 67
            dec2_size = 267
        else:
            dec1_size = 50
            dec2_size = 200
        
        self.encoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(shape=(input_dim,)),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim)
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(shape=(latent_dim,)),
            tf.keras.layers.Dense(units=dec1_size, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=dec2_size, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=output_dim)
            ]
        )

    def get_config(self):
        config = super().get_config()
        config.update({ "input_dim" : self.input_dim,
                        "latent_dim" : self.latent_dim,
                        "output_dim" : self.output_dim,
                        "predict" : self.predict })
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        latent_dim = config.pop("latent_dim")
        output_dim = config.pop("output_dim")
        predict = config.pop("predict")
        return cls(input_dim, latent_dim, output_dim, predict, **config)

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(32, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)  #            ##THIS SEAMS LIKE A GOOD PLACE TO ADD RL STUFF LOOK UP WAYS TO PUT RL WITH ENCODER DECODER SETUP
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def call(self, x):
        mean, logvar = self.encode(x[:,:self.input_dim])
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, z, mean, logvar




# Instantiate an optimizer
optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(-0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def compute_loss(model, x):
    #mean, logvar = model.encode(x)
    #z = model.reparameterize(mean, logvar)
    #x_hat = model.decode(z)
    x_hat, z, mean, logvar = model(x)
    logpx_z = log_normal_pdf(x, x_hat, 0.0)
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


Train = False
                                 ##HOW DO YOU KNOW THE TRAIN AND TEST SPLIT HERE
if Train:
    model = VAE(num_input, num_latent, num_output, Predict)
    epochs = 500
    for epoch in range(epochs):
        start_time = time.time()
        for i, train_x in enumerate(train_dataset):
            train_step(model, train_x, optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))

    model.save('vae_keras_dense_predict.keras')
    model.summary()


# Load trained model and test it
new_model = keras.saving.load_model('vae_keras_dense_predict.keras')
new_model.summary()

for i, test_x in enumerate(test_dataset):
    #mean, logvar = new_model.encode(test_x)
    #z = new_model.reparameterize(mean, logvar)
    #x_hat = new_model.decode(z)
    x_hat, z, mean, logvar = new_model(test_x)

    if False:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(mean.numpy().T)
        ax2.plot(logvar.numpy().T)
        ax3.plot(test_x.numpy().T, color='blue')
        ax3.plot(x_hat.numpy().T, color='red')
        plt.show()
        for x, xh in zip(test_x, x_hat):
            fig, ax = plt.subplots()
            ax.plot(x, color='blue')
            ax.plot(xh, color='red')
            ymin, ymax = plt.ylim()
            ax.plot([89.5, 89.5],[ymin, ymax], color='black')
            plt.show()
    else:
        #fig, ax = plt.subplots()
        #ax.plot(test_x[0,:], color='blue', linewidth=10)
        #for j in range(20):
        #    x_hat, z, mean, logvar = new_model(test_x[:1,:])
        #    ax.plot(np.squeeze(x_hat), color='red')
        #ymin, ymax = plt.ylim()
        #ax.plot([89.5, 89.5],[ymin, ymax], color='black')
        #plt.show()
        x_hat, z, mean, logvar = new_model(test_x[:1, :])
        error = np.mean(np.abs(test_x[0, :] - np.squeeze(x_hat)))
        print(f"Avg reconstruction error: {error:.4f}")
