import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import config

def build_autoencoder():
    encoder = models.Sequential([
        layers.Input(shape=(config.INPUT_DIM,)),
        layers.Dense(12, activation='relu'),
        layers.Dense(config.LATENT_DIM, activation='relu')
    ], name="Encoder")
    
    decoder = models.Sequential([
        layers.Dense(12,activation='relu'),
        layers.Dense(config.INPUT_DIM, activation='sigmoid')
    ], name="Decoder")
    
    autoencoder = models.Sequential([encoder, decoder], name="Autoencoder_Framework")
    autoencoder.compile(optimizer='adam', loss='mae')
    return autoencoder