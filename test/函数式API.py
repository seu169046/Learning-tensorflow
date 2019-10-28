import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


(train_image, train_label), (test_image, test_label)=tf.keras.datasets.fashion_mnist.load_data()

train_iamge = train_image/255
train_image1 = train_image[:50000,:]
test_image = train_image[50000:,:]
train_label_onehot = tf.keras.utils.to_categorical(train_label)
train_label_onehot1 = train_label_onehot[:50000,:]
test_label_onehot = train_label_onehot[50000:,:]


input_data = keras.Input(shape=(28, 28))
x = keras.layers.Flatten()(input_data)
x = keras.layers.Dense(64, activation = 'relu')(x)
x = keras.layers.Dense(128, activation = 'relu')(x)
y = keras.layers.Dense(10, activation = 'softmax')(x)

model = keras.Model(inputs=input_data, outputs= y)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss='categorical_crossentropy',
             metrics=['acc'])
history = model.fit(train_image1, train_label_onehot1, epochs=10, validation_data=(test_image,test_label_onehot))
