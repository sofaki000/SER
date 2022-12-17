import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Model, Input
from keras.layers import Dense
from keras_applications.densenet import layers
from tensorflow.keras import applications
from data_utilities.data_utilities import get_transformed_data

import tensorflow_hub as hub

# Loading the module
module = hub.load("https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3")


base_model = applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(40,1,1),
    include_top=False)

# we freeze base model
base_model.trainable = False

inputs = Input(shape=(40,1,1))
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. This is important for fine-tuning, as you will
# learn in a few paragraphs.
x = base_model(inputs, training=False)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = layers.GlobalAveragePooling2D()(x)
# A Dense classifier with a single unit (binary classification)
outputs = Dense(1)(x)
model = Model(inputs, outputs)

model.compile(optimizer= keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


trainX, trainY, testX, testY = get_transformed_data(dataset_number_to_load=0)
history = model.fit(trainX, trainY, epochs=20,   validation_split=0.2)

train_loss, train_acc = model.evaluate(trainX, trainY)
test_loss, test_acc = model.evaluate(testX, testY)

content_train = f'Train: Loss:{train_loss:.2f}, acc:{train_acc:.2f}\n'
content_test = f'Test: Loss:{test_loss:.2f}, acc:{test_acc:.2f}\n'
print(content_train)
print(content_test)