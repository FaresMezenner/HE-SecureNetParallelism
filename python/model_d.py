import sys
import pdb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.special import softmax
from onnx import shape_inference

import onnx
import tf2onnx
import onnxruntime as rt

#
# PREPARE THE DATA
#

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 256
x_test = x_test.astype("float32") / 256
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#
# BUILD THE MODEL
#

model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(6, 5, 1, 'valid', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(16, 5, 1, 'valid', activation='relu'),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
            ]
        )

model.output_names = ['output']
model.summary()


#
# TRAIN THE MODEL
#

batch_size = 128
epochs = 5

# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.compile(loss="categorical_crossentropy", optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01), metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

#
# EVALUATE THE MODEL
#

# Evaluate the original model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
output_path = "../onnx/model_d.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
output_names = [n.name for n in model_proto.graph.output]

print("Successfully Generate ONNX Graph File.")
