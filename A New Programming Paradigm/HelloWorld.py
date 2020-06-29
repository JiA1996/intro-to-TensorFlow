import tensorflow as tf
import numpy as np
from tensorflow import keras


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = keras.Sequential(
    [
        keras.layers.Dense(1, input_shape=(1,))
    ]
)

model.compile('sgd', 'mean_squared_error')

xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10]))