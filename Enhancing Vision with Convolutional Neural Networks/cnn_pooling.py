import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\nReached 90% accuracy and stop training")
      self.model.stop_training = True

model = keras.Sequential(
    [
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(2, 2),
        # keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10,  activation='softmax')
    ]
)

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(training_images, training_labels, epochs = 5, callbacks=[myCallback()])

test_loss, test_acc = model.evaluate(test_images, test_labels)

# two sets of convolution layers:
# 10000/10000 [==============================] - 1s 107us/sample - loss: 0.2643 - accuracy: 0.9041

# only one convolution layer (faster):
# 10000/10000 [==============================] - 2s 154us/sample - loss: 0.2539 - accuracy: 0.9124

# with 16 convolutions and 1 layer:
# 10000/10000 [==============================] - 1s 75us/sample - loss: 0.2498 - accuracy: 0.9127