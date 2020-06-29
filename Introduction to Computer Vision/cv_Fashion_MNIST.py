import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# 归一化能有效提高训练效率 如果不归一化会导致迭代变慢
training_images  = training_images / 255.0
test_images = test_images / 255.0

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.85):
      print("\nReached 85% accuracy and stop training")
      self.model.stop_training = True

callback = myCallback()

model = keras.Sequential(
    [
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dense(10,  activation=tf.nn.softmax)
    ]
)

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=8, callbacks=[callback])

model.evaluate(test_images, test_labels)


