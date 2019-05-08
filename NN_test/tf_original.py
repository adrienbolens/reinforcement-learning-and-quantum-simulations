import tensorflow as tf
from tensorflow import keras

#  import numpy as np
import matplotlib as m
m.use('TkAgg')
#  import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) =\
    fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

#  f, ax = plt.subplots(5, 5, figsize=(10, 10))
#  ax = ax.flat
#  for i in range(25):
#      axi = ax[i]
#      axi.set_xticks([])
#      axi.set_yticks([])
#      axi.grid(False)
#      axi.imshow(train_images[i], cmap=plt.cm.binary)
#      axi.set_xlabel(class_names[train_labels[i]])
#  plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)
#  test_loss, test_acc = model.evaluate(test_images, test_labels)
#  print('Test accuracy:', test_acc)
#  predictions = model.predict(test_images)
