import tensorflow as tf
import numpy as np
import os
from PIL import ImageOps
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cifar_path = os.path.join(BASE_DIR,"assignment2/models/cifar10")
mnist_path = os.path.join(BASE_DIR,"assignment2/models/mnist")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
def cifar(image):


    img = np.asarray(image.resize((32,32)))

    #img=img.resize((32,32))

    model = tf.keras.models.load_model(cifar_path)
    result = model.predict(img.reshape((1,32,32,3)))
    result = result[0].tolist()
    index = result.index(max(result))

    return class_names[index]


def mnist(image):

    image  = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    img = np.asarray(image.resize((28,28)))

    #img=img.resize((32,32))

    model = tf.keras.models.load_model(mnist_path)
    result = model.predict(img.reshape((1,784)))
    result = result[0].tolist()
    index = result.index(max(result))
    print(index , result)

    return index + 1
