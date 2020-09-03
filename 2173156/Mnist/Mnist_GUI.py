import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical as tcg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

model = Sequential([
  Flatten(input_shape=(28, 28, 1)),
  Dense(256, activation='relu'),
  Dense(128, activation='relu'),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.load_weights('MNIST.h5')

st.markdown("<h1 style='text-align: center; color: black;'>MNIST Digit Recognizer</h1>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
SIZE = 192
st.set_option('deprecation.showfileUploaderEncoding', False)

if uploaded_file is not None:
    import PIL  
    from PIL import Image  
    im1 = Image.open(uploaded_file)  
    
    im1 = im1.save("geeks.jpg") 
    image = cv2.imread('geeks.jpg')
    os.remove("geeks.jpg")

    grey = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(grey.copy(), 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    preprocessed_digits = []
    for c in contours:
         if cv2.contourArea(c) > 20:
             
            x,y,w,h = cv2.boundingRect(c)    
            cv2.rectangle(image, (x,y), (x+w, y+h), color=(0, 255, 0), thickness=2)
            cv2.drawContours(mask, [c], 0, (255), -1)
            digit = thresh[y:y+h, x:x+w]
    
            resized_digit = cv2.resize(digit, (18,18))
            padded_digit = np.pad(resized_digit, ((5,5),(5,5)), "constant", constant_values=0)
            preprocessed_digits.append(padded_digit)

    inp = np.array(preprocessed_digits)
    st.write('Model Input')
    st.image(image,use_column_width=True)


if st.button('Predict'):
    for digit in preprocessed_digits:
        prediction = model.predict(digit.reshape(1, 28, 28, 1))

        st.image(digit.reshape(28, 28), cmap="gray",width=100)

        hard_maxed_prediction = np.zeros(prediction.shape)
        hard_maxed_prediction[0][np.argmax(prediction)] = 1
        st.write('Predicted Number:',np.argmax(prediction))
        
    history=np.load('history.npy',allow_pickle='TRUE').item()

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    st.pyplot()

    plt.plot(history['val_loss'])
    plt.plot(history['val_accuracy'])
    plt.title('model loss curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    st.pyplot()
