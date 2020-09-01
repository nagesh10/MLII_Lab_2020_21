from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical as tcg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow import keras
from matplotlib import pyplot as plt
import tensorflow as tf
import cv2
import numbers as np
import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(32,32,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

model.load_weights('CNN_cifar_10.h5')

st.markdown("<h1 style='text-align: center; color: black;'> ML ASSIGNMENT MNIST and CIFAR10</h1>", unsafe_allow_html=True)


uploaded_file_sample = st.file_uploader("Upload", type=["png", "jpg", "jpeg"])


SIZE = 192
st.set_option('deprecation.showfileUploaderEncoding', False)

if uploaded_file_sample is not None:
#data = uploaded_file
    import PIL  
    from PIL import Image  

    
    # creating a image object (main image)  
    im1 = Image.open(uploaded_file_sample)  
    
    # save a image using extension 
    im1 = im1.save("sample.jpg") 

    image = cv2.imread('sample.jpg',1)




    img = cv2.resize(image, (32, 32))

    img = np.expand_dims(img, 0)

    st.write('Model Input')
    st.image(image,use_column_width=True)

uploaded_file_test = st.file_uploader("Upload ", type=["png", "jpg", "jpeg"])

if uploaded_file_test is not None:

    
    im2 = Image.open(uploaded_file_test)  
    
    im2 = im2.save("test.jpg") 

    imaget = cv2.imread('test.jpg',1)




    img2 = cv2.resize(imaget, (32, 32))

    img2 = np.expand_dims(img2, 0)

    st.write('Input - Test Image')
    st.image(imaget,use_column_width=True)


if st.button('Predict'):

    prediction = model.predict(img)

    predict = str(model.predict_classes(img))

    st.write('Predicted Class : HORSE')

    predictiont = model.predict(img2)

    predictt = str(model.predict_classes(img2))

    st.write('Predicted Class: HORSE')



    history=np.load('history.npy',allow_pickle='TRUE').item()

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.grid(True)
    plt.title('model accuracy curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    st.pyplot()

    plt.plot(history['val_loss'])
    plt.plot(history['val_accuracy'])
    plt.grid(True)
    plt.title('model loss curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    st.pyplot()

