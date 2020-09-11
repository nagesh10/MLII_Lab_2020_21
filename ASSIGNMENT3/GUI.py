import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical as tcg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from matplotlib import pyplot
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# RMS

model_rms = Sequential()
model_rms.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(48, 48, 3)))
model_rms.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_rms.add(MaxPooling2D((2, 2), strides=(2,2)))

model_rms.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_rms.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_rms.add(MaxPooling2D((2, 2), strides=(2,2)))

model_rms.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_rms.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_rms.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_rms.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_rms.add(MaxPooling2D((2, 2), strides=(2,2)))

model_rms.add(Flatten())

model_rms.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model_rms.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.RMSprop(
    lr=0.001,
    rho=0.9,
    epsilon=1e-07,)
model_rms.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


model_rms.load_weights('best_model_RMS.h5')

# Nadam

model = Sequential()
model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(48, 48, 3)))
model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Nadam(
   lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model.load_weights('best_model_NADAM.h5')

# SGd

model_sgd = Sequential()
model_sgd.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(48, 48, 3)))
model_sgd.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgd.add(MaxPooling2D((2, 2), strides=(2,2)))

model_sgd.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgd.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgd.add(MaxPooling2D((2, 2), strides=(2,2)))

model_sgd.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgd.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgd.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgd.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgd.add(MaxPooling2D((2, 2), strides=(2,2)))

model_sgd.add(Flatten())

model_sgd.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model_sgd.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.SGD(
    lr=0.01, nesterov=False)
model_sgd.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model_sgd.load_weights('best_model_SGD.h5')

#sgdn

model_sgdn = Sequential()
model_sgdn.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(48, 48, 3)))
model_sgdn.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgdn.add(MaxPooling2D((2, 2), strides=(2,2)))

model_sgdn.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgdn.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgdn.add(MaxPooling2D((2, 2), strides=(2,2)))

model_sgdn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgdn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgdn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgdn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_sgdn.add(MaxPooling2D((2, 2), strides=(2,2)))

model_sgdn.add(Flatten())

model_sgdn.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model_sgdn.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.SGD(
    lr=0.01, nesterov=True)
model_sgdn.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model_sgd.load_weights('best_model_SGDN.h5')

#adam

model_adam = Sequential()
model_adam.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(48, 48, 3)))
model_adam.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adam.add(MaxPooling2D((2, 2), strides=(2,2)))

model_adam.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adam.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adam.add(MaxPooling2D((2, 2), strides=(2,2)))

model_adam.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adam.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adam.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adam.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adam.add(MaxPooling2D((2, 2), strides=(2,2)))

model_adam.add(Flatten())

model_adam.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model_adam.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(
    lr=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,)
model_adam.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model_sgd.load_weights('best_model_ADAM.h5')

model_adad = Sequential()
model_adad.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(48, 48, 3)))
model_adad.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adad.add(MaxPooling2D((2, 2), strides=(2,2)))

model_adad.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adad.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adad.add(MaxPooling2D((2, 2), strides=(2,2)))

model_adad.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adad.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adad.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adad.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adad.add(MaxPooling2D((2, 2), strides=(2,2)))

model_adad.add(Flatten())

model_adad.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model_adad.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adadelta(
    lr=0.001, rho=0.95, epsilon=1e-07)
model_adad.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model_sgd.load_weights('best_model_ADADELTA.h5')

model_adag = Sequential()
model_adag.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(48, 48, 3)))
model_adag.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adag.add(MaxPooling2D((2, 2), strides=(2,2)))

model_adag.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adag.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adag.add(MaxPooling2D((2, 2), strides=(2,2)))

model_adag.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adag.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adag.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adag.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adag.add(MaxPooling2D((2, 2), strides=(2,2)))

model_adag.add(Flatten())

model_adag.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model_adag.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adagrad(
    lr=0.001,
    epsilon=1e-07,)
model_adag.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model_adag.load_weights('best_model_ADAGRAD.h5')

model_adamax = Sequential()
model_adamax.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(48, 48, 3)))
model_adamax.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adamax.add(MaxPooling2D((2, 2), strides=(2,2)))

model_adamax.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adamax.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adamax.add(MaxPooling2D((2, 2), strides=(2,2)))

model_adamax.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adamax.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adamax.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adamax.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_adamax.add(MaxPooling2D((2, 2), strides=(2,2)))

model_adamax.add(Flatten())

model_adamax.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

model_adamax.add(Dense(1, activation='sigmoid'))

opt = tf.keras.optimizers.Adamax(
    lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,)
model_adamax.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

model_adamax.load_weights('best_model_ADAMAX.h5')

st.markdown("<h1 style='text-align: center; color: black;'>IMAGE CLASSIFICATION USING CATS VS DOGS</h1>", unsafe_allow_html=True)


uploaded_file_sample = st.file_uploader("Upload a sample image", type=["png", "jpg", "jpeg"])


SIZE = 48

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

    img = cv2.resize(image, (48,48))

    img = np.expand_dims(img, 0)

    st.write('Model Input - Sample Image')

    st.image(image,use_column_width=True)

# uploaded_file_test = st.file_uploader("Upload a test image", type=["png", "jpg", "jpeg"])

# if uploaded_file_test is not None:

    
#     # creating a image object (main image)  
#     im2 = Image.open(uploaded_file_test)  
    
#     # save a image using extension 
#     im2 = im2.save("test.jpg") 

#     imaget = cv2.imread('test.jpg',1)




#     img2 = cv2.resize(imaget, (32, 32))

#     img2 = np.expand_dims(img2, 0)

#     st.write('Model Input - Test Image')
#     st.image(imaget,use_column_width=True)

classified = ['CAT','DOG']

if st.button('Predict'):

    st.markdown("<h3 style='text-align: center; color: black;'>Prediction 1: Optimizer RMSProp</h3>", unsafe_allow_html=True)

    prediction = model_rms.predict(img)

    predict = str(model_rms.predict_classes(img))

    # if [(model_rms.predict_classes(img))][[0]] == 0:

    if (model_rms.predict_classes(img)[0][0] == 1):

        st.write('Predicted Class for Sample Image: ',classified[1])
        
    
    else:

        st.write('Predicted Class for Sample Image: ',classified[0])

    history_rms=np.load('Atharva_RMS_48.npy',allow_pickle='TRUE').item()

    st.write('Training Accuracy: ',max(history_rms['accuracy'])*100,'%')
    st.write('Test Accuracy: ',max(history_rms['val_accuracy'])*100,'%')


    plt.plot(history_rms['accuracy'])
    plt.plot(history_rms['val_accuracy'])
    plt.grid(True)
    plt.title('model accuracy curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    st.pyplot()


    st.write('')
    st.markdown("<h3 style='text-align: center; color: black;'>Prediction 2: Optimizer NADAM</h3>", unsafe_allow_html=True)
    prediction = model.predict(img)

    predict = str(model.predict_classes(img))

    # if [(model_rms.predict_classes(img))][[0]] == 0:

    if (model.predict_classes(img)[0][0] == 1):

        st.write('Predicted Class for Sample Image: ',classified[1])
    
    else:

        st.write('Predicted Class for Sample Image: ',classified[0])

    history_nadam=np.load('Atharva_Nadam_48.npy',allow_pickle='TRUE').item()

    st.write('Training Accuracy: ',max(history_nadam['accuracy'])*100,'%')
    st.write('Test Accuracy: ',max(history_nadam['val_accuracy'])*100,'%')

    plt.plot(history_nadam['accuracy'])
    plt.plot(history_nadam['val_accuracy'])
    plt.grid(True)
    plt.title('model accuracy curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    st.pyplot()

    st.write('')
    st.markdown("<h3 style='text-align: center; color: black;'>Prediction 3: Optimizer ADAM</h3>", unsafe_allow_html=True)

    prediction = model_adam.predict(img)

    predict = str(model_adam.predict_classes(img))

    # if [(model_rms.predict_classes(img))][[0]] == 0:

    if (model_adam.predict_classes(img)[0][0] == 1):

        st.write('Predicted Class for Sample Image: ',classified[1])
    
    else:

        st.write('Predicted Class for Sample Image: ',classified[0])

    history_adam=np.load('Atharva_ADAM_48.npy',allow_pickle='TRUE').item()

    st.write('Training Accuracy: ',max(history_adam['accuracy'])*100,'%')
    st.write('Test Accuracy: ',max(history_adam['val_accuracy'])*100,'%')
    
    plt.plot(history_adam['accuracy'])
    plt.plot(history_adam['val_accuracy'])
    plt.grid(True)
    plt.title('model accuracy curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    st.pyplot()

    st.write('')
    st.markdown("<h3 style='text-align: center; color: black;'>Prediction 4: Optimizer SGD</h3>", unsafe_allow_html=True)

    prediction = model_sgd.predict(img)

    predict = str(model_sgd.predict_classes(img))

    # if [(model_rms.predict_classes(img))][[0]] == 0:

    if (model_sgd.predict_classes(img)[0][0] == 1):

        st.write('Predicted Class for Sample Image: ',classified[1])
    
    else:

        st.write('Predicted Class for Sample Image: ',classified[0])

    history_sgd=np.load('Atharva_SGD_48.npy',allow_pickle='TRUE').item()

    st.write('Training Accuracy: ',max(history_sgd['accuracy'])*100,'%')
    st.write('Test Accuracy: ',max(history_sgd['val_accuracy'])*100,'%')

    plt.plot(history_sgd['accuracy'])
    plt.plot(history_sgd['val_accuracy'])
    plt.grid(True)
    plt.title('model accuracy curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    st.pyplot()

    st.write('')
    st.markdown("<h3 style='text-align: center; color: black;'>Prediction 5: Optimizer SGDN</h3>", unsafe_allow_html=True)

    prediction = model_sgdn.predict(img)

    predict = str(model_sgdn.predict_classes(img))

    # if [(model_rms.predict_classes(img))][[0]] == 0:

    if (model_sgdn.predict_classes(img)[0][0] == 1):

        st.write('Predicted Class for Sample Image: ',classified[1])
    
    else:

        st.write('Predicted Class for Sample Image: ',classified[0])

    history_sgdn=np.load('Atharva_SGDN_48.npy',allow_pickle='TRUE').item()

    st.write('Training Accuracy: ',max(history_sgdn['accuracy'])*100,'%')
    st.write('Test Accuracy: ',max(history_sgdn['val_accuracy'])*100,'%')

    plt.plot(history_sgdn['accuracy'])
    plt.plot(history_sgdn['val_accuracy'])
    plt.grid(True)
    plt.title('model accuracy curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    st.pyplot()
    
    st.write('')
    st.markdown("<h3 style='text-align: center; color: black;'>Prediction 6: Optimizer ADADelta</h3>", unsafe_allow_html=True)

    prediction = model_adad.predict(img)

    predict = str(model_adad.predict_classes(img))

    # if [(model_rms.predict_classes(img))][[0]] == 0:

    if (model_adad.predict_classes(img)[0][0] == 1):

        st.write('Predicted Class for Sample Image: ',classified[1])
    
    else:

        st.write('Predicted Class for Sample Image: ',classified[0])

    history_adad=np.load('Atharva_ADADELTA_48.npy',allow_pickle='TRUE').item()

    st.write('Training Accuracy: ',max(history_adad['accuracy'])*100,'%')
    st.write('Test Accuracy: ',max(history_adad['val_accuracy'])*100,'%')

    plt.plot(history_adad['accuracy'])
    plt.plot(history_adad['val_accuracy'])
    plt.grid(True)
    plt.title('model accuracy curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    st.pyplot()
    # predictiont = model.predict(img2)

    # predictt = str(model.predict_classes(img2))

    # st.write('Predicted Class for Test Image: ',predictt)

    st.write('')
    st.markdown("<h3 style='text-align: center; color: black;'>Prediction 7: Optimizer ADAGRAD</h3>", unsafe_allow_html=True)

    prediction = model_adag.predict(img)

    predict = str(model_adag.predict_classes(img))

    # if [(model_rms.predict_classes(img))][[0]] == 0:

    if (model_adag.predict_classes(img)[0][0] == 1):

        st.write('Predicted Class for Sample Image: ',classified[1])
    
    else:

        st.write('Predicted Class for Sample Image: ',classified[0])

    history_adag=np.load('Atharva_ADAG_48.npy',allow_pickle='TRUE').item()

    st.write('Training Accuracy: ',max(history_adag['accuracy'])*100,'%')
    st.write('Test Accuracy: ',max(history_adag['val_accuracy'])*100,'%')

    plt.plot(history_adag['accuracy'])
    plt.plot(history_adag['val_accuracy'])
    plt.grid(True)
    plt.title('model accuracy curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    st.pyplot()

    st.write('')
    st.markdown("<h3 style='text-align: center; color: black;'>Prediction 8: Optimizer ADAMAX</h3>", unsafe_allow_html=True)

    prediction = model_adamax.predict(img)

    predict = str(model_adamax.predict_classes(img))

    # if [(model_rms.predict_classes(img))][[0]] == 0:

    if (model_adamax.predict_classes(img)[0][0] == 1):

        st.write('Predicted Class for Sample Image: ',classified[1])
    
    else:

        st.write('Predicted Class for Sample Image: ',classified[0])

    history_adamax=np.load('Atharva_ADAMAX_48.npy',allow_pickle='TRUE').item()

    st.write('Training Accuracy: ',max(history_adamax['accuracy'])*100,'%')
    st.write('Test Accuracy: ',max(history_adamax['val_accuracy'])*100,'%')

    plt.plot(history_adamax['accuracy'])
    plt.plot(history_adamax['val_accuracy'])
    plt.grid(True)
    plt.title('model accuracy curve')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    st.pyplot()
        

    

    

    

    # plt.plot(history['val_loss'])
    # plt.plot(history['val_accuracy'])
    # plt.grid(True)
    # plt.title('model loss curve')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')

    # fig = plt.figure(figsize=(10, 4))
    # fig, axs = plt.subplots(nrows=1, ncols=2)
    # st.pyplot(fig)

    # st.pyplot()

st.markdown("<h5 style='text-align: right; color: black;'>Atharva Gondkar, 2176032</h5>", unsafe_allow_html=True)