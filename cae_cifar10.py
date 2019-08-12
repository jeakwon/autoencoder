import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
%matplotlib inline

(x_train, _), (x_test, _) = cifar10.load_data()
# noise = 0.5 * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_train = x_train/255
x_test = x_test/255
x_shape = x_train[0].shape

noise_factor =0.2
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

x_test.shape

input_img = Input(shape=(x_shape))  # adapt this if using `channels_first` image data format

x = Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same')(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation=tf.nn.swish, padding='same')(x)
x = BatchNormalization()(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(16, (3, 3), activation=tf.nn.relu, padding='same')(encoded)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation=tf.nn.relu, padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation=tf.nn.relu, padding='same')(x)
x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy',)

earlystopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

autoencoder.fit(
    x_train_noisy, x_train, epochs=50, batch_size=128, validation_data=(x_test, x_test_noisy), callbacks = [earlystopping])

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    i+=1
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(x_shape))
    plt.axis('off')
    
    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(x_shape))
    plt.axis('off')
plt.show()
