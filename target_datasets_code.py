import numpy as np
import h5py
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization


#You can load the input data here   Put the input path if needed 
path = "PATH TO DATA"

with h5py.File(path+"IR_INPUT_TRAIN.mat", 'r') as f:
    train_input = np.array(f["IR_INPUT_TRAIN"])[..., np.newaxis]

with h5py.File(path+"PRECIPITATION_TARGET_TRAIN.mat", 'r') as f:
    train_target = np.array(f["PRECIPITATION_TARGET_TRAIN"])[..., np.newaxis]

with h5py.File(path+"IR_INPUT_VALIDATION.mat", 'r') as f:
    validation_input = np.array(f["IR_INPUT_VALIDATION"])[..., np.newaxis]

with h5py.File(path+"PRECIPITATION_TARGET_VALIDATION.mat", 'r') as f:
    validation_target = np.array(f["PRECIPITATION_TARGET_VALIDATION"])[..., np.newaxis]


#Creating the convolution, encoder and decoder blocks:
def conv_block(input, num_filters):
    x = keras.layers.Conv2D(num_filters, 3, padding="same", activation='relu')(input)
    x = keras.layers.Conv2D(num_filters, 3, padding="same", activation='relu')(x)
    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = keras.layers.MaxPooling2D((2,2))(x)
    return x, p 

def decoder_block(input, skip_features, num_filters):
    x = keras.layers.UpSampling2D(size=(2, 2))(input)
    x = keras.layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

# U-net architecture: 
inp = keras.layers.Input(shape=(128, 128, 1))
s1,p1 = encoder_block(inp, 8)
BatchNormalization()
s2,p2 = encoder_block(p1, 16)
BatchNormalization()
s3,p3 = encoder_block(p2, 32)
BatchNormalization()
b1 = conv_block(p3, 64)
BatchNormalization()
d1 = decoder_block(b1, s3, 32)
BatchNormalization()
d2 = decoder_block(d1, s2, 16)
BatchNormalization()
d3 = decoder_block(d2, s1, 8)
BatchNormalization()
output = keras.layers.Conv2D(1, kernel_size=3, padding="same", activation='relu')(d3)


# We load the trained model and all the weights into memory and we prediction
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
model = keras.models.Model(inputs= inp, outputs=output) 
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
print(model.summary())

##########################
model.fit(train_input,train_target, validation_data=(validation_input,validation_target),epochs=50, steps_per_epoch=20)

model.save('/content/gdrive/MyDrive/Python/mojt_unet/model.keras')

