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
def conv_block(input, num_filters, weight_decay=WEIGHT_DECAY_VALUE):
    x = keras.layers.Conv2D(num_filters, 3, padding="same", activation='relu',
                            kernel_regularizer=l2(weight_decay))(input)
    x = keras.layers.Conv2D(num_filters, 3, padding="same", activation='relu',
                            kernel_regularizer=l2(weight_decay))(x)
    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = keras.layers.MaxPooling2D((2,2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = keras.layers.UpSampling2D(size=(2, 2))(input)
    x = keras.layers.Concatenate()([x, skip_features])
    x = keras.layers.Dropout(DROPOUT_VALUE)(x)
    x = conv_block(x, num_filters)
    return x
from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import ReduceLROnPlateau

# U-net architecture:
inp = keras.layers.Input(shape=(128, 128, 1))
s1,p1 = encoder_block(inp, 16)
s2,p2 = encoder_block(p1, 32)
s3,p3 = encoder_block(p2, 64)
b1 = conv_block(p3, 128)
d1 = decoder_block(b1, s3, 64)
d2 = decoder_block(d1, s2, 32)
d3 = decoder_block(d2, s1, 16)
output = keras.layers.Conv2D(1, kernel_size=5, padding="same", activation='relu')(d3)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE_VALUE)
new_model = keras.models.Model(inputs= inp, outputs=output)
new_model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=LR_VALUE))
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=LR_SCH_PATIENCE_VALUE)

# Train the model:
print(new_model.summary())
new_model.fit(train_images,train_true, validation_data=(valid_images,valid_true),epochs=75, batch_size=BATCH_SIZE_VALUE, callbacks=[early_stop, lr_scheduler])
new_model.save('PATH_TO_SAVE/model.keras')

