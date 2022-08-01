# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:09:08 2020

@author: Rain
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras import models
from keras.layers import *
from keras.models import Model
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
import scipy.io as scio
import numpy as np
from numpy import array
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras import backend as K
from keras import models
from keras.layers import *
from complexnn.conv import ComplexConv1D
from complexnn.bn import ComplexBatchNormalization
from complexnn.dense import ComplexDense
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as scio
import random
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Dropout, concatenate, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers import CuDNNLSTM, Lambda, Concatenate, Activation, Flatten, Multiply, Add, Subtract, CuDNNGRU
from keras import backend as K
import tensorflow as tf
import pandas as pd


def rotate_matrix(theta):
    m = np.zeros((2,2))
    m[0, 0] = np.cos(theta)
    m[0, 1] = -np.sin(theta)
    m[1, 0] = np.sin(theta)
    m[1, 1] = np.cos(theta)
    print(m)
    return m

def Rotate_DA(x, y):
    [N, L, C] = np.shape(x)
    x_rotate1 = np.matmul(x, rotate_matrix(np.pi/2))
    x_rotate2 = np.matmul(x, rotate_matrix(np.pi))
    x_rotate3 = np.matmul(x, rotate_matrix(3*np.pi/2))

    x_DA = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))  

    y_DA = np.tile(y, (1, 4))
    y_DA = y_DA.T
    y_DA = y_DA.reshape(-1)
    y_DA = y_DA.T
    return x_DA, y_DA

def TrainDataset(r):
    x = np.load(f'train_RML/x_r={r}.npy')
    x = x.transpose((0, 2, 1))
    y = np.load(f'train_RML/y_r={r}.npy')
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3)
    x_train, y_train = Rotate_DA(x_train, y_train)
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)
    x_train = x_train.transpose((0, 2, 1))
    x_val = x_val.transpose((0, 2, 1))
    return x_train, x_val, y_train, y_val


def TestDataset(snr):
    x = np.load(f"test_RML/x_snr={snr}.npy")
    y = np.load(f"test_RML/y_snr={snr}.npy")
    y = to_categorical(y)
    return x, y

from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Dropout, concatenate, Reshape
from keras.layers.convolutional import Conv2D, SeparableConv2D
from keras.layers import CuDNNLSTM, Lambda, Concatenate, BatchNormalization, Activation
from keras import backend as K

def cal1(x):
    y = tf.keras.backend.cos(x)
    return y

def cal2(x):
    y = tf.keras.backend.sin(x)
    return y
def PET():
    dr = 0.5  # dropout rate (%)
    input_ = Input([2, 128] + [1], name='input1')
    input1 = Input([128, 1], name='input2')
    input2 = Input([128, 1], name='input3')

    x1 = Flatten()(input_)
    x1 = Dense(1, name='fc2')(x1)
    x1 = Activation('linear')(x1)

    cos1 = Lambda(cal1)(x1)
    sin1 = Lambda(cal2)(x1)
    x11 = Multiply()([input1, cos1])
    x12 = Multiply()([input2, sin1])
    x21 = Multiply()([input2, cos1])
    x22 = Multiply()([input1, sin1])
    y1 = Add()([x11, x12])
    y2 = Subtract()([x21, x22])
    y1 = Reshape(target_shape=(128, 1), name='reshape1')(y1)
    y2 = Reshape(target_shape=(128, 1), name='reshape2')(y2)
    x11 = concatenate([y1, y2])
    x3 = Reshape(target_shape=((128, 2, 1)), name='reshape3')(x11)

    # spatial feature
    x3 = Conv2D(75, (8, 2), padding='valid', activation="relu", name="conv1_1", kernel_initializer='glorot_uniform')(
        x3)
    x3 = Conv2D(25, (5, 1), padding='valid', activation="relu", name="conv1_2", kernel_initializer='glorot_uniform')(
        x3)
    # temporal feature
    x4 = Reshape(target_shape=((117, 25)), name='reshape4')(x3)
    x4 = keras.layers.GRU(units=128)(x4)
    # x4 = Flatten()(x4)
    #
    x = Dense(11, activation='softmax', name='softmax')(x4)

    model = Model(inputs=[input_, input1, input2], outputs=x)
    return model

model = PET()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

x_train, x_val, y_train, y_val = TrainDataset(1)

x_train_I=np.expand_dims(x_train[:,0,:], axis=2)
x_val_I=np.expand_dims(x_val[:,0,:],axis=2)

x_train_Q=np.expand_dims(x_train[:,1,:], axis=2)
x_val_Q=np.expand_dims(x_val[:,1,:],axis=2)

x_train=np.expand_dims(x_train,axis=3)
x_val=np.expand_dims(x_val,axis=3)

filename = f'PET'
checkpoint = ModelCheckpoint(f"model/{filename}.hdf5", 
               verbose=1, 
               save_best_only=True)

rl = ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patience=10, min_lr=0.0000001)
hist = model.fit([x_train,x_train_I,x_train_Q], y_train,
    batch_size=128,
    epochs=200,
    verbose=2,
    validation_data = ([x_val,x_val_I,x_val_Q], y_val),
    callbacks=[checkpoint,rl]
)

train_test_list = [hist.history['loss'],hist.history['val_loss'], hist.history['acc'],hist.history['val_acc']]
train_test_array=np.array(train_test_list).T
df = pd.DataFrame(train_test_array, columns=['Training Loss','Test Loss','Training Acc','Test Acc'])
df.to_excel(f'loss/{filename}.xlsx', index=False)

snrs=range(-20, 20, 2)
for snr in snrs:
    model.load_weights(f"model/{filename}.hdf5")
    x, y = TestDataset(snr)
    x_I=np.expand_dims(x[:,0,:], axis=2)

    x_Q=np.expand_dims(x[:,1,:], axis=2)

    x=np.expand_dims(x,axis=3)
    [loss, acc] = model.evaluate([x, x_I, x_Q], y, batch_size = 1000, verbose=2)
    print(acc)

##test time
import time
x, y = TestDataset(0)
x_I=np.expand_dims(x[:,0,:], axis=2)
x_Q=np.expand_dims(x[:,1,:], axis=2)
x=np.expand_dims(x,axis=3)
[loss, acc] = model.evaluate([x, x_I, x_Q], y, batch_size = 1000, verbose=2)
BSs = [1000]
for bs in BSs:
    t1 = time.time()
    model.evaluate([x, x_I, x_Q], y, batch_size = bs, verbose=2)
    model.evaluate([x, x_I, x_Q], y, batch_size = bs, verbose=2)
    model.evaluate([x, x_I, x_Q], y, batch_size = bs, verbose=2)
    model.evaluate([x, x_I, x_Q], y, batch_size = bs, verbose=2)
    model.evaluate([x, x_I, x_Q], y, batch_size = bs, verbose=2)
    model.evaluate([x, x_I, x_Q], y, batch_size = bs, verbose=2)
    model.evaluate([x, x_I, x_Q], y, batch_size = bs, verbose=2)
    model.evaluate([x, x_I, x_Q], y, batch_size = bs, verbose=2)
    model.evaluate([x, x_I, x_Q], y, batch_size = bs, verbose=2)
    model.evaluate([x, x_I, x_Q], y, batch_size = bs, verbose=2)
    t2 = time.time()
    print((t2-t1)/5500/10)
