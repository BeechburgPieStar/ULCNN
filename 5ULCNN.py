# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:09:08 2020

@author: Rain
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
n_neuron = 16
n_mobileunit = 6
ks = 5

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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as scio
import random


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
    return x_train, x_val, y_train, y_val


def TestDataset(snr):
    x = np.load(f"test_RML/x_snr={snr}.npy")
    x = x.transpose((0, 2, 1))
    y = np.load(f"test_RML/y_snr={snr}.npy")
    y = to_categorical(y)
    return x, y

def channel_shuffle(x):
    in_channels, D = K.int_shape(x)[1:]
    channels_per_group = in_channels // 2
    pre_shape = [-1, 2, channels_per_group, D]
    dim = (0, 2, 1, 3)
    later_shape = [-1, in_channels, D]

    x = Lambda(lambda z: K.reshape(z, pre_shape))(x)
    x = Lambda(lambda z: K.permute_dimensions(z, dim))(x)  
    x = Lambda(lambda z: K.reshape(z, later_shape))(x)

    return x

def dwconv_mobile(x, neurons):
    x = SeparableConv1D(int(2*neurons), ks, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = channel_shuffle(x)
    return x

def channelattention(x):

    x_GAP  = GlobalAveragePooling1D()(x)
    x_GMP  = GlobalMaxPooling1D()(x)
    channel = K.int_shape(x_GAP)[1]

    share_Dense1 = Dense(int(channel/16), activation = 'relu')
    share_Dense2 = Dense(channel)

    x_GAP = Reshape((1, channel))(x_GAP)
    x_GAP = share_Dense1(x_GAP)
    x_GAP = share_Dense2(x_GAP)

    
    x_GMP = Reshape((1, channel))(x_GMP)
    x_GMP = share_Dense1(x_GMP)
    x_GMP = share_Dense2(x_GMP)

    x_Mask = Add()([x_GAP, x_GMP])
    x_Mask = Activation('sigmoid')(x_Mask)

    x = Multiply()([x, x_Mask])
    return x

x_input = Input(shape=[128, 2])
x = ComplexConv1D(n_neuron, ks, padding='same')(x_input)
x = ComplexBatchNormalization()(x)
x = Activation('relu')(x)

for i in range(n_mobileunit):
    x = dwconv_mobile(x, n_neuron)
    x = channelattention(x)
    if i==3:
        f4 = GlobalAveragePooling1D()(x)
    if i==4:
        f5 = GlobalAveragePooling1D()(x)
    if i==5:
        f6 = GlobalAveragePooling1D()(x)

f = Add()([f4, f5])
f = Add()([f,f6])

f = Dense(11)(f)
c = Activation('softmax', name='modulation')(f)

model = Model(inputs = x_input, outputs=c)

model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()


x_train, x_val, y_train, y_val = TrainDataset(1)

filename = f'ULCNN_MN={n_mobileunit}_N={n_neuron}_KS={ks}'
checkpoint = ModelCheckpoint(f"model/{filename}.hdf5", 
               verbose=1, 
               save_best_only=True)

rl = ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patience=10, min_lr=0.0000001)
hist = model.fit(x_train, y_train,
    batch_size=128,
    epochs=200,
    verbose=2,
    validation_data = (x_val, y_val),
    callbacks=[checkpoint, rl]
)

train_test_list = [hist.history['loss'],hist.history['val_loss'], hist.history['acc'],hist.history['val_acc']]
train_test_array=np.array(train_test_list).T
df = pd.DataFrame(train_test_array, columns=['Training Loss','Test Loss','Training Acc','Test Acc'])
df.to_excel(f'loss/{filename}.xlsx', index=False)


snrs=range(-20, 20, 2)
for snr in snrs:
    model.load_weights(f"model/{filename}.hdf5")
    x, y = TestDataset(snr)
    [loss, acc] = model.evaluate(x, y, batch_size = 1000, verbose=2)
    print(acc)

##test time
import time

x, y = TestDataset(0)
[loss, acc] = model.evaluate(x, y, batch_size = 1000, verbose=2)
BSs = [1000]
for bs in BSs:
    t1 = time.time()
    model.evaluate(x, y, batch_size = bs, verbose=2)
    model.evaluate(x, y, batch_size = bs, verbose=2)
    model.evaluate(x, y, batch_size = bs, verbose=2)
    model.evaluate(x, y, batch_size = bs, verbose=2)
    model.evaluate(x, y, batch_size = bs, verbose=2)
    model.evaluate(x, y, batch_size = bs, verbose=2)
    model.evaluate(x, y, batch_size = bs, verbose=2)
    model.evaluate(x, y, batch_size = bs, verbose=2)
    model.evaluate(x, y, batch_size = bs, verbose=2)
    model.evaluate(x, y, batch_size = bs, verbose=2)
    t2 = time.time()
    print((t2-t1)/5500/10)