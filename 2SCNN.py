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
    x_train = x_train.transpose((0, 2, 1))
    x_val = x_val.transpose((0, 2, 1))
    return x_train, x_val, y_train, y_val


def TestDataset(snr):
    x = np.load(f"test_RML/x_snr={snr}.npy")
    y = np.load(f"test_RML/y_snr={snr}.npy")
    y = to_categorical(y)
    return x, y

dr = 0.5
def SCNN():
    in_shp = [2, 128]
    xm_input = Input(in_shp)
    xm = Reshape([128, 2], input_shape=in_shp)(xm_input)
    x1 = Conv1D(128, 16, activation='relu', padding='same')(xm)
    x2 = BatchNormalization()(x1)
    x3 = Dropout(0.5)(x2)

    x4 = SeparableConv1D(64, 8, activation='relu', padding='same')(x3)
    x5 = BatchNormalization()(x4)
    x6 = Dropout(0.5)(x5)

    x7 = Flatten()(x6)
    x8 = Dense(11)(x7)
    predicts = Activation('softmax')(x8)
    model = Model(xm_input, predicts)
    return model

model = SCNN()
model.compile(loss='categorical_crossentropy',
              optimizer='adam', 
              metrics=['accuracy'])
model.summary()

x_train, x_val, y_train, y_val = TrainDataset(1)

filename = f'SCNN'
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
    print("-----------------")
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
