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
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, Dropout, concatenate, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers import CuDNNLSTM, Lambda, Concatenate, BatchNormalization, Flatten, regularizers, SeparableConv1D, \
    Activation, MaxPooling2D, AveragePooling2D
from keras import backend as K, models
import keras


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

data_format = 'channels_first'

concat_axis = 3

def pre_block(xm, conv1_size, conv2_size, pool_size, mcSeq):
    print('pre_block')
    base = xm
    xm0 = Conv2D(32, conv1_size, padding='same', activation="relu", name=mcSeq + "_pre_block_conv1", kernel_initializer='glorot_normal', data_format='channels_last')(base)
    xm0 = AveragePooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format='channels_last')(xm0)
    xm1 = Conv2D(32, conv2_size, padding='same', activation="relu", name=mcSeq + "_pre_block_conv2", kernel_initializer='glorot_normal', data_format='channels_last')(base)
    xm1 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format='channels_last')(xm1)
    xm = concatenate([xm0, xm1], axis=concat_axis)
    return xm


# m_block
def m_block(xm, filters_size01, filters_size02, filters_size03, conv0_size, conv1_size, conv2_size, conv3_size, mcSeq):
    print('m-block')
    base = xm
    base_xm = Conv2D(filters_size01, conv0_size, padding='same', activation="relu", name=mcSeq + "_m_block_conv0", kernel_initializer='glorot_normal', data_format='channels_last')(base)
    xm0 = Conv2D(filters_size02, conv1_size, padding='same', activation="relu", name=mcSeq + "_m_block_conv1", kernel_initializer='glorot_normal', data_format='channels_last')(base_xm)
    xm1 = Conv2D(filters_size02, conv2_size, padding='same', activation="relu", name=mcSeq + "_m_block_conv2", kernel_initializer='glorot_normal', data_format='channels_last')(base_xm)
    xm2 = Conv2D(filters_size03, conv3_size, padding='same', activation="relu", name=mcSeq + "_m_block_conv3", kernel_initializer='glorot_normal', data_format='channels_last')(base_xm)
    xm = concatenate([xm0, xm1], axis=concat_axis)
    xm = concatenate([xm, xm2], axis=concat_axis)
    # concat(xm0, xm1, xm2)
    return xm

def m_block_p(xm, conv0_size, conv1_size, conv2_size, conv3_size, pool_size, mcSeq):
    print('m_block_pool')
    base = xm
    base_xm = Conv2D(32, conv0_size, padding='same', activation="relu", name=mcSeq + "_m_block_p_conv0", kernel_initializer='glorot_normal', data_format='channels_last')(base)
    xm0 = Conv2D(48, conv1_size, padding='same', activation="relu", name=mcSeq + "_pre_block_p_conv1", kernel_initializer='glorot_normal', data_format='channels_last')(base_xm)
    xm0 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format='channels_last')(xm0)
    xm1 = Conv2D(48, conv2_size, padding='same', activation="relu", name=mcSeq + "_pre_block_p_conv2", kernel_initializer='glorot_normal', data_format='channels_last')(base_xm)
    xm1 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format='channels_last')(xm1)
    xm2 = Conv2D(32, conv3_size, padding='same', activation="relu", name=mcSeq + "_pre_block_p_conv3", kernel_initializer='glorot_normal', data_format='channels_last')(base_xm)
    xm2 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format='channels_last')(xm2)
    xm = concatenate([xm0, xm1], axis=concat_axis)
    xm = concatenate([xm, xm2], axis=concat_axis)
    return xm

def MCNet():
    in_shp = [2, 128]
    xm_input = Input(in_shp)

    xm = Reshape([2, 128, 1], input_shape=in_shp)(xm_input)
    xm = Conv2D(64, kernel_size=(3, 7), strides=(1, 1), padding='same', activation="relu", name='conv0', kernel_initializer='glorot_normal', data_format='channels_last')(xm)
    xm = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid', data_format='channels_last')(xm)

##########################################################################################################
    mcPreBName = 'mc_net01'
    pre_B_conv1_size = (1, 3)
    pre_B_conv2_size = (3, 1)
    pre_B_pool_size = (1, 2)
    xm = pre_block(xm, pre_B_conv1_size, pre_B_conv2_size, pre_B_pool_size, mcPreBName)

    jumpPool1_size = (1, 2)
    jumpStrides1_size = (1, 2)
    xm = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(xm)
    xm_tmp1 = MaxPooling2D(pool_size=jumpPool1_size, strides=jumpStrides1_size, padding='valid', data_format='channels_last')(xm)
    xm_tmp1 = Reshape([2, 8, 128])(xm_tmp1)
    # pool1----->MaxPooling2D()
    xm = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid', data_format='channels_last')(xm)

    # m_Bp1----->m_block_p
    # m_block_p(xm, conv0_size, conv1_size, conv2_size, conv3_size, pool_size, mcSeq)
    mcMBp1Name = 'mc_net02'
    m_Bp1_conv0_size = (1, 1)
    m_Bp1_conv1_size = (3, 1)
    m_Bp1_conv2_size = (1, 3)
    m_Bp1_conv3_size = (1, 1)
    m_Bp1_pool_size = (1, 2)
    xm = m_block_p(xm, m_Bp1_conv0_size, m_Bp1_conv1_size, m_Bp1_conv2_size, m_Bp1_conv3_size, m_Bp1_pool_size, mcMBp1Name)
    # add  M-block 1
    xm = keras.layers.Add()([xm, xm_tmp1])

    xm_tmp2 = xm
    # m_B1----->m_block
    # m_block(xm, conv0_size, conv1_size, conv2_size, conv3_size, mcSeq)
    mcMB1Name = 'mc_net03'
    m_B1_filter_size01 = 32
    m_B1_filter_size02 = 48
    m_B1_filter_size03 = 32
    m_B1_conv0_size = (1, 1)
    m_B1_conv1_size = (1, 3)
    m_B1_conv2_size = (3, 1)
    m_B1_conv3_size = (1, 1)
    xm = m_block(xm, m_B1_filter_size01, m_B1_filter_size02, m_B1_filter_size03, m_B1_conv0_size, m_B1_conv1_size, m_B1_conv2_size, m_B1_conv3_size, mcMB1Name)
    # add  M-block 2
    # xm =concatenate([xm, xm_tmp2], axis=concat_axis)
    xm = keras.layers.Add()([xm, xm_tmp2])

    xm_tmp3 = xm
    # poolJump1----->MaxPooling2D()
    jumpPool2_size = (1, 2)
    jumpStrides2_size = (1, 2)
    xm_tmp3_pool = MaxPooling2D(pool_size=jumpPool2_size, strides=jumpStrides2_size, padding='valid', data_format='channels_last')(xm_tmp3)
    # M-block-p
    mcMBp2Name = 'mc_net04'
    m_Bp2_conv0_size = (1, 1)
    m_Bp2_conv1_size = (1, 3)
    m_Bp2_conv2_size = (3, 1)
    m_Bp2_conv3_size = (1, 1)
    m_Bp2_pool_size = (1, 2)
    xm = m_block_p(xm, m_Bp2_conv1_size, m_Bp2_conv1_size, m_Bp2_conv2_size, m_Bp2_conv3_size, m_Bp2_pool_size, mcMBp2Name)
    # add  M-block 3
    # xm = concatenate([xm, xm_tmp3_pool], axis=concat_axis)
    xm = keras.layers.Add()([xm, xm_tmp3_pool])

    xm_tmp4 = xm
    # M-block
    mcMB2Name = 'mc_net05'
    m_B2_filter_size01 = 32
    m_B2_filter_size02 = 48
    m_B2_filter_size03 = 32
    m_B2_conv0_size = (1, 1)
    m_B2_conv1_size = (1, 3)
    m_B2_conv2_size = (3, 1)
    m_B2_conv3_size = (1, 3)
    xm = m_block(xm, m_B2_filter_size01, m_B2_filter_size02, m_B2_filter_size03, m_B2_conv0_size, m_B2_conv1_size, m_B2_conv2_size, m_B2_conv3_size, mcMB2Name)
    # add  M-block 4
    # xm = concatenate([xm, xm_tmp4], axis=concat_axis)
    xm = keras.layers.Add()([xm, xm_tmp4])
    xm_tmp5 = xm
    xm = concatenate([xm, xm_tmp4], axis=concat_axis)
    # xm = keras.layers.Add()([xm, xm_tmp4])

    ############################################################################################################

    # pool2----->avg-pool----->AveragePooling2D()
    xm = AveragePooling2D(pool_size=(2, 1), strides=(1, 2), padding='valid', data_format='channels_last')(xm)
    xm = BatchNormalization()(xm)
    xm = Flatten()(xm)
    # dense fc
    xm = Dense(11, kernel_initializer='glorot_normal', name="dense3",activation='softmax')(xm)

    # xm = Lambda(squeeze_dim, name='sqe_dim')(xm)
    model = Model(inputs=xm_input, outputs=xm)
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()
    return model


model = MCNet()


x_train, x_val, y_train, y_val = TrainDataset(1)

filename = f'MCNet'
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