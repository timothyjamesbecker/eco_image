from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
import utils

#BASIC CONNECTED NN------------------------------------------------------------
for label in [1,2,3,4,5,6]:
    # in_dir = '/media/data/flow_2018_small_gray/'
    # (train_images,train_labels),(test_images,test_labels) = utils.load_data_advanced(in_dir)
    # train_images  = train_images/255.0
    # train_labels  = np.asarray([1 if i==label else 0 for i in train_labels],dtype='uint8')
    # test_images   = test_images/255.0
    # test_labels   = np.asarray([1 if i==label else 0 for i in test_labels],dtype='uint8')
    #
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=train_images.shape[1:]),
    #     keras.layers.Dense(128, activation='relu'),
    #     keras.layers.Dense(2, activation='softmax')
    # ])
    # model.compile(optimizer='adam',
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.fit(train_images,train_labels,epochs=10,verbose=0)
    # test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=0)
    # print('%s MLP accuracy for binary classifier %s'%(test_acc,label))

    #Deep CNN -------------------------------------------------------------
    in_dir = '/media/data/flow_2018_medium_color/'
    (train_images,train_labels),(test_images,test_labels) = utils.load_data_advanced(in_dir,gray_scale=False)
    train_images  = train_images/255.0
    train_labels  = np.asarray([1 if i==label else 0 for i in train_labels],dtype='uint8')
    test_images   = test_images/255.0
    test_labels   = np.asarray([1 if i==label else 0 for i in test_labels],dtype='uint8')

    model = keras.Sequential([
        keras.layers.Conv2D(64,(3,3), activation='relu',input_shape=train_images.shape[1:]),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images,train_labels,epochs=10,verbose=0)
    test_loss, test_acc = model.evaluate(test_images,test_labels,verbose=0)
    print('%s CNN accuracy for binary classifier %s '%(test_acc,label))
