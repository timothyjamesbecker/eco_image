from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow import keras
import utils

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#BASIC CONNECTED NN------------------------------------------------------------
for label in range(1,10,1):
    #Deep CNN -------------------------------------------------------------
    gray_scale,data_agmentation,batch_size,epochs = True,False,16,100
    in_dir = '/media/data/flow_2018_small_color/'
    (train_images,train_labels,train_paths),(test_images,test_labels,test_path) = utils.load_data_advanced(in_dir,gray_scale=gray_scale)
    train_images  = train_images/255.0
    # train_labels  = np.asarray([1 if i==label else 0 for i in train_labels],dtype='uint8')
    train_labels -= 1
    test_images   = test_images/255.0
    # test_labels   = np.asarray([1 if i==label else 0 for i in test_labels],dtype='uint8')
    test_labels  -= 1
    if gray_scale:
        model = keras.Sequential([
            keras.layers.Conv1D(64,(5,),activation='relu',input_shape=train_images.shape[1:]),
            keras.layers.MaxPooling1D((2,)),
            keras.layers.Conv1D(64,(3,),activation='relu'),
            keras.layers.MaxPooling1D((2,)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv1D(128,(3,),activation='relu'),
            keras.layers.MaxPooling1D((2,)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv1D(128,(3,),activation='relu'),
            keras.layers.MaxPooling1D((2,)),
            keras.layers.Conv1D(256,(3,),activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(256,activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(6,activation='softmax')
        ])
    else:
        model = keras.Sequential([
            keras.layers.Conv2D(64, (5, 5), activation='relu', input_shape=train_images.shape[1:]),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(6, activation='softmax')
        ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    if not data_agmentation:
        model.fit(train_images,train_labels,
                  batch_size=batch_size,epochs=epochs,verbose=0)
    else:
        datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=5.0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.5,  # set range for random shear
        zoom_range=0.1,  # set range for random zoom
        channel_shift_range=0.1,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.1)

        datagen.fit(train_images)
        model.fit_generator(datagen.flow(train_images,train_labels,batch_size=batch_size),
                            epochs=epochs,validation_data=(test_images,test_labels),workers=1,verbose=0)

    test_loss,test_acc = model.evaluate(test_images,test_labels,verbose=2)
    pred_conf = model.predict(test_images,batch_size=batch_size,verbose=2)
    confusion_matrix = utils.confusion_matrix([np.argmax(x) for x in pred_conf],test_labels)
    for c in confusion_matrix:
        print('%s:%s'%(c,confusion_matrix[c]))
    print('%s CNN accuracy for classifier'%(test_acc))
