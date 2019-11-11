from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import time
import argparse
import numpy as np
import utils

des="""
---------------------------------------------------
TensorFlow/Keras based hyper-param modeler
Timothy James Becker 11-04-19 to 11-10-19
---------------------------------------------------
Given input directory with partitioned labels with images inside:
in_dir/label_1, in_dir/label_2, ...
and output directory, peforms hyper parmeter search of:
model complexity, mini-batch size and epochs.
"""
parser = argparse.ArgumentParser(description=des.lstrip(" "),formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--in_dir',type=str,help='input directory of images\t[None]')
parser.add_argument('--out_dir',type=str,help='output directory\t[None]')
parser.add_argument('--classes',type=int,help='number of classes to use for training\t[length of in_dir/label_*]')
parser.add_argument('--gray_scale',action='store_true',help='convert to grayscale for 1-dim colorspace\t[False]')
parser.add_argument('--split',type=float,help='0.0 to 1.0 proportion of training data to use for full hold-out\t[0.25]')
parser.add_argument('--decay',type=float,help='0.0 to 1.0 amount of L2 weight regularization decay\t[0.001]')
parser.add_argument('--batch_norm',action='store_true',help='use batch normalization \t[False]')
parser.add_argument('--w_reg',action='store_true',help='use weight regularization \t[False]')
parser.add_argument('--data_aug',action='store_true',help='employ data augmentation\t[False]')
parser.add_argument('--hyper',type=str,help='semi-colon then comma seperated hyper parameter search cmx;batch_size,epochs\t[8;8;10]')
args = parser.parse_args()

if args.in_dir is not None:
    in_dir = args.in_dir
else:
    raise IOError
if args.out_dir is not None:
    out_dir = args.out_dir
    if not os.path.exists(out_dir): os.mkdir(out_dir)
else:
    raise IOError
if args.classes is not None:
    classes = args.classes
else:
    classes = 2
if args.split is not None:
    split = args.split
else:
    split = 0.25
if args.decay is not None:
    decay = args.decay
else:
    decay = 0.001
if args.hyper is not None:
    [cmx,batch_size,epochs] = [[int(y) for y in x.split(',')] for x in args.hyper.split(';')]
else:
    cmx,batch_size,epochs = [8],[8],[10]
batch_norm        = args.batch_norm
w_reg             = args.w_reg
data_augmentation = args.data_aug
gray_scale        = args.gray_scale

X,x = {},0
for i in range(len(epochs)):          #epochs
    for j in range(len(batch_size)):  #batch_size
        for k in range(len(cmx)):     #cmx
            X[x] = {'cmx':cmx[k],'batch_size':batch_size[j],'epochs':epochs[i]}
            x += 1
train_paths,test_paths = utils.partition_data_paths(in_dir,split=split,verbose=False)
shapes = utils.get_shapes(train_paths,gray_scale=gray_scale)

import tensorflow as tf
from tensorflow import keras
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

t_start = time.time()
H,CM = {},{}
for i in range(len(X)):
    print('\nstarting iteration %s : params %s'%(i+1,X[i]))
    start = time.time()
    if not batch_norm:
        print('batch normalization not being used...')
        if not w_reg:
            print('weight regularization not being used...')
            model = keras.Sequential([
                keras.layers.Conv2D(X[i]['cmx'], (5, 5), activation='relu',
                                    input_shape=shapes[0]),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(X[i]['cmx'], (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(2*X[i]['cmx'], (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(2*X[i]['cmx'], (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(4*X[i]['cmx'], (3, 3), activation='relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(8 * X[i]['cmx'], activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(classes, activation='softmax')
            ])
        else:
            print('weight regularization being used...')
            model = keras.Sequential([
                keras.layers.Conv2D(X[i]['cmx'],(5,5),activation='relu',
                                    input_shape=shapes[0],
                                    kernel_regularizer=keras.regularizers.l2(l=decay)),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Conv2D(X[i]['cmx'],(3,3),activation='relu',
                                    kernel_regularizer=keras.regularizers.l2(l=decay)),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(2*X[i]['cmx'],(3,3),activation='relu',
                                    kernel_regularizer=keras.regularizers.l2(l=decay)),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(2*X[i]['cmx'],(3,3),activation='relu',
                                    kernel_regularizer=keras.regularizers.l2(l=decay)),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(4*X[i]['cmx'],(3,3),activation='relu',
                                    kernel_regularizer=keras.regularizers.l2(l=decay)),
                keras.layers.Flatten(),
                keras.layers.Dense(8*X[i]['cmx'], activation='relu',
                                   kernel_regularizer=keras.regularizers.l2(l=decay)),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(classes, activation='softmax')
            ])
    else:
        print('batch normalization being used...')
        if not w_reg:
            print('weight regularization not being used...')
            model = keras.Sequential([
                keras.layers.Conv2D(X[i]['cmx'], (5, 5), activation='relu',
                                    input_shape=shapes[0]),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(X[i]['cmx'], (3, 3), use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(2*X[i]['cmx'], (3, 3), use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(2*X[i]['cmx'], (3, 3), use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(4*X[i]['cmx'], (3, 3), use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(8*X[i]['cmx'], use_bias=False),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(classes, activation='softmax')
            ])
        else:
            print('weight regularization being used...')
            model = keras.Sequential([
                keras.layers.Conv2D(X[i]['cmx'],(5,5),activation='relu',
                                    input_shape=shapes[0],
                                    kernel_regularizer=keras.regularizers.l2(l=decay)),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Conv2D(X[i]['cmx'],(3,3),use_bias=False,
                                    kernel_regularizer=keras.regularizers.l2(l=decay)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(2*X[i]['cmx'],(3,3),use_bias=False,
                                    kernel_regularizer=keras.regularizers.l2(l=decay)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(2*X[i]['cmx'],(3,3),use_bias=False,
                                    kernel_regularizer=keras.regularizers.l2(l=decay)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.MaxPooling2D((2,2)),
                keras.layers.Dropout(0.25),
                keras.layers.Conv2D(4*X[i]['cmx'],(3,3),use_bias=False,
                                    kernel_regularizer=keras.regularizers.l2(l=decay)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Flatten(),
                keras.layers.Dense(8*X[i]['cmx'], use_bias=False,
                                   kernel_regularizer=keras.regularizers.l2(l=decay)),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(classes, activation='softmax')
            ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    #Deep CNN -------------------------------------------------------------
    train_paths,test_paths = utils.partition_data_paths(in_dir,split=split)

    train_generator = utils.load_data_generator(train_paths,
                                                batch_size=X[i]['batch_size'],
                                                gray_scale=gray_scale)
    test_generator  = utils.load_data_generator(test_paths,
                                                batch_size=X[i]['batch_size'],
                                                gray_scale=gray_scale)
    eval_generator = utils.load_data_generator(test_paths,
                                               batch_size=X[i]['batch_size'],
                                               gray_scale=gray_scale)
    if not data_augmentation:
        H[i] = model.fit_generator(train_generator,
                                   steps_per_epoch=len(train_paths)/X[i]['batch_size'],
                                   validation_data=test_generator,
                                   validation_steps=len(test_paths)/X[i]['batch_size'],
                                   epochs=X[i]['epochs'],verbose=0,workers=1)
    else:
        datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=10.0,  # randomly rotate images in the range (degrees, 0 to 180)
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
        validation_split=0.0)

        #?

    stop = time.time()
    test_loss,test_acc = model.evaluate_generator(eval_generator,steps=len(test_paths)/X[i]['batch_size'],
                                                  verbose=2,workers=1)
    pred_conf = model.predict_generator(test_generator,steps=len(test_paths)/X[i]['batch_size'],
                                        verbose=2,workers=1)

    CM[i] = utils.confusion_matrix([np.argmax(x) for x in pred_conf],
                                   utils.get_labels(test_paths),print_result=True)
    print('%s CNN accuracy for %s classes in %s training sec'%(test_acc,classes,round(stop-start,2)))
    shps      = '%sx%sx%s'%(shapes[0][0],shapes[0][1],shapes[0][2])
    title     = 'cmx=%s batch=%s gray=%s in:%s'%(X[i]['cmx'],X[i]['batch_size'],gray_scale,shps)
    plt_path  = 'cmx_%s-batch_%s-gray_%s-in_%s'%(X[i]['cmx'],X[i]['batch_size'],gray_scale,shps)
    utils.plot_train_test(H[i].history,title,out_path=out_dir+'/acc_loss.%s.png'%plt_path)
    utils.plot_confusion_heatmap(CM[i],title,out_path=out_dir+'/conf_mat.%s.png'%plt_path)
t_stop = time.time()
print('total time was %s or %s per iteration'%(round(t_stop-t_start,2),round((t_stop-t_start)/(len(H)*1.0),2)))

#BASIC CONNECTED NN------------------------------------------------------------
# for iteration in range(10):
#     #Deep CNN -------------------------------------------------------------
#     gray_scale,data_agmentation,label,batch_size,epochs = True,True,1,16,10
#     in_dir = '/media/data/flow_2018_small_color/'
#     (train_images,train_labels,train_paths),(test_images,test_labels,test_path) = utils.load_data_advanced(in_dir,gray_scale=gray_scale)
#     train_images  = train_images/255.0
#     #train_labels  = np.asarray([1 if i==label else 0 for i in train_labels],dtype='uint8')
#     train_labels -= 1
#     test_images   = test_images/255.0
#     #test_labels   = np.asarray([1 if i==label else 0 for i in test_labels],dtype='uint8')
#     test_labels  -= 1
#
#     model = keras.Sequential([
#         keras.layers.Conv2D(64,(5,5),activation='relu',
#                             input_shape=train_images.shape[1:],
#                             data_format="channels_last"),
#         keras.layers.MaxPooling2D((2,2)),
#         keras.layers.Conv2D(64,(3,3),activation='relu'),
#         keras.layers.MaxPooling2D((2,2)),
#         keras.layers.Dropout(0.25),
#         keras.layers.Conv2D(128,(3,3),activation='relu'),
#         keras.layers.MaxPooling2D((2,2)),
#         keras.layers.Dropout(0.25),
#         keras.layers.Conv2D(128,(3,3),activation='relu'),
#         keras.layers.MaxPooling2D((2,2)),
#         keras.layers.Conv2D(256,(3,3),activation='relu'),
#         keras.layers.Flatten(),
#         keras.layers.Dense(256, activation='relu'),
#         keras.layers.Dropout(0.5),
#         keras.layers.Dense(6, activation='softmax')
#     ])
#     model.compile(optimizer='adam',
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#     if not data_agmentation:
#         model.fit(train_images,train_labels,
#                   batch_size=batch_size,epochs=epochs,verbose=0)
#     else:
#         datagen = keras.preprocessing.image.ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         zca_epsilon=1e-06,  # epsilon for ZCA whitening
#         rotation_range=10.0,  # randomly rotate images in the range (degrees, 0 to 180)
#         # randomly shift images horizontally (fraction of total width)
#         width_shift_range=0.1,
#         # randomly shift images vertically (fraction of total height)
#         height_shift_range=0.1,
#         shear_range=0.5,  # set range for random shear
#         zoom_range=0.1,  # set range for random zoom
#         channel_shift_range=0.1,  # set range for random channel shifts
#         # set mode for filling points outside the input boundaries
#         fill_mode='nearest',
#         cval=0.,  # value used for fill_mode = "constant"
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=False,  # randomly flip images
#         # set rescaling factor (applied before any other transformation)
#         rescale=None,
#         # set function that will be applied on each input
#         preprocessing_function=None,
#         # image data format, either "channels_first" or "channels_last"
#         data_format=None,
#         # fraction of images reserved for validation (strictly between 0 and 1)
#         validation_split=0.0)
#
#         H[iteration] = model.fit_generator(datagen.flow(train_images,train_labels,batch_size=batch_size),
#                                            epochs=epochs,validation_data=(test_images,test_labels),workers=1,verbose=0)
#
#     test_loss,test_acc = model.evaluate(test_images,test_labels,verbose=2)
#     pred_conf = model.predict(test_images,batch_size=batch_size,verbose=2)
#     confusion_matrix = utils.confusion_matrix([np.argmax(x) for x in pred_conf],test_labels)
#     for c in sorted(confusion_matrix,key=lambda x: (x[0],x[1])):
#         print('%s:%s'%(c,round(confusion_matrix[c],2)))
#     print('%s CNN accuracy for classifier'%(test_acc))
