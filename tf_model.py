from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore', category=Warning)
import glob
import json
import time
import argparse
import numpy as np
import utils

des="""
---------------------------------------------------
TensorFlow/Keras based hyper-param modeler
Timothy James Becker 11-04-19 to 03-03-20
---------------------------------------------------
Given input directory with partitioned labels with images inside:
in_dir/label_1, in_dir/label_2, ...
and output directory, peforms hyper parmeter search of:
model complexity, mini-batch size and epochs with several options
"""
parser = argparse.ArgumentParser(description=des.lstrip(" "),formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--in_dir',type=str,help='input directory of images\t[None]')
parser.add_argument('--out_dir',type=str,help='output directory\t[None]')
parser.add_argument('--classes',type=str,help='comma-seperated then semi-colon seperated groupings of labels\t[sperate: length of in_dir/label_*]')
parser.add_argument('--gray_scale',action='store_true',help='convert to grayscale for 1-dim colorspace\t[False]')
parser.add_argument('--split',type=float,help='0.0 to 1.0 proportion of training data to use for full hold-out\t[0.25]')
parser.add_argument('--decay',type=float,help='0.0 to 1.0 amount of L2 weight regularization decay\t[0.00001]')
parser.add_argument('--balance',type=float,help='1.0 to 10.0 amount of inter-class instance balancing\t[1.0]')
parser.add_argument('--batch_norm',action='store_true',help='use batch normalization \t[False]')
parser.add_argument('--w_reg',action='store_true',help='use weight regularization \t[False]')
parser.add_argument('--data_aug',action='store_true',help='employ data augmentation\t[False]')
parser.add_argument('--strict',action='store_true',help='sids are not used for test and train\t[False]')
parser.add_argument('--aug_workers',type=int,help='number of data augmentation worker threads\t[1]')
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
    X = args.classes.rsplit(';')
    class_labels,class_idx = {(x+1):[int(y) for y in X[x].rsplit(',')] for x in range(len(X))},{}
    for c in class_labels:
        for l in class_labels[c]: class_idx[l] = c
    classes = len(class_labels)
else:
    class_labels,class_idx = {(i+1):(i+1) for i in range(len(glob.glob(in_dir+'/label_*')))},{}
    for c in class_labels:
        for l in class_labels[c]: class_idx[l] = c
    classes = len(class_labels)
class_partition = '_'.join(['-'.join([str(x) for x in class_labels[c]]) for c in class_labels])
if args.split is not None:
    split = args.split
else:
    split = 0.25
if args.decay is not None:
    decay = args.decay
else:
    decay = 0.00001
if args.balance is not None:
    balance = args.balance
else:
    balance = 1.0
if args.aug_workers is not None:
    aug_workers = args.aug_workers
else:
    aug_workers = 1
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

print('scanning data paths...')
train_paths,test_paths = utils.partition_data_paths(in_dir,class_idx,split=split,verbose=False)
shapes = utils.get_shapes(train_paths,gray_scale=gray_scale)
print('scanning paths completed with shapes=%s'%shapes)

import tensorflow as tf
from tensorflow import keras
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e: print(e)

t_start = time.time()
best_score,best_score_path = 0.0,out_dir+'best_score.%s.json'%class_partition
if os.path.exists(best_score_path):
    with open(best_score_path,'r') as f:
        O = json.load(f)
        best_score = O['score']
H,CM,S,data,labels = {},{},{},None,None
for i in range(len(X)):
    S['params'] = X[i]
    S['params']['wreg'] = w_reg
    S['params']['data_augmentation'] = data_augmentation
    S['params']['gray_scale'] = gray_scale
    print('\nstarting iteration %s : params %s'%(i+1,X[i]))
    start = time.time()
    if not batch_norm:
        print('batch normalization not being used...')
        if not w_reg:
            print('weight regularization not being used...')
            model = keras.Sequential([
                keras.layers.Conv2D(X[i]['cmx'], (5,5), activation='relu',
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
                keras.layers.Conv2D(X[i]['cmx'], (5,5), activation='relu',
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
                  metrics=['sparse_categorical_accuracy'])
    #Deep CNN -------------------------------------------------------------
    train_paths,test_paths = utils.partition_data_paths(in_dir,class_idx,balance=balance,split=split,strict_test_sid=args.strict)
    print('%s training and %s test images being used'%(len(train_paths),len(test_paths)))
    test_generator  = utils.load_data_generator(test_paths,class_idx,
                                                batch_size=X[i]['batch_size'],
                                                gray_scale=gray_scale)
    eval_generator  = utils.load_data_generator(test_paths,class_idx,
                                                batch_size=len(test_paths),
                                                gray_scale=gray_scale)
    if not data_augmentation:
        train_generator = utils.load_data_generator(train_paths,class_idx,
                                                    batch_size=X[i]['batch_size'],
                                                    gray_scale=gray_scale)
        H = model.fit_generator(train_generator,
                                steps_per_epoch=len(train_paths)//X[i]['batch_size'],
                                validation_data=test_generator,
                                validation_steps=len(test_paths)//X[i]['batch_size'],
                                epochs=X[i]['epochs'],verbose=0,workers=1)
    else:
        imagegen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False,            # set input mean to 0 over the dataset
        samplewise_center=False,             # set each sample mean to 0
        featurewise_std_normalization=False, # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,                 # apply ZCA whitening
        zca_epsilon=1e-06,                   # epsilon for ZCA whitening
        brightness_range=(0.8,1.0),          # darken images
        channel_shift_range=0.15,            # set range for random channel shifts
        rotation_range=7.0,                 # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,               # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.05,             # randomly shift images vertically (fraction of total height)
        shear_range=0.1,                    # set range for random shear
        zoom_range=0.1,                      # set range for random zoom
        fill_mode='reflect',                 # set mode for filling points outside the input boundaries
        cval=0.0,                            # value used for fill_mode = "constant"
        horizontal_flip=True,                # randomly flip images
        vertical_flip=False,                 # randomly flip images
        rescale=None,                        # set rescaling factor (applied before any other transformation)
        preprocessing_function=None,         # set function that will be applied on each input
        data_format=None,                    # image data format, either "channels_first" or "channels_last"
        validation_split=0.0)                # fraction of images reserved for validation (strictly between 0 and 1)

        #redefine the train_geneartor to load all data into main memory to augment
        if data is None and labels is None:
            print('reading full training data into RAM...')
            train_generator = utils.load_data_generator(train_paths,class_idx,
                                                        batch_size=len(train_paths),
                                                        gray_scale=gray_scale)
            data,labels = next(train_generator) #cache training into RAM

        counts = {tuple(class_labels[c]):0 for c in class_labels}
        c_idx  = {c:tuple(class_labels[c]) for c in class_labels}
        for l in labels: counts[c_idx[l+1]] += 1
        print('loaded %s training instances with distribution %s'%(len(data),counts))
        H = model.fit_generator(imagegen.flow(data,labels,batch_size=X[i]['batch_size']),
                                epochs=X[i]['epochs'],
                                validation_data=test_generator,
                                validation_steps=len(test_paths)//X[i]['batch_size'],
                                verbose=0,workers=aug_workers)
    stop = time.time()
    pred,true = [],[]
    batch_data = next(eval_generator)
    pred += [np.argmax(x) for x in model.predict(batch_data[0])]
    true += [x for x in batch_data[1]]
    CM[i] = utils.confusion_matrix(pred,true,print_result=True)
    prec,rec,f1 = utils.metrics(CM[i])
    run_score = sum([f1[k] for k in f1])/(classes*1.0)
    print('%s Measured CNN accuracy for %s classes using %s test images'%(run_score,classes,len(pred)))

    S['score'] = run_score
    shps      = '%sx%sx%s'%(shapes[0][0],shapes[0][1],shapes[0][2])
    title     = 'class=%s cmx=%s batch=%s wreg=%s aug=%s gray=%s in:%s'%\
                (class_partition,X[i]['cmx'],X[i]['batch_size'],w_reg,data_augmentation,gray_scale,shps)
    plt_path  = 'class_%s.cmx_%s.batch_%s.wreg_%s.aug_%s.gray_%s.in_%s'%\
                (class_partition,X[i]['cmx'],X[i]['batch_size'],str(w_reg)[0],str(data_augmentation)[0],str(gray_scale)[0],shps)
    utils.plot_train_test(H.history,title,out_path=out_dir+'/acc_loss.%s.png'%plt_path)
    utils.plot_confusion_heatmap(CM[i],title,out_path=out_dir+'/conf_mat.%s.png'%plt_path)
    if best_score<S['score']:
        print('*** new best score detected, saving the model file ***')
        best_score = S['score']
        with open(best_score_path,'w') as f: json.dump(S,f)
        model_path = out_dir+'/model.'+class_partition+'.hdf5'
        model.save(model_path)
t_stop = time.time()
print('total time was %s or %s per iteration'%(round(t_stop-t_start,2),round((t_stop-t_start)/(len(CM)*1.0),2)))
