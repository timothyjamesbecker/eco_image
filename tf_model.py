#!/usr/env/bin/python3
from __future__ import absolute_import, division, print_function, unicode_literals #python 3.6+
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore',category=Warning)
import glob
import json
import time
import argparse
import numpy as np
import utils
import cv2
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e: print(e)

class DataGen(tf.keras.utils.Sequence):
    def __init__(self,data_path,class_idx,batch_size=64,sample_random=True,sample_batch=8,
                 gray_scale=False,norm=True,shuffle=True,offset=-1,verbose=False):
        self.data_path=data_path
        self.class_idx=class_idx
        self.batch=batch_size
        self.gray_scale=gray_scale
        self.norm=norm
        self.shuffle = shuffle
        self.offset=offset
        self.verbose=verbose
        self.sample_batch=sample_batch
        self.sample_index=0
        self.sample_random=sample_random
        self.step  = 0
        self.epoch = 0
        self.start = time.time()
        self.stop  = time.time()
        self.prep_data()

    def __len__(self):
        return int(np.ceil(len(self.X)/float(self.batch)))

    def __getitem__(self,idx):
        self.step += 1
        batch_x = self.X[idx*self.batch:(idx+1)*self.batch]
        batch_y = self.Y[idx*self.batch:(idx+1)*self.batch]
        if self.verbose: print('--==:: STEP %s ::==--'%self.step)
        return np.array(batch_x),np.array(batch_y)

    def on_epoch_end(self):
        self.step   = 0
        self.epoch += 1
        msg = '-----====||| EPOCH %s: %s tensor in %s sec |||====-----'
        print(msg%(str(self.epoch).rjust(2),str(self.X.shape).rjust(2),round(self.stop-self.start,2)))
        self.sample_index += self.sample_batch
        self.prep_data()

    def prep_data(self):
        self.start = time.time()

        # sub sample the data_path elements according to the sample_batch parameter...
        # if self.sample_random:
        #     sample_data_path = sorted(list(np.random.choice(self.data_path,min(len(self.sms_list),self.sample_batch),replace=False)))
        # else:
        #     sample_data_path = self.sms_list[self.sample_index:(self.sample_index+min(self.sample_batch,len(self.sms_list)))]

        local_data,local_labels,n_l = [],[],len(self.data_path)
        idx = np.random.choice(range(n_l),size=min(self.batch,n_l),replace=False)
        if gray_scale:
            for i in idx:
                lab     = int(self.data_path[i].rsplit('label_')[-1].rsplit('/')[0])
                local_labels += [class_idx[lab]+self.offset]
                local_data   += [cv2.imread(self.data_path[i],cv2.IMREAD_GRAYSCALE)]
            Y = np.asarray(local_labels,dtype='uint8')
            (h,w) = local_data[0].shape
            X = np.ndarray((self.batch,h,w,1),dtype='uint8')
            for i in range(self.batch): X[i,:,:,0] = local_data[i]
        else:
            for i in idx:
                lab     = int(self.data_path[i].rsplit('label_')[-1].rsplit('/')[0])
                local_labels += [class_idx[lab]+self.offset]
                local_data   += [cv2.imread(self.data_path[i])]
            Y = np.asarray(local_labels,dtype='uint8')
            (h,w,c) = local_data[0].shape
            X = np.ndarray((self.batch,h,w,c),dtype='uint8')
            for i in range(self.batch): X[i,:,:,:] = local_data[i]
        if self.norm: X = X/255.0
        self.indices = np.arange(len(X))
        if self.shuffle: np.random.shuffle(self.indices)
        self.X,self.Y = X[self.indices],Y[self.indices]
        self.stop = time.time()
        if self.verbose: print('Finished loading data for Epoch=%s'%(self.epoch+1))

if __name__ == '__main__':

    des="""
    ---------------------------------------------------
    TensorFlow/Keras based hyper-param modeler
    Timothy James Becker 11-04-19 to 11-12-21
    ---------------------------------------------------
    Given input directory with partitioned labels with images inside:
    in_dir/label_1, in_dir/label_2, ...
    and output directory, peforms hyper parmeter search of:
    model complexity, mini-batch size and epochs with several options
    """
    parser = argparse.ArgumentParser(description=des.lstrip(" "),formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--in_dir',type=str,help='input directory of images\t[None]')
    parser.add_argument('--out_dir',type=str,help='output directory\t[None]')
    parser.add_argument('--classes',type=str,help='comma-seperated then semi-colon seperated groupings of labels\t[seperate: length of in_dir/label_*]')
    parser.add_argument('--gray_scale',action='store_true',help='convert to grayscale for 1-dim colorspace\t[False]')
    parser.add_argument('--split',type=float,help='0.0 to 1.0 proportion of training data to use for full hold-out\t[0.25]')
    parser.add_argument('--balance',type=str,help='1.0 to 10.0 amount of inter-class instance balancing\t[1.0]')
    parser.add_argument('--strict',action='store_true',help='sids are not used for test and train\t[False]')
    parser.add_argument('--seed', type=int, help='integer seed for random splits, etc\t[auto generated]')
    parser.add_argument('--hyper',type=str,help='semi-colon then comma seperated hyper parameter search cmx;batch_size,epochs\t[8;8;10]')
    parser.add_argument('--level',type=str,help='1 to 8 levels of CNN layers\t[4]')
    parser.add_argument('--pool',type=str,help='pooling used 2 to 8\t[2]')
    parser.add_argument('--kf',type=str,help='kernel 3,5,7\t[3]')
    parser.add_argument('--decay', type=str, help='0.0 to 1.0 amount of L2 weight regularization decay\t[1e-5]')
    parser.add_argument('--drop', type=str, help='0.0 to 1.0 inter layer dropout\t[0.2]')
    parser.add_argument('--gpu_num',type=int,help='pick one of your available logical gpus\t[None]')
    #
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
    if args.seed is not None:
        seed = args.seed
    else:
        seed = None
    if args.drop is not None:
        drop = [float(x) for x in args.drop.split(',')]
    else:
        drop = [0.2]
    if args.decay is not None:
        decay = [float(x) for x in args.decay.split(',')]
    else:
        decay = [1e-5]
    if args.pool is not None:
        pool = [int(x) for x in args.pool.split(',')]
    else:
        pool = [2]
    if args.level is not None:
        level = [int(x) for x in args.level.split(',')]
    else:
        level = [2]
    if args.kf is not None:
        kf = [int(x) for x in args.kf.split(',')]
    else:
        kf = [2]
    if args.balance is not None:
        balance = [float(x) for x in args.balance.split(',')]
    else:
        balance = [None]
    if args.hyper is not None:
        [cmx,batch_size,epochs] = [[int(y) for y in x.split(',')] for x in args.hyper.split(';')]
    else:
        cmx,batch_size,epochs = [4],[16],[10]
    gray_scale        = args.gray_scale
    if args.gpu_num is not None: gpu_num = args.gpu_num
    else:                        gpu_num = 0

    X,x = {},0
    for e in range(len(epochs)):
        for b in range(len(batch_size)):
            for c in range(len(cmx)):
                for l in range(len(level)):
                    for k in range(len(kf)):
                        for p in range(len(pool)):
                            for d in range(len(decay)):
                                for r in range(len(drop)):
                                    for n in range(len(balance)):
                                        X[x] = {'cmx':cmx[c],'batch_size':batch_size[b],'epochs':epochs[e],
                                                'kf':kf[k],'level':level[l],'pool':pool[p],'split':split,'balance':balance[n],
                                                'strict':args.strict,'gray_scale':gray_scale,'decay':decay[d],'drop':drop[r]}
                                        x += 1
    print('scanning data paths...')
    train_paths,test_paths = utils.partition_data_paths(in_dir,class_idx,split=split,seed=seed,verbose=False)
    shapes = utils.get_shapes(train_paths,gray_scale=gray_scale)
    print('scanning paths completed with shapes=%s'%shapes)

    t_start = time.time()
    best_score,best_score_path = 0.0,out_dir+'best_score.%s.json'%class_partition
    if os.path.exists(best_score_path):
        with open(best_score_path,'r') as f:
            O = json.load(f)
            best_score = O['score']
    H,CM,S,data,labels = {},{},{},None,None

    with tf.device('/gpu:%s'%gpu_num):
        for i in range(len(X)):
            try:
                if seed is None: r_seed = int(np.random.get_state()[1][0])
                else:            r_seed = seed
                S['params'] = X[i]
                S['params']['seed'] = r_seed
                print('starting iteration %s : params %s'%(i+1,X[i]))
                print('using class labels: %s'%class_idx)
                print('split=%s, balance=%s, strict=%s, seed=%s'%(split,X[i]['balance'],args.strict,r_seed))
                start = time.time()

                model = tf.keras.Sequential()
                model.add(tf.keras.layers.Conv2D(X[i]['cmx'],(X[i]['kf'],X[i]['kf']),
                                                 activation='relu',input_shape=shapes[0],
                                                 padding='same',
                                                 kernel_regularizer=tf.keras.regularizers.l2(l=X[i]['decay'])))
                if X[i]['level'] >= 2:
                    model.add(tf.keras.layers.MaxPooling2D(pool_size=(X[i]['pool'],X[i]['pool'])))
                    model.add(tf.keras.layers.Dropout(X[i]['drop']))
                    model.add(tf.keras.layers.Conv2D(X[i]['cmx'],(X[i]['kf'],X[i]['kf']),activation='relu',
                                                     padding='same',
                                                     kernel_regularizer=tf.keras.regularizers.l2(l=X[i]['decay'])))
                    if X[i]['level'] >= 3:
                        model.add(tf.keras.layers.Dropout(X[i]['drop']))
                        model.add(tf.keras.layers.Conv2D(2*X[i]['cmx'],(X[i]['kf'],X[i]['kf']),activation='relu',
                                                         padding='same',
                                                         kernel_regularizer=tf.keras.regularizers.l2(l=X[i]['decay'])))
                        if X[i]['level'] >= 4:
                            model.add(tf.keras.layers.MaxPooling2D(pool_size=(X[i]['pool'],X[i]['pool'])))
                            model.add(tf.keras.layers.Dropout(X[i]['drop']))
                            model.add(tf.keras.layers.Conv2D(2*X[i]['cmx'],(X[i]['kf'],X[i]['kf']),activation='relu',
                                                             padding='same',
                                                             kernel_regularizer=tf.keras.regularizers.l2(l=X[i]['decay'])))
                            if X[i]['level'] >= 5:
                                model.add(tf.keras.layers.Dropout(X[i]['drop']))
                                model.add(tf.keras.layers.Conv2D(4*X[i]['cmx'],(X[i]['kf'],X[i]['kf']),activation='relu',
                                                                 padding='same',
                                                                 kernel_regularizer=tf.keras.regularizers.l2(l=X[i]['decay'])))
                                if X[i]['level'] >= 6:
                                    model.add(tf.keras.layers.MaxPooling2D(pool_size=(X[i]['pool'],X[i]['pool'])))
                                    model.add(tf.keras.layers.Dropout(X[i]['drop']))
                                    model.add(tf.keras.layers.Conv2D(4*X[i]['cmx'],(X[i]['kf'],X[i]['kf']),activation='relu',
                                                                     padding='same',
                                                                     kernel_regularizer=tf.keras.regularizers.l2(l=X[i]['decay'])))
                                    if X[i]['level'] >= 7:
                                        model.add(tf.keras.layers.Dropout(X[i]['drop']))
                                        model.add(tf.keras.layers.Conv2D(2*X[i]['cmx'],(X[i]['kf'],X[i]['kf']),activation='relu',
                                                                         padding='same',
                                                                         kernel_regularizer=tf.keras.regularizers.l2(l=X[i]['decay'])))
                                        if X[i]['level'] >= 8:
                                            model.add(tf.keras.layers.MaxPooling2D(pool_size=(X[i]['pool'],X[i]['pool'])))
                                            model.add(tf.keras.layers.Dropout(X[i]['drop']))
                                            model.add(tf.keras.layers.Conv2D(2*X[i]['cmx'],(X[i]['kf'],X[i]['kf']),activation='relu',
                                                                             padding='same',
                                                                             kernel_regularizer=tf.keras.regularizers.l2(l=X[i]['decay'])))
                model.add(tf.keras.layers.Flatten())
                if X[i]['cmx']>=4:
                    model.add(tf.keras.layers.Dense(X[i]['cmx'], activation='relu',
                                                    kernel_regularizer=tf.keras.regularizers.l2(l=X[i]['decay'])))
                    model.add(tf.keras.layers.Dropout(2*X[i]['drop']))
                model.add(tf.keras.layers.Dense(classes, activation='softmax',dtype=np.float32))
                model.compile(loss='sparse_categorical_crossentropy',
                              optimizer='adam',
                              metrics=['sparse_categorical_accuracy'])

                # seed=r_seed
                # balance=X[i]['balance']
                # strict_test_sid=args.strict
                # verbose=True
                # raise IOError

                #Deep CNN -------------------------------------------------------------
                if X[i]['balance'] is not None:
                    train_paths,test_paths,valid_paths = utils.partition_train_test_valid(in_dir,class_idx,sub_sample=X[i]['balance'],split=split)
                else:
                    train_paths, test_paths, valid_paths = utils.partition_train_test_valid(in_dir, class_idx,split=split)
                print('%s training and %s test images being used'%(len(train_paths),len(test_paths)))

                test_generator  = utils.load_data_generator(test_paths,class_idx,
                                                            batch_size=X[i]['batch_size'],
                                                            gray_scale=gray_scale)

                # test_generator = DataGen(test_paths,class_idx,
                #                           batch_size=X[i]['batch_size'],
                #                           gray_scale=gray_scale)

                eval_generator  = utils.load_data_generator(valid_paths,class_idx,
                                                            batch_size=len(valid_paths),
                                                            gray_scale=gray_scale)

                # eval_generator = DataGen(test_paths,class_idx,
                #                          batch_size=X[i]['batch_size'],
                #                          gray_scale=gray_scale)

                train_generator = utils.load_data_generator(train_paths,class_idx,
                                                            batch_size=X[i]['batch_size'],
                                                            gray_scale=gray_scale)

                # train_generator = DataGen(train_paths,class_idx,
                #                          batch_size=X[i]['batch_size'],
                #                          gray_scale=gray_scale)

                H = model.fit_generator(train_generator,
                                        steps_per_epoch=len(train_paths)//X[i]['batch_size'],
                                        validation_data=test_generator,
                                        validation_steps=len(test_paths)//X[i]['batch_size'],
                                        epochs=X[i]['epochs'],verbose=0,workers=1)


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
                if best_score<S['score']:
                    print('*** new best score detected, saving the model file ***')
                    shps      = '%sx%sx%s'%(shapes[0][0],shapes[0][1],shapes[0][2])
                    title     = 'class=%s cmx=%s batch=%s gray=%s in:%s'%\
                                (class_partition,X[i]['cmx'],X[i]['batch_size'],gray_scale,shps)
                    plt_path  = 'class_%s.cmx_%s.batch_%s.gray_%s.in_%s'%\
                                (class_partition,X[i]['cmx'],X[i]['batch_size'],str(gray_scale)[0],shps)
                    for png in glob.glob(out_dir+'/*.png'): os.remove(png)
                    utils.plot_train_test(H.history,title,out_path=out_dir+'/acc_loss.%s.png'%plt_path)
                    utils.plot_confusion_heatmap(CM[i],title,out_path=out_dir+'/conf_mat.%s.png'%plt_path)
                    best_score = S['score']
                    with open(best_score_path,'w') as f: json.dump(S,f)
                    model_path = out_dir+'/model.'+class_partition+'.hdf5'
                    model.save(model_path)
            except Exception as E:
                print('error occured: %s'%E)
        t_stop = time.time()
        print('total time was %s or %s per iteration'%(round(t_stop-t_start,2),round((t_stop-t_start)/(len(CM)*1.0),2)))
