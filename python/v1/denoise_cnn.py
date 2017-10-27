# 2017-10-27
# Trung-Hieu Tran @ IPVS

import os
import time
import numpy as np
import tensorflow as tf
import scipy.io as sio
from utils import *
import preprocess_images as pimages




class SDCNN(object): # Simple Denosing CNN
    def __init__(self,sess,flags):
        self.sess = sess
        self.batchSize = flags.batchSize
        self.patchSize = flags.patchSize
        self.dataset = flags.dataset
        self.checkpointDir = flags.checkpointDir
        self.dataDir = flags.dataDir
        self.overlapSize = flags.overlapSize
        self.learningRate = flags.learningRate
        self.logDir = flags.logDir

        data = time.strftime('%d%m')
        self.date = data
        self.build_model()
        self.count = 0

    def build_model(self):
        # Input
        self.input = tf.placeholder(tf.float32,[self.batchSize,self.patchSize,self.patchSize,1],
                                    name='input_noisy')
        # GT
        self.gt   = tf.placeholder(tf.float32,[self.batchSize,self.patchSize,self.patchSize,1],
                                    name='output_gt')

        # Define CNN
        with tf.variable_scope("denoising_net"):
            self.output = self.denoising_net(self.input)


        # Define MSE for error back-propagation
        self.MSE = tf.reduce_mean(tf.square(tf.subtract(self.output,self.gt)))

        self.saver = tf.train.Saver(max_to_keep=1)

    def train(self,flags):
        print("... Start Traning ...")
        globalStep1 = tf.Variable(0, name='global_step_train1',trainable=False)

        optimizerDenoising = tf.train.AdamOptimizer(flags.learningRate).minimize(self.MSE,global_step=globalStep1)

        t_vars = tf.trainable_variables()
        self.var_one = [var for var in t_vars if 'one' in var.name]
        self.var_two = [var for var in t_vars if 'two' in var.name]
        self.var_last= [var for var in t_vars if 'last' in var.name]

        init_op = tf.global_variables_initializer()



        #load trainging variables
        needInit = False
        if self.load_net(self.checkpointDir, self.dataset):
            print(' Successful loading pretrained network')
        else:
            print(' Fail loading pretrained network')
            needInit = True
            self.sess.run(init_op)

        # load training data
        gtPatches, noisyPatches = pimages.get_training_patches(self.dataDir, self.dataset, self.patchSize, self.overlapSize)
        noPatches = gtPatches.shape[0]
        noEval = int(math.floor(noPatches*0.01))
        noTrain = noPatches - noEval
        print(" Using %d patches for Training and %d patches for Evaluation"%(noTrain,noEval))

        gtpTrain = gtPatches[0:noTrain,:,:]
        noisypTrain = noisyPatches[0:noTrain,:,:]

        gtpEval = gtPatches[noTrain:,:,:]
        noisypEval = noisyPatches[noTrain:,:,:]
        noBatches = int(math.floor(noTrain/self.batchSize))
        for epoch in range(100000):
            # open log file

            if epoch ==0:
                logTrainFile = open(os.path.join(self.logDir,self.date,'train_epoch.log'),'w')
            else:
                logTrainFile = open(os.path.join(self.logDir, self.date, 'train_epoch.log'), 'a+')
            # random permute the training set.
            ranIndex = np.random.permutation(np.asarray(range(0,noTrain)))
            noisy_train_patches = noisypTrain[ranIndex,:,:]
            gt_train_patches = gtpTrain[ranIndex,:,:]
            for bIdx in range(0,noBatches):
                idxStart = bIdx * self.batchSize
                idxEnd = idxStart + self.batchSize
                in1 = noisy_train_patches[idxStart:idxEnd,:,:]
                gt1 = gt_train_patches[idxStart:idxEnd,:,:]

                # last dim should be 1
                in1 = np.expand_dims(in1,axis=-1)
                gt1 = np.expand_dims(gt1,axis=-1)

                [v,val_MSE] = self.sess.run([optimizerDenoising,self.MSE],
                                          feed_dict={self.input: in1, self.gt: gt1})
                self.count +=1

                print('Epoch train[%2d] total MSE: %.4f \n'%(epoch,val_MSE))


            if np.mod(epoch,100) == 0:
                logTrainFile.write('epoch %06d MSE %.6f \n'%(epoch,val_MSE))
                logTrainFile.flush()
                self.save_net(self.checkpointDir,self.dataset,0)

            logTrainFile.close()

    def load_net(self, checkpointDir, dataset):
        savedDir = os.path.join(checkpointDir, dataset)
        checkpointState = tf.train.get_checkpoint_state(savedDir)
        if checkpointState and checkpointState.model_checkpoint_path:
            checkpointName = os.path.basename(checkpointState.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpointDir, checkpointName))
            return True
        else:
            return False

    def save_net(self, checkpointDir, dataset, step):
        savedDir = os.path.join(checkpointDir, dataset)
        if not os.path.exists(savedDir):
            os.makedirs(savedDir)
        self.saver.save(self.sess, savedDir, global_step=step)

    def denoising_net(self,input_):
       with tf.variable_scope(tf.get_variable_scope()):
           print(" variable_scope_is %s"%tf.get_variable_scope()._name)
           h1 = self.conv2d(input_, output_dim=32 , k_h=5, k_w=5, d_h=1, d_w=1, padding='SAME',name='one')
           h1 = tf.nn.relu(h1)
           h2 = self.conv2d(h1,16,k_h=1,k_w=1,d_h=1,d_w=1,padding='SAME',name='two')
           h2 = tf.nn.relu(h2)
           h3 = self.conv2d(h2,1,k_h=3,k_w=3,d_h=1,d_w=1,padding='SAME',name = 'last')
           return h3


    def conv2d(self,input_, output_dim,
               k_h, k_w, d_h, d_w, stddev=0.02, padding='SAME',
               name="conv2d"):
        with tf.variable_scope(name):
            print("variable_scope is %s" % tf.get_variable_scope()._name)
            w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
            # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
            return conv
